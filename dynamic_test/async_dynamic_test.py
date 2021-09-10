from __future__ import print_function

import sys
import time
import traceback
import json
import csv
from typing import NamedTuple, Optional


def get_full_class_name(obj):
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:  # type: ignore
        return obj.__class__.__name__
    return module + '.' + obj.__class__.__name__


def fake_model(batch):
    '''Model runs inside clipper, not async'''
    print('fake_model: serving batch', batch)

    # the runtime of a batch is max(batch)
    latency_ms = max(sample[0] for sample in batch)
    # copy input to output
    output = [str(sample[0]) for sample in batch]

    # sleep
    now = time.perf_counter()
    target = now + latency_ms / 1000
    while time.perf_counter() < target:
        pass

    print('fake_model: returning', output)
    return output


async def setup_clipper(args):
    '''Setup the clipper cluster, returns the clipper connection and the endpoint url'''
    import asyncio
    import aiohttp
    from clipper_admin import ClipperConnection, DockerContainerManager
    from clipper_admin.exceptions import ClipperException
    from clipper_admin.deployers import python as python_deployer

    clipper_conn = ClipperConnection(DockerContainerManager(use_centralized_log=False))

    try:
        # start or connect to the cluster
        try:
            # this blocks until the cluster is ready
            clipper_conn.start_clipper(cache_size=0)
        except ClipperException:
            clipper_conn.connect()

        # deploy the model and register the application
        # this blocks until the model is ready
        name = 'fake-model'
        python_deployer.create_endpoint(clipper_conn, name, "floats", fake_model, slo_micros=args.slo_us, batch_size=args.batch_size)

        # wait a few second for the model container to stablize
        await asyncio.sleep(2)

        retry = 3
        # wait for replicas to spin up for 3s
        while retry > 0:
            if clipper_conn.get_num_replicas(name) > 0:
                break
            print('INFO: waiting for replicas to spin up', file=sys.stderr)
            await asyncio.sleep(1)
            retry -= 1
        else:
            # something wrong
            print('ERROR: replicas take too long to spin up, possibly died. Check container log', file=sys.stderr)
            raise TypeError('Bad python model')

        # endpoint url
        clipper_conn.get_query_addr()
        endpoint = f"http://{clipper_conn.get_query_addr()}/fake-model/predict"

        # wait for container to be ready
        async with aiohttp.ClientSession() as http_client:
            retry = 10
            while retry > 0:
                try:
                    await predict(http_client, endpoint, 1.0)
                    break
                except:
                    print('INFO: waiting for ready to serve', file=sys.stderr)
                    await asyncio.sleep(1)
                    retry -= 1
            else:
                # something wrong
                print('ERROR: replicas take too long to spin up, possibly died. Check container log', file=sys.stderr)
                raise TypeError('Bad python model')

        print('INFO: ready to go', file=sys.stderr)

        return clipper_conn, endpoint
    except Exception as e:
        # cleanup if error
        print('ERROR: error when starting clipper, clean up', file=sys.stderr)
        clipper_conn.stop_all()
        raise e


class ClipperReply(NamedTuple):
    ok: bool
    status: int
    length_us: Optional[float] = None
    error_kind: Optional[str] = None
    error_msg: Optional[str] = None


async def predict(http_client, endpoint, length_ms):
    async with http_client.post(endpoint, json={'input': [length_ms]}) as r:
        body = await r.json()
        if r.ok:
            try:
                got_length_us = float(body['output'])
                if got_length_us - length_ms * 1000.0 > 10000000:
                    print(f'WARNING: fetching {length_ms:.3f} ms but got {got_length_us:.3f} us', file=sys.stderr)
                    got_length_us = length_ms * 1000.0
            except Exception:
                got_length_us = length_ms * 1000.0
            return ClipperReply(r.ok, r.status, got_length_us, None, None)
        else:
            return ClipperReply(r.ok, r.status, length_ms * 1000.0, body['error'], body['cause'])


async def fetch(now_ms, length_ms, http_client, endpoint, args):
    try:
        if length_ms is not None:
            print(f'INFO: at {now_ms:.3f} ms fetching {length_ms:.3f} ms', file=sys.stderr)
        else:
            print(f'INFO: at {now_ms:.3f} ms fetching None ms', file=sys.stderr)

        started = time.perf_counter()
        reply = await predict(http_client, endpoint, length_ms)
        # measured latency
        latency_us = (time.perf_counter() - started) * 1000000

        # output csv
        if reply.ok:
            if latency_us > args.slo_us:
                args.print(f'{now_ms},{reply.length_us},{latency_us},past_due_client,{reply.error_kind},{reply.error_msg}')
            else:
                args.print(f'{now_ms},{reply.length_us},{latency_us},done,,')
        else:
            args.print(f'{now_ms},{reply.length_us},{latency_us},error,{reply.error_kind},{reply.error_msg}')
    except Exception as e:
        ename = get_full_class_name(e)
        args.print(f'{now_ms},,,error,Internal,{ename}')
        if args.debug:
            print('Error: ', traceback.format_exc(), file=sys.stderr)
            raise e


def incoming_file(filename: str):
    """read delay and length from csv file
    yields (delay_ms, length_ms)
    """
    with open(filename) as f:
        reader = csv.DictReader(row for row in f if not row.startswith('#'))
        jobs = [
            (float(row['Admitted']), float(row['Length']))
            for row in reader
        ]
    # jobs has to be sort by admitted
    jobs.sort()
    # take note of current time
    now = 0
    batch = []
    for admitted, length_ms in jobs:
        delay_ms = admitted - now
        if delay_ms > 0:
            yield batch, delay_ms
            now = admitted
            batch = []
        batch.append(length_ms)
    yield batch, 0


async def queryer(endpoint, args):
    import aiohttp
    import asyncio

    # csv header
    args.print('Timestamp,LengthUS,LatencyUS,State,EName,ECause')

    async with aiohttp.ClientSession() as http_client:
        # start fetching
        incoming = incoming_file(args.reqs)
        flying = []
        base_ms = time.perf_counter() * 1000
        get_time = lambda: time.perf_counter() * 1000 - base_ms
        print('INFO: rock and roll', file=sys.stderr)
        for lengths, delay_ms in incoming:
            now_ms = get_time()

            lengths_str = ', '.join(['{:.3f}'.format(l) for l in lengths])
            print(f'INFO: at {now_ms:.3f} ms batch [{lengths_str}] delay {delay_ms:.3f} ms', file=sys.stderr)

            # fire current request
            if lengths:
                flying.extend(
                    asyncio.create_task(fetch(now_ms, length_ms, http_client, endpoint, args))
                    for length_ms in lengths
                )

            remaining_ms = delay_ms
            while remaining_ms > 0:
                try:
                    # use remaining time to do some book keeping
                    remaining_ms = delay_ms - (get_time() - now_ms)
                    if remaining_ms > 5 and flying:
                        # book keeping
                        done, pending = await asyncio.wait(flying, timeout=0.001)
                        flying = list(pending)
                        # re-raise any exception if debug
                        if args.debug:
                            for r in done:
                                r.result()

                    remaining_ms = delay_ms - (get_time() - now_ms)
                    # wait until delay_ms
                    if remaining_ms > 0:
                        # wait for 1/5 remaining time
                        #await asyncio.sleep(remaining_ms / 5 / 1000)
                        remaining_ms = delay_ms - (get_time() - now_ms)
                    else:
                        if remaining_ms < -5:
                            print(f'WARNING: bookkeeping for too long: {remaining_ms}ms', file=sys.stderr)
                            continue
                    if remaining_ms < -5:
                        print(f'WARNING: slept for too long: {remaining_ms}ms', file=sys.stderr)
                        continue
                finally:
                    pass
        # wait for any flying requests to finish
        done, pending = await asyncio.wait(flying)
        # re-raise any exception if debug
        if args.debug:
            for r in done:
                r.result()
        print('INFO: done', file=sys.stderr)


async def amain():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Show response error", default=False)
    parser.add_argument("--pause", action="store_true", help="pause after setup cluster", default=False)
    parser.add_argument("--output", type=str, help="Output directory", default="output")
    parser.add_argument("--slo_us", type=int, help="SLO in microseconds", default=250000)
    parser.add_argument("--batch_size", type=int, help="max batch size", default=16)

    parser.add_argument("reqs", type=str, help="Request schedule csv file")

    args = parser.parse_args()

    with open(args.output + '/output.csv', 'w') as f:

        def printer(*args, **kwargs):
            print(*args, **{'file': f, **kwargs})
            f.flush()

        printer('# ' + json.dumps(vars(args)))
        args.print = printer

        clipper_conn, endpoint = await setup_clipper(args)
        if args.pause:
            try:
                print('Pausing')
                input()
            except KeyboardInterrupt:
                return

        try:
            await queryer(endpoint, args)
            clipper_conn.get_clipper_logs(args.output + '/logs/')
        finally:
            print('INFO: stop clipper')
            clipper_conn.stop_all()


def main():
    import asyncio
    asyncio.run(amain())


if __name__ == '__main__':
    main()
