from __future__ import print_function
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers import python as python_deployer
import json
from pandas.io.parsers import read_csv
import requests
import time
import signal
import sys
import argparse
from utils import *


# Stop Clipper on Ctrl-C
def signal_handler(signal, frame):
    print("Stopping Clipper...")
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.stop_all()
    sys.exit(0)


def sleep_func(inputs):
    output = []
    for input in inputs:
        sleep_time = input[0]
        start = time.perf_counter()
        time.sleep(sleep_time/1000)
        end = time.perf_counter()
        output.append(str((end-start)*1000.0))
    return output


def predict(addr, x, batch=False):
    url = "http://%s/random-sleep/predict" % (addr)
    if batch:
        req_json = json.dumps({'input_batch': x})
    else:
        req_json = json.dumps({'input': x})
    headers = {'Content-type': 'application/json'}
    start = time.perf_counter()
    r = requests.post(url, headers=headers, data=req_json)
    end = time.perf_counter()
    latency = (end - start) * 1000.0
    print("'%s', %f ms" % (r.text, latency))
    return latency


def queryer(req_df, durs, batch_size):
    # deploy the model and register the application
    signal.signal(signal.SIGINT, signal_handler)
    clipper_conn = ClipperConnection(DockerContainerManager(use_centralized_log=False))
    clipper_conn.start_clipper()
    python_deployer.create_endpoint(clipper_conn, "random-sleep", "floats",
                            sleep_func, slo_micros=1000000)

    # wait for the docker to initialize
    time.sleep(6)

    # insert one column of "Latency"
    req_df.insert(2, "Latency", 0)

    try:
        for batch_id, dur in enumerate(durs):
            start_time = time.perf_counter() * 1000
            if batch_size > 1:
                start_index = batch_id * batch_size
                batch_df = req_df.iloc[start_index:(start_index+batch_size)]
                input_list= [[row["Length"]] for _, row in batch_df.iterrows()]
                latency = predict(clipper_conn.get_query_addr(), input_list, batch=True)
                for idx in range(start_index, start_index+batch_size):
                    req_df.loc[idx, "Latency"] = latency
            else:
                input_list = [req_df.loc[batch_id, "Length"]]
                latency = predict(clipper_conn.get_query_addr(), input_list, batch=False)
                req_df.loc[batch_id, "Latency"] = latency
            remaining_dur = dur - (time.perf_counter()*1000 - start_time)
            if remaining_dur > 0:
                time.sleep(remaining_dur/1000)
            else:
                print("Warning: Duration between requests is too narrow: " + str(remaining_dur))
        clipper_conn.stop_all()
        return req_df
    except Exception as e:
        error = get_full_class_name(e)
        print("Error: " + str(error))
        clipper_conn.stop_all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request_file", type=str, help="Request schedule csv file", default=None)
    parser.add_argument("--dur", type=int, help="Duration between posting two request batches, in ms", default=500)
    parser.add_argument("--min_sleep", type=int, help="Minimum sleep time for a request, in ms", default=50)
    parser.add_argument("--max_sleep", type=int, help="Maximum sleep time for a request, in ms", default=200)
    parser.add_argument("--batch_size", type=int, help="Number of requests per batch", default=1)
    parser.add_argument("--max_batch", type=int, help="Max number of batches to generate", default=10)
    parser.add_argument("--seed", type=int, help="Input gen seed", default=12345)
    parser.add_argument("--output_file", type=str, help="Output request csv file", default="output.csv")

    args = parser.parse_args()
    # get requests
    if args.request_file:
        req_df = read_csv(args.request_file)
    else:
        req_df = gen_requests(args.max_batch, args.batch_size, args.min_sleep, 
                        args.max_sleep, args.seed, "requests.csv")
    
    # get durations between requests
    durs = gen_duration(args.dur, args.max_batch, args.seed)

    processed_df = queryer(req_df, durs, args.batch_size)
    if processed_df is not None:
        processed_df.to_csv("Output.csv", index=False)


if __name__ == '__main__':
    main()
