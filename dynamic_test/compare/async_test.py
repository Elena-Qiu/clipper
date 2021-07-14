from __future__ import print_function
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers import python as python_deployer
import json
import requests
import time
import random
import signal
import sys
import asyncio


# Stop Clipper on Ctrl-C
def signal_handler(signal, frame):
    print("Stopping Clipper...")
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.stop_all()
    sys.exit(0)


async def sleep_func(inputs):
    for input in inputs:
        sleep_time = input[0]
        await asyncio.sleep(sleep_time/1000)
    return inputs


async def predict(addr, x):
    url = "http://%s/random-sleep/predict" % (addr)
    req_json = json.dumps({'input': x})
    headers = {'Content-type': 'application/json'}
    start = time.perf_counter()
    r = requests.post(url, headers=headers, data=req_json)
    end = time.perf_counter()
    latency = (end - start) * 1000.0
    print("'%s', %f ms" % (r.text, latency))
    
async def query():
    signal.signal(signal.SIGINT, signal_handler)
    clipper_conn = ClipperConnection(DockerContainerManager(use_centralized_log=False))
    # clipper_conn.stop_all()
    clipper_conn.start_clipper()
    python_deployer.create_endpoint(clipper_conn, "random-sleep", "floats",
                            sleep_func, slo_micros=1000000)
    await asyncio.sleep(6)
    try:
        for _ in range(10):
            sleep_time = random.random()*200
            print("Sleep time", sleep_time)
            task = [asyncio.create_task(predict(clipper_conn.get_query_addr(), [sleep_time]))]
            await asyncio.wait(task)
            await asyncio.sleep(0.2)
        clipper_conn.stop_all()
    except Exception as e:
        print(e)
        clipper_conn.stop_all()

if __name__ == '__main__':
    asyncio.run(query())
