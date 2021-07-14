import pandas as pd
import random
import math

def gen_requests(total_batches, batch_size, min_dur, max_dur, seed, output_file):
    total_requests = total_batches * batch_size
    job_id_list = range(total_requests)
    gen = random.Random(seed)
    length_list = [gen.uniform(min_dur, max_dur) for _ in job_id_list]
    df = pd.DataFrame({"Job_id": job_id_list, "Length": length_list})
    df.to_csv(output_file, index=False)
    print("Requests saved at '" + output_file + "'!")
    return df


def read_requests(filename):
    df = pd.read_csv(filename)
    return df


def gen_duration(duration, maximum=10, seed=12345):
    gen = random.Random(seed)
    for _ in range(maximum):
        c = gen.uniform(0, 1)
        yield duration * - math.log(c)


def get_full_class_name(obj):
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + '.' + obj.__class__.__name__