import argparse
import numpy as np
from collections import defaultdict

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description = 'collect the test result')
    parser.add_argument('--path', dest='path', type = str)
    args = parser.parse_args()
    
    time = defaultdict(list)
    success = defaultdict(list)
    recovery = defaultdict(list)

    with open(args.path, "r") as f:
        for l in f.readlines():
            l = l.split(" ")
            h = int(l[0])
            if int(l[-1]):
                time[h].append(float(l[2])) 
            success[h].append(int(l[-1]))
            recovery[h].append(int(l[-2]))
    avg_time = []
    avg_success = []
    for k in sorted(time.keys()):
        print("habitat: %d, time: %.4f, success: %.2f" %(k, np.mean(time[k]), np.mean(success[k])))
        avg_time.append(np.mean(time[k]))
        avg_success.append(np.mean(success[k]))
    print("Average time: %.4f, average success: %.2f" %(np.mean(avg_time), np.mean(avg_success)))
