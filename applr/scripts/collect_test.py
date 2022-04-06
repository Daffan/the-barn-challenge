from os.path import exists, join
import json
import os
import numpy as np
import time
import pickle
import sys
import yaml
import logging
import argparse

def main(buffer_path):
    f = open(join(buffer_path, 'config.yaml'), 'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    num_trials = config["condor_config"]["num_trials"]
    ep_lengths = []
    times = []
    successes = []
    collisions = []
    recoveries = []
    flag = True

    save_path = join(buffer_path, "test_result.txt")
    outf =  open(save_path, "w")
    worlds = []
    bad_worlds = []
    for dirname, dirnames, filenames in os.walk(buffer_path):
        for filename in filenames:
            p = join(dirname, filename)
            if p.endswith('.pickle'):
                try:
                    with open(p, 'rb') as f:
                        traj = pickle.load(f)
                    world = traj[-1][-1]['world']
                    if isinstance(world, str):
                        world = int(world.split("_")[-1].split(".")[0])
                    ep_return = sum([t[2] for t in traj])
                    ep_length = len(traj)
                    success = int(traj[-1][-1]['success'])
                    time = float(traj[-1][-1]['time'])
                    collision = sum([int(s[-1]["collision"]) for s in traj])
                    recovery = float(traj[-1][-1]['recovery'])
                    
                    if len(filenames) == num_trials and world not in worlds:
                        outf.write("%d %d %f %d %.2f %d %.2f\n" %(world, ep_length, ep_return, success, time, collision, recovery))
                        if world not in [54, 94, 156, 68, 52, 101, 40, 135, 51, 42, 75, 67, 18, 53, 87, 36, 28, 61, 233, 25, 35, 20, 34, 79, 108, 46, 65, 90, 6, 73, 70, 10, 29, 167, 15, 31, 77, 116, 241, 155, 194, 99, 56, 149, 38, 261, 239, 234, 60, 173, 247, 178, 291, 16, 9, 21, 169, 257, 148, 296, 151, 259, 102, 145, 130, 205, 121, 105, 43, 242, 213, 171, 62, 202, 293, 224, 225, 152, 111, 55, 125, 200, 161, 1, 136, 106, 286, 139, 244, 230, 222, 238, 170, 267, 26, 132, 124, 23, 59, 3, 97, 119, 89, 12, 164, 39, 236, 263, 81, 188, 84, 11, 268, 192, 122, 22, 253, 219, 216, 137, 85, 195, 206, 212, 4, 274, 91, 248, 44, 131, 203, 63, 80, 37, 110, 50, 74, 120, 128, 249, 30, 14, 103, 49, 154, 82, 2, 143, 158, 147, 235, 83, 157, 142, 187, 185, 288, 45, 140, 271, 160, 146, 109, 223, 126, 98, 252, 134, 272, 115, 71, 117, 255, 141, 174, 33, 245, 92, 295, 281, 186, 260, 7, 166, 196, 66, 113, 153, 227, 107, 199, 298, 278, 114, 72, 165, 228, 176, 24, 162, 198, 180, 285, 232, 243, 207, 190, 262, 275, 172, 179, 269, 127, 86, 183, 273, 287, 215, 266, 95, 5, 299, 279, 13, 250, 96, 197, 177, 58, 289, 211, 220, 182, 282, 210, 280, 251, 283, 217, 276, 292, 221, 204, 191, 181, 209, 297, 264, 231, 254]:
                            ep_lengths.append(float(ep_length))
                            times.append(time)
                            successes.append(success)
                            collisions.append(collision)
                            recoveries.append(recovery)
                    else:
                        break
                except:
                    logging.exception("")
                    pass

        if dirname.split("/")[-1].startswith("actor"):
            if len(filenames) == num_trials:
                worlds.append(world)
            elif world not in bad_worlds:
                bad_worlds.append(world)
            else:
                print("world %s fail for all two test!" %(world))
                flag = False

    outf.close()
    # if flag:
    print("Test finished!")
    print("Find the test result under %s" %(save_path))
    print("Quick report: avg ep_len %.2f, avg time: %.2f, success rate: %.2f, collision: %.2f, recovery: %.2f" %(sum(ep_lengths)/len(ep_lengths), sum(times)/len(times), sum(successes)/len(successes), sum(collisions)/len(collisions), sum(recoveries)/len(recoveries)))
    # else:
    # print("Some tests are still running")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description = 'collect the test result')
    parser.add_argument('--buffer_path', dest='buffer_path', type = str)
    buffer_path = parser.parse_args().buffer_path
    main(buffer_path)
