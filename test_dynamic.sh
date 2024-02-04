#!/bin/bash
for i in {300..359} ; do
    for j in {1..10} ; do            
        # run the test
        python3 run.py --world_idx $i
        sleep 5
    done
done