#!/bin/bash
for i in {0..49} ; do
    n=`expr $i \* 6` # 50 test BARN worlds with equal spacing indices: [0, 6, 12, ..., 294]
        for j in {1..10} ; do            
            # run the test
            python3 run.py --world_idx $n
            sleep 5
        done
done