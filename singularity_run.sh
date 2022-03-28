#!/bin/bash
singularity exec -i --nv -n --network=none -p ${1} /bin/bash /jackal_ws/src/nav-competition-icra2022/entrypoint.sh ${@:2}
