#!/bin/bash
singularity exec -i --nv -n --network=none -p -B `pwd`:/jackal_ws/src/the-barn-challenge ${1} /bin/bash /jackal_ws/src/the-barn-challenge/entrypoint.sh ${@:2}
