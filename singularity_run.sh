#!/bin/bash
singularity exec -i --nv -n --network=none -p -B `pwd`:/jackal_ws/src/barn-competition-icra ${1} /bin/bash /jackal_ws/src/barn-competition-icra/entrypoint.sh ${@:2}
