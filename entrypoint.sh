#!/bin/bash
source /jackal_ws/devel/setup.bash
cd /jackal_ws/src/barn-competition-icra
exec ${@:1}
