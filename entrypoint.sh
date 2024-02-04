#!/bin/bash
source /jackal_ws/devel/setup.bash
cd /jackal_ws/src/the-barn-challenge
exec ${@:1}
