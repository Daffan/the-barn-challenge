################################################################################
# The script generates submission files and submit them to HTCondor.
# Submission file only has executable/actor.sh
# We now prefer to run central node locally, because central learner on the 
# computing node is not stable. Idle node will destory the central learner
################################################################################

import subprocess
import yaml
import os
import time
import uuid
import argparse

parser = argparse.ArgumentParser(description = 'Start condor training')
parser.add_argument('--config_path', dest='config_path', default="../configs/config.ymal")
parser.add_argument('--local_update', dest='local_update', action="store_true")
args = parser.parse_args()

# Load condor config
CONFIG_PATH = args.config_path

if not os.getenv('BUFFER_PATH'):
    hash_code = uuid.uuid4().hex[:8]
    buffer_path = os.path.join(os.environ['HOME'], hash_code)
    os.environ['BUFFER_PATH'] = buffer_path
else:
    buffer_path = os.environ['BUFFER_PATH']

if not os.path.exists(buffer_path):
    os.mkdir(buffer_path)

out_path = "out"
out_path = os.path.join(os.environ['BUFFER_PATH'], out_path)
print("Find the logging under path: %s" %(out_path))
if not os.path.exists(out_path):
    os.mkdir(out_path)

if not args.local_update:
    # Central learner submission
    submission_file = os.path.join(buffer_path, 'central_learner.sub')
    cfile = open(submission_file, 'w')
    s = 'executable/run_central_learner.sh'
    common_command = "\
        requirements       = InMastodon \n\
        +Group              = \"GRAD\" \n\
        +Project            = \"AI_ROBOTICS\" \n\
        +ProjectDescription = \"Adaptive Planner Parameter Learning From Reinforcement\" \n\
        Executable          = %s \n\
        Universe            = vanilla\n\
        getenv              = true\n\
        transfer_executable = false \n\n" %(s)
    cfile.write(common_command)

    run_command = "\
        arguments  = %s\n\
        output     = %s/out.txt\n\
        log        = %s/log.txt\n\
        error      = %s/err.txt\n\
        queue 1\n\n" % (args.config_path, out_path, out_path, out_path)
    cfile.write(run_command)

    cfile.close()

    subprocess.run(["condor_submit", submission_file])

    time.sleep(60)  # wait for central learner to be initialized 

# Actor submission
with open(os.path.join(buffer_path, "config.yaml"), 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
num_actor = config["condor_config"]["num_actor"]

submission_file = os.path.join(buffer_path, 'actors.sub')
cfile = open(submission_file, 'w')
s = 'executable/actor.sh'
common_command = "\
    requirements       = InMastodon \n\
    +Group              = \"GRAD\" \n\
    +Project            = \"AI_ROBOTICS\" \n\
    +ProjectDescription = \"Adaptive Planner Parameter Learning From Reinforcement\" \n\
    Executable          = %s \n\
    Universe            = vanilla\n\
    getenv              = true\n\
    transfer_executable = false \n\n" %(s)
cfile.write(common_command)
'''
# Add actor arguments
for a in range(num_actor):
    run_command = "\
        arguments  = %d\n\
        output     = %s/out_%d.txt\n\
        log        = %s/log_%d.txt\n\
        error      = %s/err_%d.txt\n\
        queue 1\n\n" % (a, out_path, a, out_path, a, out_path, a)
    cfile.write(run_command)
cfile.close()
'''
# Add actor arguments
for a in range(num_actor):
    run_command = "\
        arguments  = %d\n\
        queue 1\n\n" % (a)
    cfile.write(run_command)
cfile.close()

subprocess.run(["condor_submit", submission_file])
