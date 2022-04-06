################################################################################
# The script generates submission files and submit them to HTCondor.
# Submission file only has executable/actor.sh
# We now prefer to run central node locally, because central learner on the 
# computing node is not stable. Idle node will destory the central learner
################################################################################

import subprocess
import yaml
import os
from os.path import join
import time
import argparse
import uuid
import shutil

parser = argparse.ArgumentParser(description = 'test on condor cluster')
parser.add_argument('--model_dir', dest='model_dir', type = str)

model_dir = parser.parse_args().model_dir

print(">>>>>>>> Loading the model from %s" %(model_dir))

# Load condor config
CONFIG_PATH = join(model_dir, "config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# We test each test world with two actors, so duplicate the lict by a factor of two
num_actor = len(config["condor_config"]["test_worlds"]) * 2
num_trials = config["condor_config"]["num_trials"]

# Create buffer folder
hash_code = uuid.uuid4().hex
# buffer_path = os.path.join(os.environ['HOME'], hash_code)
buffer_path = os.path.join("/scratch/cluster/zifan", hash_code)
os.environ['BUFFER_PATH'] = buffer_path
if not os.path.exists(buffer_path):
    os.mkdir(buffer_path)

# Copy the model files
shutil.copyfile(
    join(model_dir, "config.yaml"), 
    join(buffer_path, "config.yaml")    
)
shutil.copyfile(
    join(model_dir, "policy_actor"), 
    join(buffer_path, "policy_actor")
)
shutil.copyfile(
    join(model_dir, "policy_noise"), 
    join(buffer_path, "policy_noise")
)
# Set the exploration noise to be 0
with open(join(buffer_path, 'eps.txt'), 'w') as f:
    f.write(str(0))

# Create folders for HTCondor logging files
out_path = "out"
out_path = join(buffer_path, out_path)
print("Find the logging under path: %s" %(out_path))
if not os.path.exists(out_path):
    os.mkdir(out_path)

# Tester submission
submission_file = join(buffer_path, 'testers.sub')
cfile = open(submission_file, 'w')
s = 'executable/tester.sh'
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

"""
# Add tester arguments
for a in range(num_actor):
    run_command = "\
        arguments  = --id %d\n\
        output     = %s/out_%d.txt\n\
        log        = %s/log_%d.txt\n\
        error      = %s/err_%d.txt\n\
        queue 1\n\n" % (a, out_path, a, out_path, a, out_path, a)
    cfile.write(run_command)
cfile.close()
"""

# Add tester arguments
for a in range(num_actor):
    run_command = "\
        arguments  = --id %d\n\
        queue 1\n\n" % (a)
    cfile.write(run_command)
cfile.close()

subprocess.run(["condor_submit", submission_file])
