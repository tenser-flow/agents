import os
import argparse
import datetime
import time
import struct
from tqdm import tqdm

import numpy as np
import tensorflow as tf

print (tf.__version__)
print (tf.__file__)
from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, cbt_global_iterator
from util.logging import TimeLogger

import gym
#import csv
import json
import collections

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()
#np.set_printoptions(threshold=np.inf)

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
VISUAL_OBS_SPEC = [210,160,3]
NUM_ACTIONS=2
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(512,)
LEARNING_RATE=0.00042
EPSILON = 0.5

num_iterations = 20000 # @param {type:"integer"}
initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}
batch_size = 64  # @param {type:"integer"}
log_interval = 200  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='breakout-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='breakout')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=2)
    parser.add_argument('--num-episodes', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=5)
    parser.add_argument('--log-time', default=False, action='store_true')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)
    cbt_batcher = cbt_table.mutations_batcher(flush_count=args.num_episodes, max_row_bytes=500000000)

    #INITIALIZE ENVIRONMENT
    print("-> Initializing Gym environement...")
    env = tf_py_environment.TFPyEnvironment(suite_gym.load('Breakout-v0'))
    print("-> Environment intialized.")

    #Initialize Q_Network
    q_net = q_network.QNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=FC_LAYER_PARAMS)

    """ Now use tf_agents.agents.dqn.dqn_agent to instantiate a DqnAgent. In addition to the time_step_spec, 
        action_spec and the QNetwork, the agent constructor also requires an optimizer (in this case, AdamOptimizer), 
        a loss function, and an integer step counter.
    """
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                env.action_spec())
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    #Data Collection
    #@test {"skip": true}
    def collect_step(environment, policy):
        """ Returns State, Action, Next State for one action step
        Args:
            environment : An Environment object
            policy : A collection policy
        """
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        
        #print("timestep:")
        #print(time_step)
        #print("time step type:", type(time_step))
        #print("actionstep")
        #print(action_step)
        #print("action_step type", type(action_step))
        #print("next_time_step")
        #print(next_time_step)
        #print("next_time_step type", type(next_time_step))
        return time_step, action_step, next_time_step

    def collect_data(env, policy, steps):
        """ Collects Data for N number of steps. then writes results to collection.csv file

        Args:
            env:  An Environment object
            policy: A collection policy
            steps: Maximum number of steps to collect
        """
        data = []
        Traj = collections.namedtuple('Traj', ('time_step', 'action_step', 'next_time_step'))
        for i in range(steps):
            time_step, action_step, next_time_step = collect_step(env, policy)
            d = Traj._make([time_step, action_step, next_time_step])
            #d = Traj('time_step' = time_step, 'action_step' = action_step, 'next_time_step'=next_time_step)
            #print (d.time_step)
            data.append(d)
        return data

    def write_data(data, ident):
        """
            data : trajectory objects
            id : string identifier
        print("data items")
        print ([i for i in data])
        print (data[1]._fields)
        print ("attr")
        print (getattr(data[1], 'action_step'))
        print(data[1].action_step)
        print(data[1].time_step)
        #print(data)
        """
        with open(r'collection.json', 'a') as fd:
            i = 0
            for item in data:
                print("items: ", item)
                print("ident", ident)
                #print("data[1]", item)
                items_to_json = { ident: 
                    {
                        'time_step_'+ (str(i)): {
                        "step_type": item.time_step.step_type.numpy(),
                        "reward": item.time_step.reward.numpy(),
                        "observation": item.time_step.observation.numpy(),
                        "discount" : item.time_step.discount.numpy()
                        },
                    
                    'action_step_'+ (str(i)): {
                        "action": item.action_step.action.numpy(),
                        "state": item.action_step.state,
                        "info": item.action_step.info
                        },

                    'next_time_step_'+ (str(i)):{
                        "step_type": item.time_step.step_type.numpy(),
                        "reward": item.time_step.reward.numpy(),
                        "observation": item.time_step.observation.numpy(),
                        "discount" : item.time_step.discount.numpy()
                        }
                    }
                }
                data = json.dumps(items_to_json, separators=(',', ':'), cls=NumpyEncoder)
                json.dump(data, fd)
                fd.write("\n")
                i =+ 1

    #GLOBAL ITERATOR
    global_i = cbt_global_iterator(cbt_table)
    print("global_i = {}".format(global_i))

    if args.log_time is True:
        time_logger = TimeLogger(["Collect Data" , "Serialize Data", "Write Cells", "Mutate Rows"], num_cycles=args.num_episodes)

    #COLLECT DATA FOR CBT
    print("-> Starting data collection...")
    rows = []
    for cycle in range(args.num_cycles):
        #gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)
        for i in tqdm(range(args.num_episodes), "Cycle {}".format(cycle)):
            if args.log_time is True: time_logger.reset()

            #RL LOOP GENERATES A TRAJECTORY
            data = collect_data(env, random_policy, steps=args.max_steps)
            print("data: ", data[1].action_step.action.numpy())
            write_data(data, "Rab-Agent_: " + (str(cycle)) + "_" + (str(i)))
    env.close()
    print("Done Collecting --- Reading file")

    with open(r'collection.json', 'r') as file_reader :
        json_data = [json.loads(line) for line in file_reader]
        #print (json_data)
        count = 0
        for i in json_data:
            print ("count: ", count)
            count += 1
    print("-> Done!")