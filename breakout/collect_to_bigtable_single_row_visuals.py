import os
import argparse
import datetime
import time
import struct
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info, Visual_obs
from breakout.dqn_model import DQN_Model
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, cbt_global_iterator, cbt_load_table
from util.logging import TimeLogger

import gym

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

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
VISUAL_OBS_SPEC = [210,160,3]
NUM_ACTIONS=2
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(200,)
LEARNING_RATE=0.00042
EPSILON = 0.5

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable-test')
    parser.add_argument('--cbt-table-name', type=str, default='breakout-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='breakout')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=1000000)
    parser.add_argument('--num-episodes', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--log-time', default=False, action='store_true')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)
    cbt_table_visual = cbt_load_table(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name+'visual', credentials)
    cbt_batcher = cbt_table.mutations_batcher(flush_count=args.num_episodes, max_row_bytes=10080100)
    #cbt_batcher_visual = cbt_table_visual.mutations_batcher(flush_count=args.num_episodes, max_row_bytes=10080100)
    
    #INITIALIZE ENVIRONMENT
    print("-> Initializing Gym environement...")
    
    #Custom DQN
    #env = gym.make('Breakout-v0')
    env_name = 'Breakout-v0'
    env = suite_gym.load(env_name)

    print("-> Environment intialized.")

    #LOAD MODEL
    model = DQN_Model(input_shape=env.observation_space.shape,
                      num_actions=env.action_space.n,
                      conv_layer_params=CONV_LAYER_PARAMS,
                      fc_layer_params=FC_LAYER_PARAMS,
                      learning_rate=LEARNING_RATE)

    #GLOBAL ITERATOR
    global_i = cbt_global_iterator(cbt_table)
    print("global_i = {}".format(global_i))

    if args.log_time is True:
        time_logger = TimeLogger(["0Collect Data" , "1Take Action / conv b", "2Append new obs / conv b", "3Generate Visual obs keys", "4Build pb2 objects traj", "5Write Cells visual", "6Batch visual", "7Write cells traj", "8Write cells traj"], num_cycles=args.num_episodes)

#COLLECT DATA FOR CBT
    print("-> Starting data collection...")
    rows, visual_obs_rows = [], []
    for cycle in range(args.num_cycles):
        gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)
        for i in tqdm(range(args.num_episodes), "Cycle {}".format(cycle)):
            if args.log_time is True: time_logger.reset()

            #CREATE ROW_KEY
            row_key_i = i + global_i + (cycle * args.num_episodes)
            row_key = '{}_trajectory_{}'.format(args.prefix,row_key_i).encode()

            #RL LOOP GENERATES A TRAJECTORY
            observations, actions, rewards = [], [], []

            obs = np.asarray(env.reset()).astype(np.dtype('b'))

            reward = 0
            done = False

            if args.log_time is True: time_logger.log(0)
            for i in range(args.max_steps):
                action = model.step_epsilon_greedy((obs.astype(int) / 255).astype(float), EPSILON)
                if args.log_time is True: time_logger.log(1)
                new_obs, reward, done, info = env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                if args.log_time is True: time_logger.log(2)
                obs = np.asarray(new_obs).astype(bytes)
                if args.log_time is True: time_logger.log(3)
            visual_obs_keys = ['{}_visual_{}'.format(row_key, x) for x in range(len(observations))]
            if args.log_time is True: time_logger.log(4)
            

            #BUILD PB2 OBJECTS
            traj, info, visual_obs = Trajectory(), Info(), []
            traj.visual_obs_key.extend(visual_obs_keys)
            traj.actions.extend(actions)
            traj.rewards.extend(rewards)
            info.vector_obs_spec.extend(observations[0].shape)
            info.num_steps = len(actions)
            index = 0
            if args.log_time is True: time_logger.log(5)
            for ob in observations:
                visual_ob = Visual_obs()
                visual_ob.data.extend(np.asarray(ob).flatten().astype(bytes))
                row = cbt_table_visual.row(visual_obs_keys[index])
                index += 1
                row.set_cell(column_family_id='trajectory',
                            column='data'.encode(),
                            value=visual_ob.SerializeToString())
                visual_obs_rows.append(row)
            if args.log_time is True: time_logger.log(6)
            
            #BATCH VISUAL OBS
            response = cbt_table_visual.mutate_rows(visual_obs_rows)
            visual_obs_rows = []
            if args.log_time is True: time_logger.log(7)

            #WRITE TO AND APPEND ROW
            row = cbt_table.row(row_key)
            row.set_cell(column_family_id='trajectory',
                        column='traj'.encode(),
                        value=traj.SerializeToString())
            row.set_cell(column_family_id='trajectory',
                        column='info'.encode(),
                        value=info.SerializeToString())
            rows.append(row)
            if args.log_time is True: time_logger.log(8)
        gi_row = cbt_table.row('global_iterator'.encode())
        gi_row.set_cell(column_family_id='global',
                        column='i'.encode(),
                        value=struct.pack('i',row_key_i+1),
                        timestamp=datetime.datetime.utcnow())
        rows.append(gi_row)
        cbt_table.mutate_rows(rows)
        if args.log_time is True: time_logger.print_logs()
        rows = []
        print("-> Saved trajectories {} - {}.".format(row_key_i - (args.num_episodes-1), row_key_i))
    env.close()
    print("-> Done!")
