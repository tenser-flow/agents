import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from breakout.dqn_model import DQN_Model
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_rows
from util.logging import TimeLogger

import json
import collections
import gin

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

#for replaybuffer class
from tf_agents.replay_buffers import replay_buffer
from tf_agents.replay_buffers import json_table
from tf_agents.specs import tensor_spec
#np.set_printoptions(threshold=np.inf)

""" Collect Breakout TF-agents DQN

"""

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
VISUAL_OBS_SPEC = [210,160,3]
NUM_ACTIONS=2
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(512,200)
LEARNING_RATE=0.00042
GAMMA = 0.9

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@gin.configurable
class Json_TFReplayBuffer(tf_uniform_replay_buffer.TFUniformReplayBuffer):
    def __init__(self,
               data_spec,
               batch_size,
               max_length=1000,
               scope='Json_TFReplayBuffer',
               device='cpu:*',
               table_fn=json_table.Table,
               dataset_drop_remainder=False,
               dataset_window_shift=None,
               stateful_dataset=False):
        #self._data_spec = data_spec
        super(Json_TFReplayBuffer, self).__init__(data_spec, batch_size)
        #super(TFUniformReplayBuffer, self).__init__(data_spec, capacity, stateful_dataset)
    
    def _write_data(self, data, ident):
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
        #print ("data", data)
        with open(r'collection-main.json', 'a') as fd:
            i = 0
            #print (dir(data))
            #print ("action_step :", data.action_step)
            #print ("action :", data.action)
            #print ("policy info", data.policy_info)
            #print ("step_type", data.step_type)
            #print("ident", ident)
            #print("data[1]", item)
            items_to_json = { ident:
                {
                    'Trajectory'+ (str(i)): {
                    "step_type": data.step_type.numpy(),
                    "observation": data.observation.numpy(),
                    "action": data.action.numpy(),
                    "policy_info": data.policy_info,
                    "next_step_type": data.next_step_type.numpy(),
                    "reward": data.reward.numpy(),
                    "discount" : data.discount.numpy()
                    },
                }
            }
            data = json.dumps(items_to_json, separators=(',', ':'), cls=NumpyEncoder)
            json.dump(data, fd)
            fd.write("\n")
            i =+ 1

    def add_batch(self, items):
        return self._add_batch_to_json(items)
        
    def _add_batch_to_json(self, items):
        tf.nest.assert_same_structure(items, self._data_spec)
        #print("items: ", items)
        with tf.device(self._device), tf.name_scope(self._scope):
            id_ = self._increment_last_id()
            write_rows = self._get_rows_for_id(id_)
            write_id_op = self._id_table.write(write_rows, id_)
            write_data_op = self._data_table.write(write_rows, items)
            #print ("items :", items)
            data = self._write_data(items, "1")
            #print ()
            #print("device: ", self._device)
            #print("name_scope: ", self._scope)
            #print("write_rows: ", write_rows)
            #print("id_", id_)
            #print ("write_id_op: ", write_id_op)
            #print ("write_data_op:", write_data_op)
            #print ("group: ", tf.group(write_id_op, write_data_op))
            return tf.group(write_id_op, write_data_op)

@gin.configurable
def collector_v1(
    #root_dir,
    env_name='Breakout-v0',
    num_iterations=100000,
    train_sequence_length=1,
    # Params for QNetwork
    fc_layer_params=(500,),
    # Params for QRnnNetwork
    input_fc_layer_params=(50,),
    lstm_size=(20,),
    output_fc_layer_params=(20,),

    # Params for collect
    initial_collect_steps=10,
    collect_steps_per_iteration=10,
    epsilon_greedy=0.1,
    replay_buffer_capacity=100,
    ):

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

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=env.batch_size,
        table_fn=json_table.Table,
        max_length=replay_buffer_capacity)
    
    print ("--------------------------------")

    initial_collect_policy = random_tf_policy.RandomTFPolicy(
        env.time_step_spec(), env.action_spec())

    dynamic_step_driver.DynamicStepDriver(
        env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps).run()

    #print ("collect_driver: ", dynamic_step_driver)

    time_step = None
    policy_state = collect_policy.get_initial_state(env.batch_size)

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='breakout-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='breakout')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--train-epochs', type=int, default=1000000)
    parser.add_argument('--train-steps', type=int, default=10)
    parser.add_argument('--period', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    parser.add_argument('--log-time', default=False, action='store_true')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)

    collector_v1(num_iterations=1)
    
    print("Done Collecting --- Reading file")
    """
    with open(r'collection.json', 'r') as file_reader :
        json_data = [json.loads(line) for line in file_reader]
        #print (json_data)
        count = 0
        for i in json_data:
            print ("count: ", count)
            count += 1
        print (json_data)
    print("-> Done!")
    """