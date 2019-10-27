import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from breakout.dqn_model import DQN_Model, ExperienceBuffer
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_rows
from util.logging import TimeLogger

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
    parser.add_argument('--buffer-size', type=int, default=1008000)
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    parser.add_argument('--log-time', default=False, action='store_true')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)

    #LOAD MODEL
    model = DQN_Model(input_shape=VISUAL_OBS_SPEC,
                      num_actions=NUM_ACTIONS,
                      conv_layer_params=CONV_LAYER_PARAMS,
                      fc_layer_params=FC_LAYER_PARAMS,
                      learning_rate=LEARNING_RATE)
    gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)

    #SETUP TENSORBOARD/LOGGING
    train_log_dir = os.path.join(args.output_dir, 'logs/')
    os.makedirs(os.path.dirname(train_log_dir), exist_ok=True)
    loss_metrics = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    if args.log_time is True:
        time_logger = TimeLogger(["Fetch Data", "Parse Data", "Compute Loss", "Generate Grads"], num_cycles=args.train_steps)

    #TRAINING LOOP
    train_step = 0
    exp_buff = ExperienceBuffer(args.buffer_size)
    print("-> Starting training...")
    for epoch in range(args.train_epochs):
        if args.log_time is True: time_logger.reset()

        #FETCH DATA
        global_i = cbt_global_iterator(cbt_table)
        rows = cbt_read_rows(cbt_table, args.prefix, args.train_steps, global_i)

        if args.log_time is True: time_logger.log(0)

        for row in tqdm(rows, "Trajectories {} - {}".format(global_i - args.train_steps, global_i - 1)):
            #DESERIALIZE DATA
            bytes_traj = row.cells['trajectory']['traj'.encode()][0].value
            bytes_info = row.cells['trajectory']['info'.encode()][0].value
            traj, info = Trajectory(), Info()
            traj.ParseFromString(bytes_traj)
            info.ParseFromString(bytes_info)

            #FORMAT DATA
            obs_shape = np.append(info.num_steps, info.visual_obs_spec).astype(int)
            obs = np.asarray(traj.visual_obs).reshape(obs_shape)

            exp_buff.add_trajectory(obs, traj.actions, traj.rewards, info.num_steps)

            if args.log_time is True: time_logger.log(1)

            #COMPUTE LOSS
            with tf.GradientTape() as tape:
                q_pred, q_next = model(exp_buff.obs), model(exp_buff.next_obs)
                one_hot_actions = tf.one_hot(exp_buff.actions, NUM_ACTIONS)
                q_pred = tf.reduce_sum(q_pred * one_hot_actions, axis=-1)
                q_next = tf.reduce_max(q_next, axis=-1)
                q_next = q_next * exp_buff.next_mask
                q_target = exp_buff.rewards + tf.multiply(tf.constant(GAMMA, dtype=tf.float32), q_next)
                loss = model.loss(q_pred, q_target)
            
            if args.log_time is True: time_logger.log(2)

            #GENERATE GRADIENTS
            total_grads = tape.gradient(loss, model.trainable_weights)
            model.opt.apply_gradients(zip(total_grads, model.trainable_weights))

            if args.log_time is True: time_logger.log(3)

            #TENSORBOARD LOGGING
            loss_metrics(loss)
            total_reward = np.sum(traj.rewards)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_metrics.result(), step=train_step)
                tf.summary.scalar('total reward', total_reward, step=train_step)
            train_step += 1

        if args.log_time is True: time_logger.print_logs()

        #SAVE MODEL WEIGHTS
        model_filename = args.prefix + '_model.h5'
        gcs_save_weights(model, gcs_bucket, args.tmp_weights_filepath, model_filename)
    print("-> Done!")
