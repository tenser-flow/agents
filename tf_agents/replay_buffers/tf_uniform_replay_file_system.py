# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A batched replay buffer of nests of Tensors which can be sampled uniformly.

- Each add assumes tensors have batch_size as first dimension, and will store
each element of the batch in an offset segment, so that each batch dimension has
its own contiguous memory. Within batch segments, behaves as a circular buffer.

The get_next function returns 'ids' in addition to the data. This is not really
needed for the batched replay buffer, but is returned to be consistent with
the API for a priority replay buffer, which needs the ids to update priorities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from itertools import cycle
import gin
import numpy as np
import tensorflow as tf
import glob

from tf_agents.replay_buffers import replay_buffer
from tf_agents.replay_buffers import table as table
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.utils import nest_utils


BufferInfo = collections.namedtuple('BufferInfo',
                                    ['ids', 'probabilities'])


def getint(name):
  """Returns the integer representation of a filename for sorting"""
  #_, num = name.split('_')
  num, _ = name.split('.')
  return int(num)

class TFUniformReplayBuffer(replay_buffer.ReplayBuffer):
  """A TFUniformReplayBuffer with batched adds and uniform sampling."""

  def __init__(self,
               data_spec,
               batch_size,
               max_length=1000,
               scope='TFUniformReplayBuffer',
               device='cpu:*',
               dataset_drop_remainder=False,
               dataset_window_shift=None,
               stateful_dataset=False,
               ):
    """Creates a TFUniformReplayBuffer.

    The TFUniformReplayBuffer stores episodes in `B == batch_size` blocks of
    size `L == max_length`, with total frame capacity
    `C == L * B`.  Storage looks like:

    ```
    block1 ep1 frame1
               frame2
           ...
           ep2 frame1
               frame2
           ...
           <L frames total>
    block2 ep1 frame1
               frame2
           ...
           ep2 frame1
               frame2
           ...
           <L frames total>
    ...
    blockB ep1 frame1
               frame2
           ...
           ep2 frame1
               frame2
           ...
           <L frames total>
    ```
    Multiple episodes may be stored within a given block, up to `max_length`
    frames total.  In practice, new episodes will overwrite old ones as the
    block rolls over its `max_length`.

    Args:
      data_spec: A TensorSpec or a list/tuple/nest of TensorSpecs describing a
        single item that can be stored in this buffer.
      batch_size: Batch dimension of tensors when adding to buffer.
      max_length: The maximum number of items that can be stored in a single
        batch segment of the buffer.
      scope: Scope prefix for variables and ops created by this class.
      device: A TensorFlow device to place the Variables and ops.
      table_fn: Function to create tables `table_fn(data_spec, capacity)` that
        can read/write nested tensors.
      dataset_drop_remainder: If `True`, then when calling
        `as_dataset` with arguments `single_deterministic_pass=True` and
        `sample_batch_size is not None`, the final batch will be dropped if it
        does not contain exactly `sample_batch_size` items.  This is helpful for
        static shape inference as the resulting tensors will always have
        leading dimension `sample_batch_size` instead of `None`.
      dataset_window_shift: Window shift used when calling
        `as_dataset` with arguments `single_deterministic_pass=True` and
        `num_steps is not None`.  This determines how the resulting frames are
        windowed.  If `None`, then there is no overlap created between frames
        and each frame is seen exactly once.  For example, if `max_length=5`,
        `num_steps=2`, `sample_batch_size=None`, and
        `dataset_window_shift=None`, then the datasets returned will have
        frames `{[0, 1], [2, 3], [4]}`.

        If `num_steps is not None`, then windows are created
        with a window overlap of `dataset_window_shift` and you will see each
        frame up to `num_steps` times.  For example, if `max_length=5`,
        `num_steps=2`, `sample_batch_size=None`, and `dataset_window_shift=1`,
        then the datasets returned will have windows of shifted repeated frames:
        `{[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]}`.

        For more details, see the documentation of `tf.data.Dataset.window`,
        specifically for the `shift` argument.

        The default behavior is to not overlap frames
        (`dataset_window_shift=None`) but users often want to see all
        combinations of frame sequences, in which case `dataset_window_shift=1`
        is the appropriate value.
      stateful_dataset: whether the dataset contains stateful ops or not.
    """
    self._batch_size = batch_size
    self._max_length = max_length
    capacity = self._batch_size * self._max_length
    super(TFUniformReplayBuffer, self).__init__(
        data_spec, capacity, stateful_dataset)

    self._id_spec = tensor_spec.TensorSpec([], dtype=tf.int64, name='id')
    #self._capacity_value = np.int64(self._capacity)
    self._batch_offsets = (
        tf.range(self._batch_size, dtype=tf.int64) * self._max_length)
    self._scope = scope
    self._device = device
    #self._table_fn = table_fn
    self._dataset_drop_remainder = dataset_drop_remainder
    self._dataset_window_shift = dataset_window_shift
    self._filenames = glob.glob("*.tfrecords")
    print (self._filenames)
    print (type(self._filenames))
    self._filenames = sorted(self._filenames, key=getint)
    self._max_record_idx = len(self._filenames)
    self._last_record_idx = 0
    #with tf.device(self._device), tf.compat.v1.variable_scope(self._scope):
    self._last_id = common.create_variable('last_id', -1)
    self._last_id_cs = tf.CriticalSection(name='last_id')

  def variables(self):
    return (self._data_table.variables() +
            self._id_table.variables() +
            [self._last_id])

  @property
  def device(self):
    """ A TensorFlow device to place the Variables and ops."""
    return self._device

  @property
  def table_fn(self):
    """Function to create tables `table_fn(data_spec, capacity)` that can 
    read/write nested tensors.
    """
    return self._table_fn

  @property
  def scope(self):
    """Scope prefix for variables and ops created by this class."""
    return self._scope

  # Methods defined in ReplayBuffer base class

  def _num_frames(self):
    """Returns the number of frames in the replay buffer."""
    num_items_single_batch_segment = self._get_last_id() + 1
    total_frames = num_items_single_batch_segment * self._batch_size
    return tf.minimum(total_frames, self._capacity)

  def _add_batch(self, items):
    """Adds a batch of items to the replay buffer.

    Args:
      items: A tensor or list/tuple/nest of tensors representing a batch of
      items to be added to the replay buffer. Each element of `items` must match
      the data_spec of this class. Should be shape [batch_size, data_spec, ...]
    Returns:
      An op that adds `items` to the replay buffer.
    Raises:
      ValueError: If called more than once.
    """
    with tf.device(self._device), tf.name_scope(self._scope):
      def _tf_example_encoder(flattened_slot, flattened_value):
        def _bytes_feature(value):
          """Returns a bytes_list from a string / byte."""
          if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
          return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _float_feature(value):
          """Returns a float_list from a float / double."""
          return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def _int64_feature(value):
          """Returns an int64_list from a bool / enum / int / uint."""
          return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        print ("type flatvalue", (flattened_value))
        if flattened_slot == 'step_type':
          encoded_value = _int64_feature(flattened_value.numpy())
        #print ("type step_type encoded:", encoded_value)
        elif flattened_slot == 'observation':
          value = flattened_value.numpy()[0]
          encoded_value = tf.train.Feature(float_list=tf.train.FloatList(value=value))
        elif flattened_slot == 'policy_info':
          encoded_value = _bytes_feature(flattened_value.numpy())
        elif flattened_slot == 'action':
          encoded_value = _int64_feature(flattened_value.numpy())
          #print ("type  action encoded:", encoded_value)
        elif flattened_slot == 'next_step_type':
          encoded_value = _int64_feature(flattened_value.numpy())
        elif flattened_slot == 'reward':
          encoded_value = _float_feature(flattened_value.numpy())
        elif flattened_slot == 'discount':
          encoded_value = _float_feature(flattened_value.numpy())
        else:
          encoded_value = _bytes_feature(b'zero')
        return encoded_value
      
      print ("add_batch is eager?", tf.executing_eagerly())
      tf.nest.assert_same_structure(items, self._data_spec)
      
      slots = self._data_spec
      
      flatten_slots = tf.nest.flatten(slots)
      flatten_items = tf.nest.flatten(items)
      iter_slots = cycle(flatten_slots)
      id_ = self._increment_last_id()
      print ("data : ", items)
      print ("data step_type", items.step_type)
      print ("data step_type.numpy", items.step_type.numpy())
      print ("iter_slots", iter_slots)
      print ("type iter _ iter_slots", iter_slots)

      #def tf_serialize(items, iter_slots):
      dictionary = {}
      for value , slot in zip(flatten_items, iter_slots):
        name = slot.name
        print ("name", name)
        print ("value", value)
        #print ("value.numpy", value.numpy())
        encoded_value = _tf_example_encoder(name, value)
        dictionary[name] = encoded_value
        example_dictionary = tf.train.Example(features=tf.train.Features(feature=dictionary))
      serialized_dictionary = example_dictionary.SerializeToString()
      print ("serialzied :", serialized_dictionary)
      print ("type ", type(serialized_dictionary))
      idx = str(id_.numpy())
      filename = idx + '.tfrecords'

      with tf.io.TFRecordWriter(filename) as writer:
        writer.write(serialized_dictionary)
    return 1
    #return tf.group(write_id_op, write_data_op)

  def _get_next(self,
                sample_batch_size=None,
                num_steps=None,
                time_stacked=True):
    """Returns an item or batch of items sampled uniformly from the buffer.

    Sample transitions uniformly from replay buffer. When sub-episodes are
    desired, specify num_steps, although note that for the returned items to
    truly be sub-episodes also requires that experience collection be
    single-threaded.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See get_next() documentation.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See get_next() documentation.
      time_stacked: Bool, when true and num_steps > 1 get_next on the buffer
        would return the items stack on the time dimension. The outputs would be
        [B, T, ..] if sample_batch_size is given or [T, ..] otherwise.
    Returns:
      A 2 tuple, containing:
        - An item, sequence of items, or batch thereof sampled uniformly
          from the buffer.
        - BufferInfo NamedTuple, containing:
          - The items' ids.
          - The sampling probability of each item.
    Raises:
      ValueError: if num_steps is bigger than the capacity.
    """
    with tf.device(self._device), tf.name_scope(self._scope):
        filenames = self._filenames
        record_index = self._last_record_idx
        
        #if sample_batch_size == None:
        #  sample_batch_size = 1
        #if num_steps == None:
        #  num_steps = 1

        #ids = np.arange(record_index, )
        #Create a description of the features.
        feature_description = {
        'step_type': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'observation': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'action': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'policy_info': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'next_step_type': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'reward': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        'discount': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        }

        def _parse_function(example_proto):
          # Parse the input `tf.Example` proto using the dictionary above.
          return tf.io.parse_single_example(example_proto, feature_description)  

        raw_dataset = tf.data.TFRecordDataset(filenames)
        print ("filesnames: ", filenames)
        print ("raw_dataset :", raw_dataset)
        
        parsed_dataset = raw_dataset.map(_parse_function)

        step_type, obs, action, next_step_type, reward, discount, policy_info, record = [], [], [], [], [] , [], (), []

        #shape (sample_batch_size, num_steps, data)
        
        i = 1
        for features in parsed_dataset:
          if tf.executing_eagerly():
            step_type.append(features['step_type'].numpy())
          else:
            step_type.append(features['step_type'].eval(session=self._scope))
          #print("step_type py_func")
          #print ([i for i in step_type])
          #step_type.append(features['step_type'].numpy())
          obs.append(features['observation'].numpy())
          action.append(features['action'].numpy())
          policy_info = ()
          next_step_type.append(features['next_step_type'].numpy())
          reward.append(features['reward'])
          discount.append(features['discount'].numpy())
          record.append(record_index)
          #data = np.stack([step_type, obs, action, policy_info, next_step_type, reward, discount])

          #np.insert(batch_array,[i, 0], action, axis = 0)
              
          i += 1
          record_index += 1
          if record_index > self._max_record_idx:
                record_index = 0
          if i > (sample_batch_size * num_steps):
            break
        
        self._last_record_idx = record_index

        print ("max_record:", self._max_record_idx)
        print ("record_index:", record_index)
        print ("list step_type:", step_type)
        #shape = (sample_batch_size, num_steps)
        shape = (sample_batch_size * num_steps)
        make_arrays = lambda struct: tf.nest.map_structure(np.asarray, struct)
        obs_spec = (sample_batch_size, num_steps, 4)

        step_type  = np.full(shape, step_type)
        obs = np.full((sample_batch_size * num_steps, 4), obs)
        action = np.full(shape, action)
        #policy_info = np.full(shape, policy_info)
        next_step_type = np.full(shape, next_step_type)
        reward = np.full(shape, reward)
        discount = np.full(shape, discount)
        record = np.full(shape, record)

        print ("np step_type :", step_type)


        shape = (sample_batch_size, num_steps)
        step_type = np.reshape(step_type, shape)
        print ("reshape step_type", step_type)
        obs = np.reshape(obs, obs_spec)
        action = np.reshape(action, shape)
        #policy_info = np.reshape(policy_info, shape)
        next_step_type = np.reshape(next_step_type, shape)
        reward = np.reshape(reward, shape)
        discount = np.reshape(discount, shape)
        record = np.reshape(record, shape)

        make_tensors = lambda struct: tf.nest.map_structure(tf.identity, struct)
        #batch_array = tuple(trajectory.Trajectory(data[0],data[1],data[2],data[3],data[4],data[5],data[6]))
        Traj = trajectory.Trajectory(step_type, obs,action,policy_info,next_step_type,reward,discount)
        
        probabilities = np.full(sample_batch_size, (sample_batch_size/100), dtype=float)

        buffer_info = BufferInfo(record, probabilities)
        data = tuple(Traj)
        
        print ("type data:", type(data))
        print ("type buffer :", type(buffer_info))

        print ("data : ", data)
        print ("type data", type(data))
        print ("[0]", data[0])

    return data, buffer_info

  def as_dataset(self,
                 sample_batch_size=None,
                 num_steps=None,
                 num_parallel_calls=None,
                 single_deterministic_pass=False):
    """Returns the parent class .as_dataset method"""
    return super(TFUniformReplayBuffer, self).as_dataset(
        sample_batch_size, num_steps, num_parallel_calls,
        single_deterministic_pass=single_deterministic_pass)
  
  def _as_dataset(self,
                  sample_batch_size=None,
                  num_steps=None,
                  num_parallel_calls=None):
    """Creates a dataset that returns entries from the buffer in shuffled order.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See as_dataset() documentation.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See as_dataset() documentation.
      num_parallel_calls: (Optional.) Number elements to process in parallel.
        See as_dataset() documentation.
    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).
    """

    """
    print ("as_dataset is eager ", tf.executing_eagerly())
    def generator_fn():
      print ("generator_fn is eager", tf.executing_eagerly())
      while True:
        yield _get_next(sample_batch_size, num_steps)


 

    ds = tf.data.Dataset.from_generator(generator_fn, dtypes, shapes).map(lambda *items: tf.nest.pack_sequence_as(data_spec, items))
    print ("data from generator: ", ds)
    """
    """
    data = self._get_next(sample_batch_size, num_steps)
    print ("Data from get_next", data)
    print ("type", type(data[0]))
    
    #dtypes1 = tuple(tf.dtypes.as_dtype(s) for s in tf.nest.flatten(data[0]))
    #dtypes2 = tuple(tf.dtypes.as_dtype(s) for s in tf.nest.flatten(data[1]))
    def generator_fn():
      while True:
        print ("generator_fn is eager", tf.executing_eagerly())
        yield self._get_next(sample_batch_size, num_steps)
    
    data_spec = self._data_spec
    #shapes = tuple(s.shape for s in tf.nest.flatten(data_spec))
    #print ("generator shapes", shapes)
    #dtypes = tuple(s.dtype for s in tf.nest.flatten(data_spec))
    #print ("generator dtypes", dtypes)

    dataset = tf.data.Dataset.from_generator(
        generator_fn, (dtypes1,dtypes2)).map(lambda *items: tf.nest.pack_sequence_as(data_spec, items))
    """

    def get_next(_):
      data, buffer = self.get_next(sample_batch_size, num_steps, time_stacked=True)
      return data,buffer

    dataset = tf.data.experimental.Counter().map(get_next, num_parallel_calls=num_parallel_calls)

    return dataset

    #dataset = self._get_next(sample_batch_size, num_steps)

    #print ("dataset from as_data:", dataset)
    #ds = tf.data.Dataset.from_tensors(dataset)




  def _single_deterministic_pass_dataset(self,
                                         sample_batch_size=None,
                                         num_steps=None,
                                         num_parallel_calls=None):
    """Creates a dataset that returns entries from the buffer in fixed order.

    Args:
      sample_batch_size: (Optional.) An optional batch_size to specify the
        number of items to return. See as_dataset() documentation.
      num_steps: (Optional.)  Optional way to specify that sub-episodes are
        desired. See as_dataset() documentation.
      num_parallel_calls: (Optional.) Number elements to process in parallel.
        See as_dataset() documentation.
    Returns:
      A dataset of type tf.data.Dataset, elements of which are 2-tuples of:
        - An item or sequence of items or batch thereof
        - Auxiliary info for the items (i.e. ids, probs).

    Raises:
      ValueError: If `dataset_drop_remainder` is set, and
        `sample_batch_size > self.batch_size`.  In this case all data will
        be dropped.
    """
    static_size = tf.get_static_value(sample_batch_size)
    static_num_steps = tf.get_static_value(num_steps)
    static_self_batch_size = tf.get_static_value(self._batch_size)
    static_self_max_length = tf.get_static_value(self._max_length)
    if (self._dataset_drop_remainder
        and static_size is not None
        and static_self_batch_size is not None
        and static_size > static_self_batch_size):
      raise ValueError(
          'sample_batch_size ({}) > self.batch_size ({}) and '
          'dataset_drop_remainder is True.  In '
          'this case, ALL data will be dropped by the deterministic dataset.'
          .format(static_size, static_self_batch_size))
    if (self._dataset_drop_remainder
        and static_num_steps is not None
        and static_self_max_length is not None
        and static_num_steps > static_self_max_length):
      raise ValueError(
          'num_steps_size ({}) > self.max_length ({}) and '
          'dataset_drop_remainder is True.  In '
          'this case, ALL data will be dropped by the deterministic dataset.'
          .format(static_num_steps, static_self_max_length))

    def get_row_ids(_):
      """Passed to Dataset.range(self._batch_size).flat_map(.), gets row ids."""
      with tf.device(self._device), tf.name_scope(self._scope):
        with tf.name_scope('single_deterministic_pass_dataset'):
          # Here we pass num_steps=None because _valid_range_ids uses
          # num_steps to determine a hard stop when sampling num_steps starting
          # from the returned indices.  But in our case, we want all the indices
          # and we'll use TF dataset's window() mechanism to get
          # num_steps-length blocks.  The window mechanism handles this stuff
          # for us.
          min_frame_offset, max_frame_offset = _valid_range_ids(
              self._get_last_id(), self._max_length, num_steps=None)
          # With auto-deps the top-level return of assert_less is not touched,
          # even though the operation is executed.  So we add a mark_used call.
          tf.compat.v1.assert_less(
              min_frame_offset,
              max_frame_offset,
              message='TFUniformReplayBuffer is empty. Make sure to add items '
              'before asking the buffer for data.').mark_used()

          min_max_frame_range = tf.range(min_frame_offset, max_frame_offset)

          drop_remainder = self._dataset_drop_remainder
          window_shift = self._dataset_window_shift
          def group_windows(ds_):
            return ds_.batch(num_steps, drop_remainder=drop_remainder)

          if sample_batch_size is None:
            def row_ids(b):
              # Create a vector of shape [num_frames] and slice it along each
              # frame.
              ids = tf.data.Dataset.from_tensor_slices(
                  b * self._max_length + min_max_frame_range)
              if num_steps is not None:
                ids = (ids.window(num_steps, shift=window_shift)
                       .flat_map(group_windows))
              return ids
            return tf.data.Dataset.range(self._batch_size).flat_map(row_ids)
          else:
            def batched_row_ids(batch):
              # Create a matrix of indices shaped [num_frames, batch_size]
              # and slice it along each frame row to get groups of batches
              # for frame 0, frame 1, ...
              return tf.data.Dataset.from_tensor_slices(
                  (min_max_frame_range[:, tf.newaxis]
                   + batch * self._max_length))

            indices_ds = (
                tf.data.Dataset.range(self._batch_size)
                .batch(sample_batch_size, drop_remainder=drop_remainder)
                .flat_map(batched_row_ids))

            if num_steps is not None:
              # We have sequences of num_frames rows shaped [sample_batch_size].
              # Window and group these to rows of shape
              # [num_steps, sample_batch_size], then
              # transpose them to get index tensors of shape
              # [sample_batch_size, num_steps].
              indices_ds = (indices_ds.window(num_steps, shift=window_shift)
                            .flat_map(group_windows)
                            .map(tf.transpose))

            return indices_ds

    # Get our indices as a dataset; each time we reinitialize the iterator we
    # update our min/max id bounds from the state of the replay buffer.
    ds = tf.data.Dataset.range(1).flat_map(get_row_ids)

    def get_data(id_):
      with tf.device(self._device), tf.name_scope(self._scope):
        with tf.name_scope('single_deterministic_pass_dataset'):
          data = self._data_table.read(id_ % self._capacity)
      buffer_info = BufferInfo(ids=id_, probabilities=())
      return (data, buffer_info)

    # Deterministic even though num_parallel_calls > 1.  Operations are
    # run in parallel but then the results are returned in original stream
    # order.
    ds = ds.map(get_data, num_parallel_calls=num_parallel_calls)

    return ds

  def _gather_all(self):
    """Returns all the items in buffer, shape [batch_size, timestep, ...].

    Returns:
      All the items currently in the buffer.
    """
    with tf.device(self._device), tf.name_scope(self._scope):
      with tf.name_scope('gather_all'):
        # Make ids, repeated over batch_size. Shape [batch_size, num_ids, ...].
        min_val, max_val = _valid_range_ids(
            self._get_last_id(), self._max_length)
        ids = tf.range(min_val, max_val)
        ids = tf.stack([ids] * self._batch_size)
        rows = tf.math.mod(ids, self._max_length)

        # Make batch_offsets, shape [batch_size, 1], then add to rows.
        batch_offsets = tf.expand_dims(
            tf.range(self._batch_size, dtype=tf.int64) * self._max_length,
            1)
        rows += batch_offsets

        # Expected shape is [batch_size, max_length, ...].
        data = self._data_table.read(rows)
    return data

  def _clear(self, clear_all_variables=False):
    """Return op that resets the contents of replay buffer.

    Args:
      clear_all_variables: boolean indicating if all variables should be
        cleared. By default, table contents will be unlinked from
        replay buffer, but values are unmodified for efficiency. Set
        `clear_all_variables=True` to reset all variables including Table
        contents.

    Returns:
      op that clears or unlinks the replay buffer contents.
    """
    table_vars = self._data_table.variables() + self._id_table.variables()
    def _init_vars():
      assignments = [self._last_id.assign(-1)]
      if clear_all_variables:
        assignments += [v.assign(tf.zeros_like(v)) for v in table_vars]
      return tf.group(*assignments, name='clear')
    return self._last_id_cs.execute(_init_vars)

  #  Helper functions.
  def _increment_last_id(self, increment=1):
    """Increments the last_id in a thread safe manner.

    Args:
      increment: amount to increment last_id by.
    Returns:
      An op that increments the last_id.
    """
    def _assign_add():
      return self._last_id.assign_add(increment).value()
    return self._last_id_cs.execute(_assign_add)

  def _get_last_id(self):

    def last_id():
      return self._last_id.value()

    return self._last_id_cs.execute(last_id)

  def _get_rows_for_id(self, id_):
    """Make a batch_size length list of tensors, with row ids for write."""
    id_mod = tf.math.mod(id_, self._max_length)
    rows = self._batch_offsets + id_mod
    return rows


def _valid_range_ids(last_id, max_length, num_steps=None):
  """Returns the [min_val, max_val) range of ids.

  When num_steps is provided, [min_val, max_val+num_steps) are also valid ids.

  Args:
    last_id: The last id added to the buffer.
    max_length: The max length of each batch segment in the buffer.
    num_steps: Optional way to specify that how many ids need to be valid.
  Returns:
    A tuple (min_id, max_id) for the range [min_id, max_id) of valid ids.
  """
  if num_steps is None:
    num_steps = tf.constant(1, tf.int64)

  min_id_not_full = tf.constant(0, dtype=tf.int64)
  max_id_not_full = tf.maximum(last_id + 1 - num_steps + 1, 0)

  min_id_full = last_id + 1 - max_length
  max_id_full = last_id + 1 - num_steps + 1

  return (tf.where(last_id < max_length, min_id_not_full, min_id_full),
          tf.where(last_id < max_length, max_id_not_full, max_id_full))