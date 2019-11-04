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


"""A tensorflow table stored in tf.Variables.

The row is the index or location at which the value is saved, and the value is
a nest of Tensors.

This class is not threadsafe.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import json

from tf_agents.utils import common

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
          return int(obj)
        elif isinstance(obj, np.floating):
          return float(obj)
        elif isinstance(obj, np.ndarray):
          return obj.tolist()
        elif isinstance(obj, tf.Tensor):
          try:
            return obj.numpy()
          except:
            return 1
        return json.JSONEncoder.default(self, obj)

def numpyify(value):
  try:
    value = value.numpy()
  except:
    pass
  return(value)

class Table(tf.Module):
  """A table that can store Tensors or nested Tensors."""

  def __init__(self, tensor_spec, capacity, scope='Table'):
    """Creates a table.

    Args:
      tensor_spec: A nest of TensorSpec representing each value that can be
        stored in the table.
      capacity: Maximum number of values the table can store.
      scope: Variable scope for the Table.
    Raises:
      ValueError: If the names in tensor_spec are empty or not unique.
    """
    super(Table, self).__init__(name=scope)
    self._tensor_spec = tensor_spec
    self._capacity = capacity
    self._file_name = 'collection.json'

    def _create_unique_slot_name(spec):
      #print ("------------------")
      #print ("spec: ", spec)
      #print ("test name: ", spec.name)
      results = tf.compat.v1.get_default_graph().unique_name(spec.name or 'slot')
      #print ("_create_unique_slot_name: ", results)
      return results

    self._slots = tf.nest.map_structure(_create_unique_slot_name,
                                        self._tensor_spec)
    #print ("self._slots", self._slots)

    def _create_storage(spec, slot_name):
      """Create storage for a slot, track it."""
      shape = [self._capacity] + spec.shape.as_list()

      #print ("shape: ", shape)
      new_storage = common.create_variable(
          name=slot_name,
          initializer=tf.zeros(shape, dtype=spec.dtype),
          shape=None,
          dtype=spec.dtype,
          unique_name=False)
      
      #print ("new storage: ", new_storage)
      return new_storage

    with tf.compat.v1.variable_scope(scope):
      self._storage = tf.nest.map_structure(_create_storage, self._tensor_spec,
                                            self._slots)
    print("self._storage: ", self._storage)

    self._slot2storage_map = dict(
        zip(tf.nest.flatten(self._slots), tf.nest.flatten(self._storage)))
    
    #print ("self._slot2storage_map:", self._slot2storage_map)
  @property
  def slots(self):
    return self._slots

  def variables(self):
    return tf.nest.flatten(self._storage)

  def read(self, rows, slots=None):
    """Returns values for the given rows.

    Args:
      rows: A scalar/list/tensor of location(s) to read values from. If rows is
        a scalar, a single value is returned without a batch dimension. If rows
        is a list of integers or a rank-1 int Tensor a batch of values will be
        returned with each Tensor having an extra first dimension equal to the
        length of rows.
      slots: Optional list/tuple/nest of slots to read from. If None, all
        tensors at the given rows are retrieved and the return value has the
        same structure as the tensor_spec. Otherwise, only tensors with names
        matching the slots are retrieved, and the return value has the same
        structure as slots.

    Returns:
      Values at given rows.
    """
    print ("-------------------------")
    data = []
    
    #for line in open(self._file_name, 'r'):
    #  data.append(json.loads(line))
    #with open(self._file_name, 'r') as fd:
    #  data = json.load(fd)

    print ("DATA :")
    print ("READ CALL")
    print ("self: ", self)
    print ("rows: ", rows)
    print ("slots: ", slots)


    print ("type ", type(rows))

    flattened_rows = tf.nest.flatten(rows)
    print ("flatten_rows: ", rows)
    #print("row session", (tf.Session().run(rows)))
    slots = slots or self._slots
    flattened_slots = tf.nest.flatten(slots)
    print ("slots2: ", slots)
    print ("flatten_slots: ", flattened_slots)
    
    print ("FROM READ self._slot2storage: ", self._slot2storage_map)
    values = [
        self._slot2storage_map[slot].sparse_read(rows)
        for slot in flattened_slots
    ]
    print ("values: ", values)
    print ("Read returns", tf.nest.pack_sequence_as(slots, values))
    #print ("value numpy: ", [i for i in values])
    
    
    return tf.nest.pack_sequence_as(slots, values)


  def write(self, rows, values, slots=None):
    """Returns ops for writing values at the given rows.

    Args:
      rows: A scalar/list/tensor of location(s) to write values at.
      values: A nest of Tensors to write. If rows has more than one element,
        values can have an extra first dimension representing the batch size.
        Values must have the same structure as the tensor_spec of this class
        if `slots` is None, otherwise it must have the same structure as
        `slots`.
      slots: Optional list/tuple/nest of slots to write. If None, all tensors
        in the table are updated. Otherwise, only tensors with names matching
        the slots are updated.

    Returns:
      Ops for writing values at rows.
    """

    #print ("rows: ", rows)
    #print ("values: ", values)
    #print ("slots: ", slots)
    #print ("values type", type(values))
    slots = slots or self._slots
    flattened_slots = tf.nest.flatten(slots)
    flattened_values = tf.nest.flatten(values)

    #print ("values: ", values)
    #print ("flatten values: ", flattened_values)

    #print ("flattened_slots: ", flattened_slots)
    #print ("flattened_values: ", flattened_values)
    with open(self._file_name, 'a') as fd:
      dictionary = {}
      for (slot, value) in zip(flattened_slots, flattened_values):
        #print("slot :", slot)
        #print("type slot:", type(slot))
        #print("value:", value)
        #print("type value:", type(value))
        #print("numpyify value", numpyify(value))
        
        dictionary[slot] = numpyify(value)
        #print ("dictionary: ", dictionary)
      #data = json.dumps(dictionary, separators=(',', ':'))
      #data = json.dumps({'Json_replay_buffer': [dictionary]}, separators=(',', ':'), cls=NumpyEncoder)
      data = json.dumps({'Json_replay_buffer': dictionary}, separators=(',', ':'), cls=NumpyEncoder)
      json.dump(data, fd)
      fd.write('\n')

    write_ops = [
        tf.compat.v1.scatter_update(self._slot2storage_map[slot], rows,
                                    value).op
        for (slot, value) in zip(flattened_slots, flattened_values)
    ]



    return tf.group(*write_ops)
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_agents.utils import common

import json


class Table(tf.Module):
  """A table that can store Tensors or nested Tensors."""

  def __init__(self, tensor_spec, capacity, scope='Table'):

    """Creates a table.

    Args:
      tensor_spec: A nest of TensorSpec representing each value that can be
        stored in the table.
      capacity: Maximum number of values the table can store.
      scope: Variable scope for the Table.
    Raises:
      ValueError: If the names in tensor_spec are empty or not unique.
    """

    super(Table, self).__init__(name=scope)
    self._tensor_spec = tensor_spec
    self._capacity = capacity
    self._file_name='collection.json'

  def _create_unique_slot_name(spec):
    """ Returns spec.name.
    unique_name will mark a name returned as being used, if the name is already used
    name will be incremented by 1.
    
    Args:
      spec: A spec of (shape, dtype, name, ...variables)
    """
    results = tf.compat.v1.get_default_graph().unique_name(spec.name or 'slot')
    return results

  #Create slots for each unique spec spec name
  self._slots = tf.nest.map_structure(_create_unique_slot_name, self._tensor_spec)

  def _create_storage(spec, slot_name):
    """Create storage for a slot, track it.
    shape:  [1000]
    new storage:  <tf.Variable 'TFUniformReplayBuffer/Table/id:0' shape=(1000,) dtype=int64, numpy=
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ...,0])>
    """
    shape = [self._capacity] + spec.shape.as_list()

    print ("shape: ", shape)
    new_storage = common.create_variable(
        name=slot_name,
        initializer=tf.zeros(shape, dtype=spec.dtype),
        shape=None,
        dtype=spec.dtype,
        unique_name=False)
    
    print ("new storage: ", new_storage)
    return new_storage
  
  with tf.compat.v1.variable_scope(scope):
    self._storage = tf.nest.map_structure(_create_storage, self._tensor_spec,
                                          self._slots)

  self._slot2storage_map = dict(
      zip(tf.nest.flatten(self._slots), tf.nest.flatten(self._storage)))
  
  @property
  def slots(self):
      return self._slots

  def variables(self):
      return tf.nest.flatten(self._storage)
  
  def write(self, rows, values, slots=None):
    """Returns ops for writing values at the given rows.

    Args:
      rows: A scalar/list/tensor of location(s) to write values at.
      values: A nest of Tensors to write. If rows has more than one element,
        values can have an extra first dimension representing the batch size.
        Values must have the same structure as the tensor_spec of this class
        if `slots` is None, otherwise it must have the same structure as
        `slots`.
      slots: Optional list/tuple/nest of slots to write. If None, all tensors
        in the table are updated. Otherwise, only tensors with names matching
        the slots are updated.

    Returns:
      Ops for writing values at rows.
    """

    print ("rows: ", rows)
    print ("values: ", values)
    print ("slots: ", slots)
    print ("values type", type(values))
  
    print ("flattened_slots: ", flattened_slots)
    print ("flattened_values: ", flattened_values)
    slots = slots or self._slots
    flattened_slots = tf.nest.flatten(slots)
    flattened_values = tf.nest.flatten(values)
    with open(self._file_name, 'a') as fd:
      dictionary = {}
      for (slot, value) in zip(flattened_slots, flattened_values):
        dictionary[slot] = value
      data = json.dumps({'Trajectory': dictionary}, separators=(',', ':'), cls=NumpyEncoder)
      json.dump(data, fd)  

    write_ops = [
        tf.compat.v1.scatter_update(self._slot2storage_map[slot], rows,
                                    value).op
        for (slot, value) in zip(flattened_slots, flattened_values)
    ]
    return tf.group(*write_ops)
'''

