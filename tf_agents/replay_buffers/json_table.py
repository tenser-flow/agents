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

from tf_agents.utils import common
from tf_agents.replay_buffers import table
import json

class table(tf.Module):
    """A table that can store Tensors or nested Tensors."""

    def __init__self, tensor_spec, capacity, scope='Table'):

        super(Table, self).__init__(name=scope)
        self._tensor_spec = tensor_spec
        self._capacity = capacity

    def _create_unique_slot_name(spec):
      """ Returns spec.name.
      unique_name will mark a name returned as being used, if the name is already used
      name will be incremented by 1.
      
      Args:
        spec: A spec of (shape, dtype, name, ...variables)
      """
      results = tf.compat.v1.get_default_graph().unique_name(spec.name or 'slot')
      return results

    self._slots = tf.nest.map_structure(_create_unique_slot_name,
                                        self._tensor_spec)
    
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
    write_ops = [
        tf.compat.v1.scatter_update(self._slot2storage_map[slot], rows,
                                    value).op
        for (slot, value) in zip(flattened_slots, flattened_values)
    ]
    return tf.group(*write_ops)


