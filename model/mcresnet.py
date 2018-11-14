# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the original form of Residual Networks.

This file are modified to add multi-connection module
based on the resnet_v1.py from slim of tensorflow.

@WindWang2, wangjicheng11@163.com
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import resnet_utils

resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = tf.contrib.slim


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False,
               use_actfn=True):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(inputs,
                             depth, [1, 1],
                             stride=stride,
                             activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                             scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    if use_actfn:
      if use_bounded_activations:
        # Use clip_by_value to simulate bandpass activation.
        residual = tf.clip_by_value(residual, -6.0, 6.0)
        output = tf.nn.relu6(shortcut + residual)
      else:
        output = tf.nn.relu(shortcut + residual)
    else:
      output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def _mcResnet(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=False,
              reuse=None,
              scope=None):
  """Generator for mcResnet models.

  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        outputs_collections = 'bottleneck'
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = resnet_utils.conv2d_same(net, 64, 3, stride=1, scope='conv1_1')
          net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1_1')

          reconv1 = slim.conv2d(net, 32, 1, stride=1, scope='conv1_1_res1')
          reconv1 = slim.conv2d(reconv1, 32, 3, stride=1, padding='SAME',
                                scope='conv1_1_res2')
          reconv1 = slim.conv2d(reconv1, 64, 1, stride=1, scope='conv1_1_resnet',
                                activation_fn=None)
          reconv1 = tf.nn.relu(net+reconv1)
          net = slim.utils.collect_named_outputs(outputs_collections, 'conv1', reconv1)

          net = resnet_utils.conv2d_same(net, 128, 3, stride=1, scope='conv1_2')
          net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1_2')

          reconv2 = slim.conv2d(net, 64, 1, stride=1, scope='conv1_2_res1')
          reconv2 = slim.conv2d(reconv2, 64, 3, stride=1, padding='SAME',
                                scope='conv1_2_res2')
          reconv2 = slim.conv2d(reconv2, 128, 1, stride=1, scope='conv1_2_resnet',
                                activation_fn=None)
          reconv2 = tf.nn.relu(reconv2+net)
          net = slim.utils.collect_named_outputs(outputs_collections, 'conv2', reconv2)
          # Add this import to get the clear results
        net = resnet_utils.stack_blocks_dense_with_shotcut(net, blocks, output_stride,
                                                           outputs_collections=outputs_collections)

        # print(net)
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm):
          net = resnet_utils.conv2d_same(net, 1024, 1, stride=1, scope='conv_up_1')
          outputs_points = slim.utils.convert_collection_to_dict(outputs_collections)

          # upres1
          shortcut = outputs_points['resnet_v1_101/block3']
          # print(shortcut)
          conv_blk3 = slim.conv2d(shortcut, 512, 1, stride=1, scope='conv_blk3_1')
          conv_blk3 = slim.conv2d(conv_blk3, 512, 3, stride=1, padding='SAME',
                                  scope='conv_blk3_2')
          conv_blk3 = slim.conv2d(conv_blk3, 1024, 1, stride=1, scope='conv_blk3_3',
                                  activation_fn=None)
          # shortcut = outputs_points['resnet_v1_101/block3']
          # shortcut = slim.conv2d(shortcut, 512, 1, stride=1, scope='newOut1')
          # conv_blk3 = slim.relu(shortcut+conv_blk3)
          shortcut = tf.nn.relu(conv_blk3 + shortcut + net)

          residual = slim.conv2d(shortcut, 512, [1, 1], stride=1,
                                 scope='b1conv1')
          residual = resnet_utils.conv2d_same(residual, 512, 3, 1,
                                              rate=1, scope='b1conv2')
          residual = slim.conv2d(residual, 1024, [1, 1], stride=1,
                                 activation_fn=None, scope='b1conv3')
          net = tf.nn.relu(shortcut + residual)

          # upres2
          shortcut = outputs_points['resnet_v1_101/block2']
          conv_blk2 = slim.conv2d(shortcut, 256, 1, stride=1, scope='conv_blk2_1')
          conv_blk2 = slim.conv2d(conv_blk2, 256, 3, stride=1, padding='SAME',
                                  scope='conv_blk2_2')
          conv_blk2 = slim.conv2d(conv_blk2, 512, 1, stride=1, scope='conv_blk2_3',
                                  activation_fn=None)
          # conv_blk2 = slim.relu(shortcut+conv_blk2)
          net = slim.conv2d(net, 512, [1, 1], stride=1, scope='b2conv0')
          shortcut = tf.nn.relu(conv_blk2 + shortcut + net)

          residual = slim.conv2d(shortcut, 256, [1, 1], stride=1,
                                 scope='b2conv1')
          residual = resnet_utils.conv2d_same(residual, 256, 3, 1,
                                              rate=1, scope='b2conv2')
          residual = slim.conv2d(residual, 512, [1, 1], stride=1,
                                 activation_fn=None, scope='b2conv3')
          net = tf.nn.relu(shortcut + residual)

          # upres3
          shortcut = outputs_points['resnet_v1_101/block1']
          # print(shortcut)
          conv_blk1 = slim.conv2d(shortcut, 128, 1, stride=1, scope='conv_blk1_1')
          conv_blk1 = slim.conv2d(conv_blk1, 128, 3, stride=1, padding='SAME',
                                  scope='conv_blk1_2')
          conv_blk1 = slim.conv2d(conv_blk1, 256, 1, stride=1, scope='conv_blk1_3',
                                  activation_fn=None)
          net = slim.conv2d(net, 256, [1, 1], stride=1, scope='b3conv0')
          shortcut = tf.nn.relu(conv_blk1 + shortcut + net)

          residual = slim.conv2d(shortcut, 128, [1, 1], stride=1,
                                 scope='b3conv1')
          residual = slim.conv2d_transpose(residual, 128, 3, stride=2, scope='b3conv2')
          residual = slim.conv2d(residual, 256, [1, 1], stride=1,
                                 activation_fn=None, scope='b3conv3')
          shortcut_0 = tf.image.resize_images(shortcut,
                                              (shortcut.shape[1]*2, shortcut.shape[2]*2))
          net = tf.nn.relu(shortcut_0 + residual)

          # upres4
          shortcut = outputs_points['conv2']
          # print(shortcut)
          conv_blk0 = slim.conv2d(shortcut, 64, 1, stride=1, scope='conv_blk0_1')
          conv_blk0 = slim.conv2d(conv_blk0, 64, 3, stride=1, padding='SAME',
                                  scope='conv_blk0_2')
          conv_blk0 = slim.conv2d(conv_blk0, 128, 1, stride=1, scope='conv_blk0_3',
                                  activation_fn=None)
          net_0 = slim.conv2d(net, 128, 1, scope='b4conv0')
          # net_0 = tf.image.resize_images(net_0, (net.shape[1]*2, net.shape[2]*2))
          shortcut = tf.nn.relu(net_0 + shortcut + conv_blk0)

          residual = slim.conv2d(shortcut, 64, [1, 1], stride=1,
                                 scope='b4conv1')
          residual = slim.conv2d_transpose(residual, 64, 3, stride=2, scope='b4conv2')
          # residual = resnet_utils.conv2d_same(residual, 128, 3, 1,
                                              # rate=1, scope='b4conv2')
          residual = slim.conv2d(residual, 128, [1, 1], stride=1,
                                 activation_fn=None, scope='b4conv3')
          shortcut_0 = tf.image.resize_images(shortcut,
                                              (shortcut.shape[1]*2, shortcut.shape[2]*2))
          net = tf.nn.relu(shortcut_0 + residual)

          # upres5
          shortcut = outputs_points['conv1']
          conv_blk00 = slim.conv2d(shortcut, 32, 1, stride=1, scope='conv_blk00_1')
          conv_blk00 = slim.conv2d(conv_blk00, 32, 3, stride=1,padding='SAME',
                                   scope='conv_blk00_2')
          conv_blk00 = slim.conv2d(conv_blk00, 64, 1, stride=1, scope='conv_blk00_3',
                                  activation_fn=None)
          net_0 = slim.conv2d(net, 64, 1, scope='b5conv0')
          # net_0 = tf.image.resize_images(net_0, (net.shape[1]*2, net.shape[2]*2))
          # net_0 = slim.conv2d_transpose(net, 64, 3, stride=2, scope='b5conv0')
          shortcut = tf.nn.relu(net_0 + shortcut + conv_blk00)
          residual = slim.conv2d(shortcut, 32, [1, 1], stride=1,
                                 scope='b5conv1')
          residual = slim.conv2d_transpose(residual, 32, 3, stride=2, scope='b5conv2')
          residual = slim.conv2d(residual, 64, [1, 1], stride=1,
                                 activation_fn=None, scope='b5conv3')
          shortcut_0 = tf.image.resize_images(shortcut,
                                              (shortcut.shape[1]*2, shortcut.shape[2]*2))
          net = tf.nn.relu(shortcut_0 + residual)
          # Convert end_points_collection into a dictionary of end_points.
          end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)

          if global_pool:
            # Global average pooling.
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            end_points['global_pool'] = net
          if num_classes:
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='logits')
            end_points[sc.name + '/logits'] = net
            if spatial_squeeze:
              net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
              end_points[sc.name + '/spatial_squeeze'] = net
            end_points['predictions'] = slim.softmax(net, scope='predictions')
          return net, end_points, outputs_points


def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])



def mcResnet(inputs,
             num_classes=None,
             is_training=True,
             global_pool=True,
             output_stride=None,
             spatial_squeeze=True,
             reuse=None,
             scope='resnet_v1_101'):
  """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return _mcResnet(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
