# -*- coding: utf-8 -*-
# This file is main to train all the images
# @WindWang2, wangjicheng11@163.com
# Date: 2019-05-22
import datetime as dt
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.python.ops import control_flow_ops

import dataIO as io
import telegram_send
from model import attention_mcResnet as resnet_fcn

if len(sys.argv) != 2:
    print('Please enter the information (e.g., swjtu:0, no space)')
    sys.exit()
server_name = sys.argv[1]
# For specfic server
if server_name.split(':')[0]=='swjtu':
    os.environ['https_proxy']='socks5://127.0.0.1:1080'
short_des='fcn-ms_residual_new_all'
# select the GPU to train
# ------------------------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = server_name.split(':')[1]
# ------------------------------------------

# Define the global variables
# ------------------------------------------
WorkDir = '../../../tf_resnet101_work/resnet-fcn-ms-residual'
is_Finetuning = True
# ckpt_path ='../../../tf_resnet101_work/resnet-fcn-ms-residual/checkpoints/resnet_fcn_ms_more_1099.ckpt-1000000'
# ckpt_path ='../../../tf_resnet101_work/resnet-fcn-ms-residual/checkpoints/resnet_fcn_ms_more_1099.ckpt-1000000'
# ckpt_path ='../../../tf_resnet101_work/resnet-fcn-ms-residual/checkpoints/resnet_fcn_ms_2_more_1099.ckpt-2000000'
ckpt_path ='../../../tf_resnet101_work/resnet-fcn-ms-residual/checkpoints/resnet_fcn_ms_3_more_1099.ckpt-3010000'
# ckpt_path = os.path.join(WorkDir, 'checkpoints/resnet_fcn_ms1099.ckpt-3550000')

num_classes = 6
num_epoches = 100
num_per_epoch = 10000
epoch_per_display = 100
display_step = 100
xsize = 320
ysize = 320
init_learning_rate = 0.004
batch_size = 3
file_writer_path = os.path.join(WorkDir, 'log')
checkpoint_path = os.path.join(WorkDir, 'checkpoints')
train_file = '../../../Data/train_no_dsm_a_postdam_all.tfrecords'
val_file = '../../../Data/val_no_dsm_postdam.tfrecords'
# ------------------------------------------

if __name__ == '__main__':
    # Build the models
    # 1. Get the input tensor from the dataio
    train_it = io.get_train_batch(train_file, batch_size)
    val_it = io.get_val_batch(val_file, batch_size)
    # img_batch, label_batch = train_it.get_next()
    # img_batch_val, label_batch_val = val_it.get_next()

    # 2. Build model
    resfcn = resnet_fcn.resnet_fcn_101_skip(6, train_it=train_it, val_it=val_it)
    is_train = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    loss, acc = resfcn.build(is_train)
    global_steps = tf.Variable(0, trainable=False)


    # # 3. Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # ckpt preprocessing.
    saver = tf.train.Saver(tf.global_variables())
    # Use the ckpt to restore the same name variables
    if is_Finetuning is True:
        # variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
        ckpt_reader = tf.train.NewCheckpointReader(ckpt_path)
        var_list_ckpt = ckpt_reader.get_variable_to_shape_map()
        # the up conv have been initialised by init.
        # var_list_ckpt = {v:var_list_ckpt[v] for v in var_list_ckpt if 'resnet' in v}
        var_list_graph = tf.global_variables()
        # The variables in graph have the ':0' at the end
        name_var_graph = {v.name[:-2]: v  for v in var_list_graph}
        resore_list = {v: name_var_graph[v] for v in var_list_ckpt if v in name_var_graph}
        reader = tf.train.Saver(resore_list)
        print(resore_list)
    # 4. Train OP
    # All the Layers to train
    first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # Only the new add Layers to train
    second_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='upconv')
    loss_all = 0.1 * loss[0] + 0.1 * loss[1] + 0.1 * loss[2] + loss[3] + 0.1*loss[4]
    # loss_all = loss[3] + loss[4]
    # train_op = optimizer.minimize(loss_all, global_step=global_steps, var_list=first_train_vars)
    train_op = slim.learning.create_train_op(loss_all, optimizer, global_step=global_steps)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss_all = control_flow_ops.with_dependencies([updates], loss_all)

    # 5. Begin train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if is_Finetuning is True:
            reader.restore(sess, ckpt_path)

        sess.run(train_it.initializer)
        for epoch in range(0, num_epoches):
            sess.run(val_it.initializer)
            path = os.path.join(file_writer_path, str(epoch))
            if not os.path.exists(path):
                os.mkdir(path)
            print(epoch)
            step = 1
            lr = init_learning_rate *\
                 (1. - float(epoch) / float(num_epoches)) ** 0.9

            # Train per epoch
            try:
                for i in range(num_per_epoch):
                    localtime = time.asctime(time.localtime(time.time()))
                    l1, l2, l3, la, lw, _ = sess.run([loss[0], loss[1], loss[2], loss[3], loss[4],
                                                      train_op],
                                                     feed_dict={learning_rate: lr, is_train: True})
                    if (i+1) % display_step == 0:
                        print(localtime)
                        print(str(step), "loss1 =", l1, "loss2=", l2,
                              "loss3 =", l3, "lossa=", la, 'lossw=', lw)
                    step += 1
            except tf.errors.OutOfRangeError:
                print('*'*100)
                print("out of range")
                print('*'*100)
            # Val per epoch
            num = 0
            all_a1 = 0.0
            all_a2 = 0.0
            all_a3 = 0.0
            all_aa = 0.0
            print('-'*60)
            localtime = time.asctime(time.localtime(time.time()))
            print(' '*15 + 'Begin Validation' + '.'*20)
            print(' '*15 + str(localtime))
            print('-'*60)
            try:
                for i in range(10000000):
                    a1, a2, a3, aa = sess.run([acc[0], acc[1], acc[2], acc[3]],
                                              feed_dict={is_train: False})
                    all_a1 += a1
                    all_a2 += a2
                    all_a3 += a3
                    all_aa += aa
                    num += 1
                    localtime = time.asctime(time.localtime(time.time()))
            except:
                print(' '*15 + 'Validation End!')
            print(localtime)
            print('Test Accuracy 1: ', all_a1/num)
            print('Test Accuracy 2: ', all_a2/num)
            print('Test Accuracy 3: ', all_a3/num)
            print('Test Accuracy A: ', all_aa/num)
            print('-'*60)
            # Send to Telegram
            message = server_name + ': ' + 'epoch: ' + str(epoch) + ' ' + str(all_a1/num) + \
                      '\n ' + str(all_a2/num) + \
                      '\n ' + str(all_a3/num) + \
                      '\n ' + str(all_aa/num)
            message = message + '\n' + short_des
            # except telegram_send.telegram.error:
            try:
                telegram_send.send([message])
            except Exception:
                print('telegram_errors')
            # Save the model per epoch
            checkpoint_name = os.path.join(checkpoint_path,
                                           'resnet_fcn_ms_4_more_'+str(epoch+1000)+'.ckpt')
            saver.save(sess, checkpoint_name, global_step=global_steps)
