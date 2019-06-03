# -*- coding: utf-8 -*-
# This file use to generate the tensor
# though tfcoords.
# @WindWang2, wangjicheng11@163.com
# Date: 2019-05-13

import matplotlib.pyplot as plt
import numpy as np
# Only for test
# import matplotlib.pyplot as plt
import tensorflow as tf


def _parse_function(example_proto):
    ps = 320
    feature = {'img': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}
    features = tf.parse_single_example(example_proto, feature)
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, (ps, ps, 3))
    img = tf.to_float(img)
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, (ps, ps))
    return img, label

def get_val_batch(file_name, batch_size=10):
    file = [file_name]
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(_parse_function)
    # dataset = dataset.map(lambda i)
    # dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(1).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator

def get_train_batch(file_name, batch_size=10):
    file = [file_name]
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=800)
    dataset = dataset.repeat().batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator

def _test(filename='../Data/val.tfrecords', bs=30):
    # init2 = tf.local_variables_initializer()
    init1 = tf.global_variables_initializer()
    train_it = get_train_batch('../../../Data/train_no_dsm.tfrecords', bs)
    val_it = get_val_batch('../../../Data/val_no_dsm.tfrecords', bs)
    is_training = tf.placeholder(tf.bool)
    ra, rb = tf.cond(is_training, lambda:train_it.get_next(), lambda:val_it.get_next())
    # ra, rb = it.get_next()
    # ra,rb = tf.cond(is_training, lambda: (ta, tb), lambda: (va, vb))
    # with tf.get_defaut_graph().control_dependencies([ra, rb]):
    rc= tf.concat((ra, tf.expand_dims(tf.to_float(rb), axis=-1)), axis=-1)
    # rd = tf.concat((va, tf.expand_dims(tf.to_float(vb), axis=-1)), axis=-1)
    with tf.Session() as sess:
        sess.run(init1)
        sess.run(train_it.initializer)
        for i in range(2):
            sess.run(val_it.initializer)
            try:
                count = 0
                for j in range(2):
                    a, b = sess.run([ra, rb], feed_dict={is_training:True})
                    # plt.imshow(train_a[2,:,:,:3])
                    # plt.show()
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.uint8(a[0,:,:,:]))
                    plt.subplot(1, 2, 2)
                    plt.imshow(b[0,:,:])
                    plt.savefig('../../../Data/test'+str(i)+'_'+str(j)+'.jpg')
                    if j > 200:
                        print(j)
                        print('train: ', j, a.shape)
                for _ in range(2):
                    print(_)
                    # val_a = sess.run(rc, feed_dict={is_training:False})
                    # count += val_a.shape[0]
                    # print('val: ', _, val_a.shape)
            except tf.errors.OutOfRangeError:
                print('*'*200)
                print(count)
            print('test')
