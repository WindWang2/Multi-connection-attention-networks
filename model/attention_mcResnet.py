# -*- coding: utf-8 -*-
# This file is use to create the tensorflow model,
# privide the input and some params to get the the output.
# @WindWang2, wangjicheng11@163.com
# Date: 2019-05-19

import tensorflow as tf
import tensorflow.contrib.slim as slim

from model import mcresnet as resnet_v1


class resnet_fcn_101_skip:
    def __init__(self, num_classes,
                 inference_tensor=None,
                 is_inference=False,
                 train_it=None,
                 val_it=None, mean=None):

        if mean is None:
            mean = [107.26956879, 84.22938178, 87.15694324]
        self.mean = tf.constant(mean,
                                dtype=tf.float32,
                                shape=[1, 1, 1, 3],
                                name='img_mean')
        # TODO: Add the judgement about is_inference
        self.is_inference = False
        self.inference_tensor = inference_tensor
        self.train_it = train_it
        self.val_it = val_it
        self.num_cls = num_classes
        self.re = None
        self.loss = None
        self.val = None

    # def _build for node
    def _build(self, in_tensor, reuse=False, sub='', istraining=True):
        in_tensor = in_tensor - self.mean
        print('*'*100)
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, _1, _2 = resnet_v1.mcResnet(in_tensor,
                                             output_stride=8,
                                             global_pool=False,
                                             is_training=istraining,
                                             reuse=reuse)
            # the last conv2d's weights is not share
            out = slim.conv2d(net, 6, kernel_size=1, stride=1, padding='SAME', scope='re'+sub)
            # the last conv2d's weights is not share
            # out_re = slim.conv2d(net, 6, kernel_size=1, stride=1, padding='SAME', scope='re_w'+sub)
            reconv1 = slim.conv2d(net, 32, 1, stride=1, scope='wight_conv1'+sub)
            reconv1 = slim.conv2d(reconv1, 32, 3, stride=1, scope='wight_conv2'+sub,
                                  padding='SAME')
            reconv1 = slim.conv2d(reconv1, 64, 1, stride=1, scope='wight_conv3'+sub,
                                  activation_fn=None)
            reconv1 = tf.nn.relu(net+reconv1)
            out_re = slim.conv2d(reconv1, 6, kernel_size=1, stride=1, padding='SAME', scope='re_w'+sub)

            # print(out)
            out = tf.image.resize_images(out, tf.shape(in_tensor)[1:3])
            out_re = tf.image.resize_images(out_re, tf.shape(in_tensor)[1:3])
            print(out, out_re)
        return out, out_re

    # implementation of FPN
    def _build_FPN(self, in_tensor, reuse=False, sub='', istraining=True):
        in_tensor = in_tensor - self.mean
        print('*'*100)
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, _1, _2 = resnet_v1.mcResnet(in_tensor,
                                             output_stride=8,
                                             global_pool=False,
                                             is_training=istraining,
                                             reuse=reuse)
            # the last conv2d's weights is not share
            # out = slim.conv2d(net, 6, kernel_size=1, stride=1, padding='SAME', scope='re'+sub)
            # the last conv2d's weights is not share
            # print(out)
            # out = tf.image.resize_images(out, tf.shape(in_tensor)[1:3])
        return net

    def build_FPN(self, train_phase):
        if(self.is_inference):
            img_batch = self.inference_tensor
        else:
            img_batch, label_batch = tf.cond(train_phase,
                                             lambda: self.train_it.get_next(),
                                             lambda: self.val_it.get_next())
        img_batch_2 = tf.image.resize_images(img_batch, (240, 240))
        img_batch_3 = tf.image.resize_images(img_batch, (160, 160))
        out1 = self._build_FPN(img_batch, reuse=False, sub='')
        out2 = self._build_FPN(img_batch_2, reuse=True, sub='_1')
        out3 = self._build_FPN(img_batch_3, reuse=True, sub='_2')



    # train_phase is the bool tensor
    def build(self, train_phase):
        if(self.is_inference):
            img_batch = self.inference_tensor
        else:
            img_batch, label_batch = tf.cond(train_phase,
                                             lambda: self.train_it.get_next(),
                                             lambda: self.val_it.get_next())

        img_batch_2 = tf.image.resize_images(img_batch, (240, 240))
        img_batch_3 = tf.image.resize_images(img_batch, (160, 160))

        label_batch_2 = tf.image.resize_images(tf.expand_dims(label_batch,axis=-1), (240, 240),
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_batch_2 = tf.squeeze(label_batch_2, axis=-1)
        label_batch_3 = tf.image.resize_images(tf.expand_dims(label_batch,axis=-1), (160, 160),
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_batch_3 = tf.squeeze(label_batch_3, axis=-1)

        out1, out1_w = self._build(img_batch, reuse=False, sub='')
        out2, out2_w = self._build(img_batch_2, reuse=True, sub='_1')
        out3, out3_w = self._build(img_batch_3, reuse=True, sub='_2')

        out1_re = tf.image.resize_images(out1, (320, 320))
        out2_re = tf.image.resize_images(out2, (240, 240))
        out3_re = tf.image.resize_images(out3, (160, 160))

        out2_re_2 = tf.image.resize_images(out2, (320, 320))
        out3_re_2 = tf.image.resize_images(out3, (320, 320))

        out1_w_r = tf.image.resize_images(out1_w, (320, 320))
        out2_w_r = tf.image.resize_images(out2_w, (320, 320))
        out3_w_r = tf.image.resize_images(out3_w, (320, 320))

        out_1_expand = tf.expand_dims(out1_w_r, axis=-1)
        out_2_expand = tf.expand_dims(out2_w_r, axis=-1)
        out_3_expand = tf.expand_dims(out3_w_r, axis=-1)

        out = tf.concat((out_1_expand, out_2_expand, out_3_expand), axis=-1)
        out_softmax = tf.nn.softmax(out, dim=-1)
        out_softmax1, out_softmax2, out_softmax3 = tf.split(out_softmax, 3, axis=-1)
        out_re = out1_re * tf.squeeze(out_softmax1, axis=-1) +\
                 out2_re_2 * tf.squeeze(out_softmax2, axis=-1) +\
                 out3_re_2 * tf.squeeze(out_softmax3, axis=-1)

        loss = self.get_loss(out_re, label_batch)
        loss_1 = self.get_loss(out1_re, label_batch)
        loss_2 = self.get_loss(out2_re, label_batch_2)
        loss_3 = self.get_loss(out3_re, label_batch_3)
        w_loss = self.get_weight_loss()

        acc = self.get_acc(out_re, label_batch)
        acc1 = self.get_acc(out1_re, label_batch)
        acc2 = self.get_acc(out2_re_2, label_batch)
        acc3 = self.get_acc(out3_re_2, label_batch)
        return [loss_1, loss_2, loss_3, loss, w_loss], [acc1, acc2, acc3, acc]

    def build_max_pooling(self, train_phase):
        if(self.is_inference):
            img_batch = self.inference_tensor
        else:
            img_batch, label_batch = tf.cond(train_phase,
                                             lambda: self.train_it.get_next(),
                                             lambda: self.val_it.get_next())
        img_batch_2 = tf.image.resize_images(img_batch, (240, 240))
        img_batch_3 = tf.image.resize_images(img_batch, (160, 160))

        label_batch_2 = tf.image.resize_images(tf.expand_dims(label_batch, axis=-1), (240, 240),
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_batch_2 = tf.squeeze(label_batch_2, axis=-1)
        label_batch_3 = tf.image.resize_images(tf.expand_dims(label_batch, axis=-1), (160, 160),
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_batch_3 = tf.squeeze(label_batch_3, axis=-1)

        out1, _ = self._build(img_batch, reuse=False, sub='')
        out2, _ = self._build(img_batch_2, reuse=True, sub='_1')
        out3, _ = self._build(img_batch_3, reuse=True, sub='_2')

        out2_re = tf.image.resize_images(out2, (320, 320))
        out3_re = tf.image.resize_images(out3, (320, 320))

        out_1_expand = tf.expand_dims(out1, axis=-1)
        out_2_expand = tf.expand_dims(out2, axis=-1)
        out_3_expand = tf.expand_dims(out3, axis=-1)

        out = tf.concat((out_1_expand, out_2_expand, out_3_expand), axis=-1)
        out_re = tf.reduce_max(out, axis=-1)

        loss = self.get_loss(out_re, label_batch)
        loss_1 = self.get_loss(out1, label_batch)
        loss_2 = self.get_loss(out2, label_batch_2)
        loss_3 = self.get_loss(out3, label_batch_3)
        w_loss = self.get_weight_loss()

        acc = self.get_acc(out_re, label_batch)
        acc1 = self.get_acc(out1, label_batch)
        acc2 = self.get_acc(out2_re, label_batch)
        acc3 = self.get_acc(out3_re, label_batch)
        return [loss_1, loss_2, loss_3, loss, w_loss], [acc1, acc2, acc3, acc]

    def inference_max_pooling(self):
        img_batch = self.inference_tensor
        img_batch_2 = tf.image.resize_images(img_batch, (240, 240))
        img_batch_3 = tf.image.resize_images(img_batch, (160, 160))

        out1, _ = self._build(img_batch, reuse=False, sub='')
        out2, _ = self._build(img_batch_2, reuse=True, sub='_1')
        out3, _ = self._build(img_batch_3, reuse=True, sub='_2')

        out_1_expand = tf.expand_dims(out1, axis=-1)
        out_2_expand = tf.expand_dims(out2, axis=-1)
        out_3_expand = tf.expand_dims(out3, axis=-1)

        out = tf.concat((out_1_expand, out_2_expand, out_3_expand), axis=-1)
        out_re = tf.reduce_max(out, axis=-1)

        return out_re

    def build_mean_pooling(self, train_phase):
        if(self.is_inference):
            img_batch = self.inference_tensor
        else:
            img_batch, label_batch = tf.cond(train_phase,
                                             lambda: self.train_it.get_next(),
                                             lambda: self.val_it.get_next())
        img_batch_2 = tf.image.resize_images(img_batch, (240, 240))
        img_batch_3 = tf.image.resize_images(img_batch, (160, 160))

        label_batch_2 = tf.image.resize_images(tf.expand_dims(label_batch, axis=-1), (240, 240),
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_batch_2 = tf.squeeze(label_batch_2, axis=-1)
        label_batch_3 = tf.image.resize_images(tf.expand_dims(label_batch, axis=-1), (160, 160),
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        label_batch_3 = tf.squeeze(label_batch_3, axis=-1)

        out1, _ = self._build(img_batch, reuse=False, sub='')
        out2, _ = self._build(img_batch_2, reuse=True, sub='_1')
        out3, _ = self._build(img_batch_3, reuse=True, sub='_2')

        out2_re = tf.image.resize_images(out2, (320, 320))
        out3_re = tf.image.resize_images(out3, (320, 320))

        out_1_expand = tf.expand_dims(out1, axis=-1)
        out_2_expand = tf.expand_dims(out2, axis=-1)
        out_3_expand = tf.expand_dims(out3, axis=-1)

        out = tf.concat((out_1_expand, out_2_expand, out_3_expand), axis=-1)
        out_re = tf.reduce_mean(out, axis=-1)

        loss = self.get_loss(out_re, label_batch)
        loss_1 = self.get_loss(out1, label_batch)
        loss_2 = self.get_loss(out2, label_batch_2)
        loss_3 = self.get_loss(out3, label_batch_3)
        w_loss = self.get_weight_loss()

        acc = self.get_acc(out_re, label_batch)
        acc1 = self.get_acc(out1, label_batch)
        acc2 = self.get_acc(out2_re, label_batch)
        acc3 = self.get_acc(out3_re, label_batch)
        return [loss_1, loss_2, loss_3, loss, w_loss], [acc1, acc2, acc3, acc]

    def inference_max_pooling(self):
        img_batch = self.inference_tensor
        img_batch_2 = tf.image.resize_images(img_batch, (240, 240))
        img_batch_3 = tf.image.resize_images(img_batch, (160, 160))

        out1, _ = self._build(img_batch, reuse=False, sub='')
        out2, _ = self._build(img_batch_2, reuse=True, sub='_1')
        out3, _ = self._build(img_batch_3, reuse=True, sub='_2')

        out_1_expand = tf.expand_dims(out1, axis=-1)
        out_2_expand = tf.expand_dims(out2, axis=-1)
        out_3_expand = tf.expand_dims(out3, axis=-1)

        out = tf.concat((out_1_expand, out_2_expand, out_3_expand), axis=-1)
        out_re = tf.reduce_mean(out, axis=-1)

        return out_re

    def inference(self, is_training=False):

        img_batch = self.inference_tensor
        img_batch_2 = tf.image.resize_images(img_batch, (240, 240))
        img_batch_3 = tf.image.resize_images(img_batch, (160, 160))


        out1, out1_w = self._build(img_batch, istraining=is_training,
                                   reuse=False, sub='')
        out2, out2_w = self._build(img_batch_2, reuse=True,
                                   istraining=is_training,sub='_1')
        out3, out3_w = self._build(img_batch_3, reuse=True,
                                   istraining=is_training,sub='_2')

        out1_re = tf.image.resize_images(out1, (320, 320))
        out2_re = tf.image.resize_images(out2, (240, 240))
        out3_re = tf.image.resize_images(out3, (160, 160))

        out2_re_2 = tf.image.resize_images(out2, (320, 320))
        out3_re_2 = tf.image.resize_images(out3, (320, 320))

        out1_w_r = tf.image.resize_images(out1_w, (320, 320))
        out2_w_r = tf.image.resize_images(out2_w, (320, 320))
        out3_w_r = tf.image.resize_images(out3_w, (320, 320))

        out_1_expand = tf.expand_dims(out1_w_r, axis=-1)
        out_2_expand = tf.expand_dims(out2_w_r, axis=-1)
        out_3_expand = tf.expand_dims(out3_w_r, axis=-1)

        out = tf.concat((out_1_expand, out_2_expand, out_3_expand), axis=-1)
        out_softmax = tf.nn.softmax(out, dim=-1)
        out_softmax1, out_softmax2, out_softmax3 = tf.split(out_softmax, 3, axis=-1)
        out_re = out1_re * tf.squeeze(out_softmax1, axis=-1) +\
                 out2_re_2 * tf.squeeze(out_softmax2, axis=-1) +\
                 out3_re_2 * tf.squeeze(out_softmax3, axis=-1)
        return out_re

    def get_loss(self, pred, label):
        # pred = self.build()
        logits_reshape = tf.reshape(pred, (-1, self.num_cls), name='logits_reshape')
        labels_one_hot = tf.one_hot(label, self.num_cls)
        labels_reshape = tf.to_float(tf.reshape(labels_one_hot, (-1, self.num_cls)),
                                     name='labels')
        classfication_loss = tf.losses.softmax_cross_entropy(labels_reshape,
                                                             logits_reshape)
        return classfication_loss

    def get_weight_loss(self):
        return tf.add_n(tf.losses.get_regularization_losses())

    def get_acc(self, pred, label):
        # TODO: Add the if to judge the re or loss is NULL
        with tf.name_scope('Acc'):
            re_logits = tf.to_int32(tf.argmax(pred, axis=3))
            tf_equal = tf.equal(tf.to_int32(label), re_logits)
            tf_real = tf.count_nonzero(tf_equal)
            tf_count = tf.size(tf_equal)
            val_op = tf.div(tf.to_float(tf_real), tf.to_float(tf_count))
            self.val = val_op
            return val_op

    def results(self, out_tensor):
        pass
