# -*- coding: utf-8 -*-
# Test for model
# @WindWang2, wangjicheng11@163.com

import glob
import math
import os

import gdal
# import gdal
import numpy as np
import skimage.io as ski
import tensorflow as tf

from resnet_fcn_skip import resnet_fcn_101_skip

ps = 320
# ckpt_path='../../../tf_resnet101_work/road_6/checkpoints/resnet_fcn_101_epoch1070.ckpt-2180000'
ckpt_path='../../../tf_resnet101_work/resnet-fcn-ms/checkpoints/resnet_fcn_ms_more_1099.ckpt-4550000'
# Define the cmap
cmap = np.zeros((3, 256), dtype=np.uint16)
cmap[:, 0] = [255, 255, 255]
cmap[:, 1] = [0, 0, 255]
cmap[:, 2] = [0, 255, 255]
cmap[:, 3] = [0, 255, 0]
cmap[:, 4] = [255, 255, 0]
cmap[:, 5] = [255, 0, 0]
cmap = cmap * 256
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_cls = 6
def test(image, name):
    '''
    image is the 4d image with size
    '''
    h = image.shape[1]
    w = image.shape[2]
    xx = np.zeros((h, w, num_cls))
    stride = 40
    # patch_siez
    ps = 320

    with tf.Graph().as_default():
        image_test = tf.placeholder(tf.float32, shape=(1, ps, ps, 3))
        model = resnet_fcn_101_skip(6, is_inference=True,
                                    inference_tensor=image_test)
        out = model.inference()

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            saver.restore(sess, ckpt_path)
            for i in range(0, h-ps+stride, stride):
                print(i)
                for j in range(0, w-ps+stride, stride):
                    x = i
                    y = j
                    input_patch = image[:, x: x + ps, y: y+ps, :]
                    temp= sess.run(out,
                                   feed_dict={image_test: input_patch})
                    # print(temp.dtype)
                    xx[x: x+ps, y: y+ps,:] += temp[0, :, :, :]
            attention_data = np.argmax(xx, axis=-1)
            ski.imsave(name, np.uint8(attention_data), colormap=cmap)

if __name__ == '__main__':
    # image_list = glob.glob('../../Data/val/*image*.tif')
    image_list = glob.glob('../../../Data/test_new/*.tif')
    # dsm_list = [f.replace('image', 'dsm') for f in image_list]
    out_dir = 'test_new_results/resnet-fcn-ms'
    for i in image_list:
        print(i)
        image = ski.imread(i).astype(np.float32)
        data_new = image[np.newaxis, :]
        name = i.replace('test_new', out_dir)
        name = name.replace('.tif', '_resnet.tif')
        print(data_new.shape)
        test(data_new, name)
