from __future__ import division
from __future__ import print_function
import os, scipy.io
import time, datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob

import model


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                        im[0:H:2, 1:W:2, :],
                        im[1:H:2, 1:W:2, :],
                        im[1:H:2, 0:W:2, :]), axis=2)
    return out

def run(inputImage, checkpoint_dir, network):
    fileNum = len(inputImage)
    
    with tf.Session() as sess:
        in_image = tf.placeholder(tf.float32, [None, None, None, 4])
        out_image = network(in_image)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt:
            print('loaded', checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)

        output = sess.run(out_image, feed_dict={in_image: inputImage[0]})
        startTime = datetime.datetime.now()

        for index, item in enumerate(inputImage):
            output = sess.run(out_image, feed_dict={in_image: item})
            output = np.minimum(np.maximum(output, 0), 1)
            output = output[0, :, :, :]

        endTime = datetime.datetime.now()
        fps =  fileNum / ((endTime - startTime).seconds + (endTime - startTime).microseconds * 1e-6)
        return fps

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
multi_checkpoint_dir = './checkpoint/Multi/'
unet_checkpoint_dir = './checkpoint/Unet/'

# get test IDs
test_fns = glob.glob(gt_dir + '/2*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

inputImage = []

print('Loading test data...')

for test_id in test_ids:
    # test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
        input_full = np.minimum(input_full, 1.0)
        inputImage.append(input_full)
        
        if len(inputImage) >= 30:
            break
    if len(inputImage) >= 30:
        break

print('Use Basic method to process...')
fps1 = run(inputImage, unet_checkpoint_dir, model.unet)

print('Use improved method to process...')
tf.reset_default_graph()
fps2 = run(inputImage, multi_checkpoint_dir, model.multi)

print("Basic method:\nEnd2End fps: {prec: .4f}".format(prec=fps1))
print("Improved method:\nEnd2End fps: {prec: .4f}".format(prec=fps2))
