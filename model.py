import tensorflow as tf
import tensorflow.contrib.slim as slim

def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels, pool_size):
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def designA(i_s, i_c, outputlayer, is_pool = True):
    dw1 = slim.separable_conv2d(i_s, None, [3,3],1, activation_fn=lrelu)
    if is_pool:
        dw1 = slim.avg_pool2d(dw1, [2,2], padding='SAME')
    dw2 = slim.separable_conv2d(dw1, None, [3,3],1, activation_fn=lrelu)
    if is_pool:
        dw2 = slim.avg_pool2d(dw2, [2,2], padding='SAME')

    pw1 = slim.conv2d(i_c, outputlayer, [1,1], activation_fn=lrelu)
    if is_pool:
        pw1 = slim.avg_pool2d(pw1, [2,2], padding='SAME')
    pw2 = slim.conv2d(pw1, outputlayer, [1,1], activation_fn=lrelu)
    if is_pool:
        pw2 = slim.avg_pool2d(pw2, [2,2], padding='SAME')

    tmp = tf.concat([i_s, i_c], 3)
    tmp = slim.conv2d(tmp, 2*outputlayer, [3,3], padding='SAME', activation_fn=lrelu)
    if is_pool:
        tmp = slim.avg_pool2d(tmp, [4,4],stride=4, padding='SAME')
    os, oc = tf.split(tmp, num_or_size_splits=2, axis=3)
    s = tf.concat([dw2, os], 3)
    c = tf.concat([pw2, oc], 3)
    return s, c

def multiBranch(input):
    conv1 = slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1,16,[3,3], rate=1, activation_fn=lrelu, scope='g_conv1_2')

    s1, c1 = designA(conv1, conv1, 16, is_pool=False)
    s2, c2 = designA(s1, c1, 32)
    s3, c3 = designA(s2, c2, 64)
    s4, c4 = designA(s3, c3, 128)

    channel_weight_output = tf.multiply(s4, c4)
    pool4 = slim.max_pool2d(channel_weight_output, [4, 4], stride=4, padding='SAME')

    conv5 = slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
    conv5 = slim.conv2d(conv5,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')

    up6 =  upsample_and_concat(conv5, s4, 256, 512, 4)
    conv6 = slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
    conv6 = slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

    up7 =  upsample_and_concat(conv6, s3, 128, 256, 4)
    conv7 = slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
    conv7 = slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

    up8 =  upsample_and_concat(conv7, s2, 64, 128, 4)
    conv8 = slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
    conv8 = slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

    up9 =  upsample_and_concat(conv8, s1, 32, 64, 4)
    conv9 = slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9 = slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

    conv10 = slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10,2)
    return out

def unet(input):
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512, 2)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256, 2)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128, 2)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64, 2)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out
