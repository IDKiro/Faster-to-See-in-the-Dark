import tensorflow as tf
import tensorflow.contrib.slim as slim

def lrelu(x):
    return tf.maximum(x*0.2,x)

def concat_and_upsample(x1, x2, output_shape, in_channels):
    pool_size = 4
    concat_out = tf.concat([x1, x2], 3)

    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, 2 * in_channels, 2 * in_channels], stddev=0.02))
    deconv_output = tf.nn.conv2d_transpose(concat_out, deconv_filter, output_shape, strides=[1, pool_size, pool_size, 1])
    deconv_output.set_shape([None, None, None, 2 * in_channels])

    return deconv_output

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def designA(i_s, i_c, in_channels):
    dw1 = slim.separable_conv2d(i_s, None, [3,3],1, activation_fn=lrelu)
    dw1 = slim.avg_pool2d(dw1, [2,2], padding='SAME')
    dw2 = slim.separable_conv2d(dw1, None, [3,3],1, activation_fn=lrelu)
    dw2 = slim.avg_pool2d(dw2, [2,2], padding='SAME')

    pw1 = slim.conv2d(i_c, in_channels, [1,1], activation_fn=lrelu)
    pw1 = slim.avg_pool2d(pw1, [2,2], padding='SAME')
    pw2 = slim.conv2d(pw1, in_channels, [1,1], activation_fn=lrelu)
    pw2 = slim.avg_pool2d(pw2, [2,2], padding='SAME')

    tmp = tf.concat([i_s, i_c], 3)
    tmp = slim.conv2d(tmp, 2*in_channels, [3,3], padding='SAME', activation_fn=lrelu)
    tmp = slim.avg_pool2d(tmp, [4,4], stride=4, padding='SAME')
    os, oc = tf.split(tmp, num_or_size_splits=2, axis=3)

    s = tf.concat([dw2, os], 3)
    c = tf.concat([pw2, oc], 3)
    return s, c

def designB(i_s, i_c, in_channels):
    dw1 = slim.separable_conv2d(i_s, None, [3,3],1, activation_fn=lrelu)
    dw1 = slim.avg_pool2d(dw1, [2,2], padding='SAME')
    dw2 = slim.separable_conv2d(dw1, None, [3,3],1, activation_fn=lrelu)
    dw2 = slim.avg_pool2d(dw2, [2,2], padding='SAME')

    pw1 = slim.conv2d(i_c, in_channels, [1,1], activation_fn=lrelu)
    pw1 = slim.avg_pool2d(pw1, [2,2], padding='SAME')
    pw2 = slim.conv2d(pw1, in_channels, [1,1], activation_fn=lrelu)
    pw2 = slim.avg_pool2d(pw2, [2,2], padding='SAME')

    ws = tf.Variable(1, dtype=tf.float32, name="ws")
    wc = tf.Variable(1, dtype=tf.float32, name="wc")
    wsc = tf.Variable(0, dtype=tf.float32, name="wsc")
    os = ws * i_s + wsc * i_c
    oc = wsc * i_s + wc * i_c
    os = slim.avg_pool2d(os, [4,4], stride=4, padding='SAME')
    oc = slim.avg_pool2d(oc, [4,4], stride=4, padding='SAME')

    s = tf.concat([dw2, os], 3)
    c = tf.concat([pw2, oc], 3)
    return s, c

def multiBranch(input):
    conv1 = slim.conv2d(input, 16, [3,3], rate=1, activation_fn=lrelu, scope='g_conv1_1')

    s1, c1 = designA(conv1, conv1, 16)
    s2, c2 = designA(s1, c1, 32)
    s3, c3 = designA(s2, c2, 64)

    channel_weight_output = tf.multiply(s3, c3)

    conv4 = slim.conv2d(channel_weight_output, 128, [3,3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 128, [3,3], rate=1, activation_fn=lrelu, scope='g_conv4_2')

    shape5 = tf.multiply(tf.shape(s2), tf.constant([1,1,1,4]))
    up5 =  concat_and_upsample(conv4, s3, shape5, 128)
    conv5 = slim.conv2d(up5,  64, [3,3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 64, [3,3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    shape6 = tf.multiply(tf.shape(s1), tf.constant([1,1,1,4]))
    up6 =  concat_and_upsample(conv5, s2, shape6, 64)
    conv6 = slim.conv2d(up6,  32, [3,3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 32, [3,3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    shape7 = tf.multiply(tf.shape(conv1), tf.constant([1,1,1,4]))
    up7 =  concat_and_upsample(conv6, s1, shape7, 32)
    conv7 = slim.conv2d(up7,  16, [3,3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 16, [3,3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    conv8 = slim.conv2d(conv7, 12, [1,1], rate=1, activation_fn=None, scope='g_conv8')
    out = tf.depth_to_space(conv8, 2)
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

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2)
    return out
