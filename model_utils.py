import tensorflow as tf
import tensorlayer as tl
import numpy as np


def _channel_shuffle(x, n_group):
    n, h, w, c = x.shape.as_list()
    x_reshaped = tf.reshape(x, [-1, h, w, n_group, c // n_group])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])
    return output


def _group_norm_and_channel_shuffle(x, is_train, G=32, epsilon=1e-12, use_shuffle=False, name='_group_norm'):
    with tf.variable_scope(name):
        N, H, W, C = x.get_shape().as_list()
        if N == None:
            N = -1
        G = min(G, C)
        x = tf.reshape(x, [N, G, H, W, C // G])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + epsilon)
        # shuffle channel
        if use_shuffle:
            x = tf.transpose(x, [0, 4, 2, 3, 1])
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0), trainable=is_train)
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.0), trainable=is_train)
        gamma = tf.reshape(gamma, [1, 1, 1, C])
        beta = tf.reshape(beta, [1, 1, 1, C])

        output = tf.reshape(x, [N, H, W, C]) * gamma + beta
    return output


def _switch_norm(x, name='_switch_norm') :
    with tf.variable_scope(name) :
        ch = x.shape[-1]
        eps = 1e-5

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
        ins_mean, ins_var = tf.nn.moments(x, [1, 2], keep_dims=True)
        layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)

        gamma = tf.get_variable("gamma", [ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", [ch], initializer=tf.constant_initializer(0.0))

        mean_weight = tf.nn.softmax(tf.get_variable("mean_weight", [3], initializer=tf.constant_initializer(1.0)))
        var_wegiht = tf.nn.softmax(tf.get_variable("var_weight", [3], initializer=tf.constant_initializer(1.0)))

        mean = mean_weight[0] * batch_mean + mean_weight[1] * ins_mean + mean_weight[2] * layer_mean
        var = var_wegiht[0] * batch_var + var_wegiht[1] * ins_var + var_wegiht[2] * layer_var

        x = (x - mean) / (tf.sqrt(var + eps))
        x = x * gamma + beta

        return x


def _add_coord(x):
    batch_size = tf.shape(x)[0]
    height, width = x.shape.as_list()[1:3]

    # 加1是为了使坐标值为[0,1]，不加1则是[0,1)
    y_coord = tf.range(0, height, dtype=tf.float32)
    y_coord = tf.reshape(y_coord, [1, -1, 1, 1])  # b,h,w,c
    y_coord = tf.tile(y_coord, [batch_size, 1, width, 1]) / (height-1)

    x_coord = tf.range(0, width, dtype=tf.float32)
    x_coord = tf.reshape(x_coord, [1, 1, -1, 1])  # b,h,w,c
    x_coord = tf.tile(x_coord, [batch_size, height, 1, 1]) / (width-1)

    o = tf.concat([x, y_coord, x_coord], 3)
    return o


def coord_layer(net):
    return tl.layers.LambdaLayer(net, _add_coord, name='coord_layer')


def switchnorm_layer(net, act, name):
    net = tl.layers.LambdaLayer(net, _switch_norm, name=name)
    if act is not None:
        net = tl.layers.LambdaLayer(net, act, name=name)
    return net


def groupnorm_layer(net, is_train, G, use_shuffle, act, name):
    net = tl.layers.LambdaLayer(net, _group_norm_and_channel_shuffle, {'is_train':is_train, 'G':G, 'use_shuffle':use_shuffle, 'name':name}, name=name)
    if act is not None:
        net = tl.layers.LambdaLayer(net, act, name=name)
    return net


def upsampling_layer(net, shortpoint):
    hw = shortpoint.outputs.shape.as_list()[1:3]
    net_upsamping = tl.layers.UpSampling2dLayer(net, hw, is_scale=False)
    net = tl.layers.ConcatLayer([net_upsamping, shortpoint], -1)
    return net


def upsampling_layer2(net, shortpoint, name):
    with tf.variable_scope(name):
        hw = shortpoint.outputs.shape.as_list()[1:3]
        dim1 = net.outputs.shape.as_list()[3]
        dim2 = shortpoint.outputs.shape.as_list()[3]

        net = conv2d(net, dim1//2, 1, 1, None, 'SAME', True, True, False, 'up1')
        shortpoint = conv2d(shortpoint, dim2//2, 1, 1, None, 'SAME', True, True, False, 'up2')

        net = tl.layers.UpSampling2dLayer(net, hw, is_scale=False)
        net = tl.layers.ConcatLayer([net, shortpoint], -1)
    return net


def upsampling_layer3(net, shortpoint):
    hw = shortpoint.outputs.shape.as_list()[1:3]
    shortpoint = tl.layers.LambdaLayer(shortpoint, lambda x: tf.split(x, 2, -1)[0])
    net = tl.layers.UpSampling2dLayer(net, hw, is_scale=False)
    net = tl.layers.ConcatLayer([net, shortpoint], -1)
    return net


def conv2d(net, n_filter, filter_size, strides, act, padding, use_norm, name):
    filter_size = np.broadcast_to(filter_size, [2])
    strides = np.broadcast_to(strides, [2])
    with tf.variable_scope(name):
        if use_norm:
            net = tl.layers.Conv2d(net, n_filter, filter_size, strides, None, padding, b_init=None, name='c2d')
            # net = groupnorm_layer(net, is_train, n_group, use_shuffle, act, 'gn')
            net = switchnorm_layer(net, act, 'sn')
        else:
            net = tl.layers.Conv2d(net, n_filter, filter_size, strides, act, padding, name='c2d')
    return net


def groupconv2d(net, n_filter, filter_size, strides, n_group, act, padding, use_norm, use_shuffle, name):
    filter_size = np.broadcast_to(filter_size, [2])
    strides = np.broadcast_to(strides, [2])
    with tf.variable_scope(name):
        if use_norm:
            net = tl.layers.GroupConv2d(net, n_filter, filter_size, strides, n_group, None, padding, b_init=None, name='gc2d')
            net = switchnorm_layer(net, act, 'sn')
            if use_shuffle:
                net = tl.layers.LambdaLayer(net, lambda x: _channel_shuffle(x, n_group))
        else:
            net = tl.layers.GroupConv2d(net, n_filter, filter_size, strides, n_group, act, padding, name='gc2d')

    return net


def deconv2d(net, n_filter, filter_size, strides, act, padding, use_norm, name):
    filter_size = np.broadcast_to(filter_size, [2])
    strides = np.broadcast_to(strides, [2])
    with tf.variable_scope(name):
        if use_norm:
            net = tl.layers.DeConv2d(net, n_filter, filter_size, strides=strides, padding=padding, b_init=None, name='dc2d')
            net = switchnorm_layer(net, act, 'sn')
        else:
            net = tl.layers.DeConv2d(net, n_filter, filter_size, strides=strides, act=act, padding=padding, name='dc2d')
    return net


def depthwiseconv2d(net, depth_multiplier, filter_size, strides, act, padding, use_norm, name, dilation_rate=1):
    filter_size = np.broadcast_to(filter_size, [2])
    strides = np.broadcast_to(strides, [2])
    dilation_rate = np.broadcast_to(dilation_rate, [2])
    with tf.variable_scope(name):
        if use_norm:
            net = tl.layers.DepthwiseConv2d(net, filter_size, strides, None, padding, dilation_rate=dilation_rate, depth_multiplier=depth_multiplier, b_init=None, name='dwc2d')
            net = switchnorm_layer(net, act, 'sn')
        else:
            net = tl.layers.DepthwiseConv2d(net, filter_size, strides, act, padding, dilation_rate=dilation_rate, depth_multiplier=depth_multiplier, name='dwc2d')

    return net


def resblock_1(net, n_filter, strides, act, name):
    strides = np.broadcast_to(strides, [2])
    with tf.variable_scope(name):
        if np.max(strides) > 1 or net.outputs.shape.as_list()[-1] != n_filter:
            shortcut = conv2d(net, n_filter, (3, 3), strides, None, 'SAME', True, 'shortcut')
        else:
            shortcut = net
        net = coord_layer(net)
        net = conv2d(net, n_filter, 1, 1, None, 'SAME', False, 'c1')
        net = groupconv2d(net, n_filter, 3, 1, 20, act, 'SAME', True, True, 'gc1')
        net = depthwiseconv2d(net, 1, 3, strides, None, 'SAME', True, 'dwc2')
        net = groupconv2d(net, n_filter, 1, 1, 20, act, 'SAME', True, True, 'gc2')
        net = tl.layers.ElementwiseLayer([shortcut, net], tf.add)
    return net


def resblock_2(net, n_filter, strides, act, name):
    strides = np.broadcast_to(strides, [2])
    with tf.variable_scope(name):
        if np.max(strides) > 1 or net.outputs.shape.as_list()[-1] != n_filter:
            shortcut = conv2d(net, n_filter, (3, 3), strides, None, 'SAME', True, 'shortcut')
        else:
            shortcut = net
        net = conv2d(net, n_filter, 3, strides, act, 'SAME', True, 'c1')
        net = conv2d(net, n_filter, 3, 1, act, 'SAME', True, 'c1')
        net = tl.layers.ElementwiseLayer([shortcut, net], tf.add)
    return net


def ablock(net, n_filter, strides, act, n_block, name):
    with tf.variable_scope('ab_' + name):
        net = resblock_1(net, n_filter, strides, act, 'rb_0')
        for i in range(1, n_block):
            net = resblock_1(net, n_filter, 1, act, 'rb_%d' % i)
    return net


def group_block(net, n_filter, strides, act, block_type, n_block, name):
    with tf.variable_scope('gb_' + name):
        net = block_type(net, n_filter=n_filter, strides=strides, act=act, name='b_0')
        for i in range(1, n_block):
            net = block_type(net, n_filter, 1, act, 'b_%d' % i)
    return net

