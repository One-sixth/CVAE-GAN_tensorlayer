from model_utils import *

act = lambda x: tl.act.leaky_twice_relu6(x, 0.1, 0.1)
n_hidden = 64
# block

def get_encoder(img, c, reuse):
    hw = img.shape.as_list()[1:3]
    with tf.variable_scope('encoder', reuse=reuse):
        net = tl.layers.InputLayer(img)
        # 输入类别信息
        c_net = tl.layers.OneHotInputLayer(c, 10, axis=-1, dtype=tf.float32)
        c_net = tl.layers.DenseLayer(c_net, hw[0]*hw[1], act, name='d1')
        c_net = tl.layers.ReshapeLayer(c_net, (-1, hw[0], hw[1], 1))
        net = tl.layers.ConcatLayer([net, c_net], -1)

        b_id = 0
        def get_unique_name():
            nonlocal b_id
            b_id += 1
            return str(b_id)

        net = ablock(net, 20, 1, act, 2, get_unique_name())
        net = ablock(net, 40, 2, act, 3, get_unique_name())
        net = ablock(net, 60, 2, act, 3, get_unique_name())
        net = ablock(net, 80, 2, act, 4, get_unique_name())

        net = tl.layers.GlobalMeanPool2d(net)
        net = tl.layers.DenseLayer(net, n_hidden, act, name='out')

        mean = tl.layers.DenseLayer(net, n_hidden, act, name='mean')
        log_sigma = tl.layers.DenseLayer(net, n_hidden, act, name='log_sigma')

        net = tl.layers.merge_networks([net, mean, log_sigma])

        mean = mean.outputs
        log_sigma = log_sigma.outputs
        std = log_sigma * 0.5
        noise = tf.random_normal(tf.shape(mean))

        z = mean + noise * tf.exp(tf.minimum(std, 20))

    return net, [z, mean, log_sigma]


def get_decoder(z, c, reuse):
    z_len = z.shape.as_list()[-1]
    with tf.variable_scope('decoder', reuse=reuse):
        net = tl.layers.InputLayer(z)
        # 便于鉴别器复用这里解码器，鉴别器没有c输入
        if c is not None:
            c_net = tl.layers.OneHotInputLayer(c, 10, axis=-1, dtype=tf.float32)
            c_net = tl.layers.DenseLayer(c_net, z_len, act, name='d1')
            net = tl.layers.ConcatLayer([net, c_net], -1)
            net = tl.layers.DenseLayer(net, z_len, act, name='d2')

        b_id = 0
        def get_unique_name():
            nonlocal b_id
            b_id += 1
            return str(b_id)

        net = tl.layers.ReshapeLayer(net, (-1, 1, 1, n_hidden))
        net = tl.layers.TileLayer(net, [1, 4, 4, 1])

        net = ablock(net, 80, 1, act, 3, get_unique_name())
        net = tl.layers.UpSampling2dLayer(net, (2, 2), True, 1)
        net = ablock(net, 60, 1, act, 3, get_unique_name())
        net = tl.layers.UpSampling2dLayer(net, (2, 2), True, 1)
        net = ablock(net, 40, 1, act, 3, get_unique_name())
        net = tl.layers.UpSampling2dLayer(net, (2, 2), True, 1)
        net = ablock(net, 20, 1, act, 3, get_unique_name())

        out_act = lambda x: tf.where(x<0, 0.1*x, tf.where(x>1, 0.1*x+1, x))
        net = conv2d(net, 1, 3, 1, out_act, 'SAME', False, 'out')

    return net, net.outputs


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 32, 32, 1])
    classes_label = tf.placeholder(tf.int32, [None, ])
    samples_placeholder = tf.placeholder(tf.float32, [None, 64])

    encoder, encoder_output = get_encoder(x, classes_label, False)
    print(encoder_output)
    decoder, decoder_output = get_decoder(encoder_output[0], classes_label, False)
    print(decoder_output)
