from model_utils import *
import vae_net

act = lambda x: tl.act.leaky_twice_relu6(x, 0.1, 0.1)

def get_discriminator(img, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
        net = tl.layers.InputLayer(img)

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
        net = tl.layers.DenseLayer(net, 64, None, name='out')

        # 自编码器
        net2, net2_output = vae_net.get_decoder(net.outputs, None, reuse)

        net = tl.layers.merge_networks([net2, net])

    return net, net.outputs


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 32, 32, 1])
    discriminator, discriminator_output = get_discriminator(x, False)
    print(discriminator_output)
