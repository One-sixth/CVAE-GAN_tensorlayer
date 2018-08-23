from model_utils import *

act = lambda x: tl.act.leaky_twice_relu6(x, 0.1, 0.1)

def get_classifier(img, reuse):
    with tf.variable_scope('classifier', reuse=reuse):
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
        net = tl.layers.DenseLayer(net, 10, act, name='out')

    return net, net.outputs


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 32, 32, 1])
    classifier, classifier_output = get_classifier(x, False)
    print(classifier_output)