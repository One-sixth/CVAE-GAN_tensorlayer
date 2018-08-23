from time import time
t1 = time()
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import vae_net
import discriminator_net
import classifier_net
# from skimage.transform import resize
import my_py_lib.utils
import os
from progressbar import progressbar
print('加载 tf 耗时', time()-t1)

t1 = time()
# 加载和处理数据
x_dataset, y_dataset = tl.files.load_mnist_dataset((-1, 28, 28, 1))[:2]
# x_dataset = np.array([resize(i, (32, 32), 3, 'constant', 0, True, False, False) for i in x_dataset], np.float32)
x_dataset = tl.prepro.threading_data(x_dataset, tl.prepro.imresize, size=(32, 32)) / 255.
print('加载数据集耗时', time() - t1)

img = tf.placeholder(tf.float32, [None, 32, 32, 1])
classes_label = tf.placeholder(tf.int32, [None, ])
kt = tf.Variable(0, False, dtype=tf.float32, name='kt')
lr_placeholder = tf.placeholder(tf.float32, name='lr_placeholder')
gamma = 0.5
lamda = 0.5
# for test
samples_placeholder = tf.placeholder(tf.float32, [None, 64])

encoder, encoder_output = vae_net.get_encoder(img, classes_label, False)
decoder, decoder_output = vae_net.get_decoder(encoder_output[0], classes_label, False)
# for test
_, samples_decoder_output = vae_net.get_decoder(samples_placeholder, classes_label, True)

classifier_real, classifier_real_output = classifier_net.get_classifier(img, False)
classifier_fake, classifier_fake_output = classifier_net.get_classifier(decoder_output, True)
discriminator_real, discriminator_real_output = discriminator_net.get_discriminator(img, False)
discriminator_fake, discriminator_fake_output = discriminator_net.get_discriminator(decoder_output, True)

def get_kl_loss(mean, log_sigma):
    # 限制 exp 的值，以免爆炸
    log_sigma = tf.minimum(log_sigma, 20)
    return tf.reduce_mean(-0.5 * tf.reduce_sum(1 + log_sigma - tf.square(mean) - tf.exp(log_sigma), 1))

kl_loss_op = get_kl_loss(*encoder_output[1:])

d_loss_real = tf.reduce_mean(tf.abs(discriminator_real_output - img))
d_loss_fake = tf.reduce_mean(tf.abs(discriminator_real_output - decoder_output))

d_loss_op = d_loss_real - kt * d_loss_fake

kt_update_op = tf.assign(kt, tf.clip_by_value(kt + lamda * (gamma * d_loss_real - d_loss_fake), 0., 1.))
m_global = d_loss_real + tf.abs(gamma * d_loss_real - d_loss_fake)

g_loss_img = tf.reduce_mean(tf.abs(img - decoder_output))
g_loss_disc = d_loss_fake
g_loss_class = tf.losses.sigmoid_cross_entropy(tf.one_hot(classes_label, 10, dtype=tf.int32), classifier_fake_output)

g_loss_op = g_loss_img + g_loss_disc + g_loss_class

c_loss_op = tf.losses.sigmoid_cross_entropy(tf.one_hot(classes_label, 10, dtype=tf.int32), classifier_real_output)

c_top_1 = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(classifier_real_output, classes_label, 1)))


tf.summary.scalar('g/kl_loss_op', kl_loss_op)
tf.summary.histogram('g/z', encoder_output[0])
tf.summary.scalar('g/img', g_loss_img)
tf.summary.scalar('g/disc', g_loss_disc)
tf.summary.scalar('g/class', g_loss_class)
tf.summary.image('g/ori_img', tf.cast(tf.clip_by_value(img*255, 0, 255), tf.uint8), 3)
tf.summary.image('g/gen_img', tf.cast(tf.clip_by_value(decoder_output*255, 0, 255), tf.uint8), 3)
tf.summary.image('d/real_img', tf.cast(tf.clip_by_value(discriminator_real_output*255, 0, 255), tf.uint8), 3)
tf.summary.image('d/fake_img', tf.cast(tf.clip_by_value(discriminator_fake_output*255, 0, 255), tf.uint8), 3)
tf.summary.scalar('d/real', d_loss_real)
tf.summary.scalar('d/fake', d_loss_fake)
tf.summary.scalar('c/loss', c_loss_op)
tf.summary.scalar('c/in_top_1', c_top_1)
tf.summary.scalar('misc/kt', kt)
tf.summary.scalar('misc/m_global', m_global)
tf.summary.scalar('misc/lr', lr_placeholder)


os.makedirs('imgs', exist_ok=True)
os.makedirs('logs', exist_ok=True)
merge_summary_op = tf.summary.merge_all()
sum_file = tf.summary.FileWriter('logs', tf.get_default_graph())

g_optim = tf.train.AdamOptimizer(lr_placeholder).minimize(kl_loss_op + g_loss_op, var_list=encoder.all_params + decoder.all_params)
d_optim = tf.train.AdamOptimizer(lr_placeholder).minimize(d_loss_op + c_loss_op, var_list=discriminator_real.all_params + classifier_real.all_params)

sess = my_py_lib.utils.get_auto_grow_session()
sess.run(tf.global_variables_initializer())

tl.files.load_and_assign_npz(sess, 'encoder.npz', encoder)
tl.files.load_and_assign_npz(sess, 'decoder.npz', decoder)
tl.files.load_and_assign_npz(sess, 'classifier.npz', classifier_real)
tl.files.load_and_assign_npz(sess, 'discriminator.npz', discriminator_real)


n_epoch = 200
batch_size = 30
n_batch = int(np.ceil(len(x_dataset)/batch_size))
lr = 0.001

for e in range(n_epoch):
    for b in progressbar(range(n_batch)):
        feed_dict = {img: x_dataset[b*batch_size:(b+1)*batch_size], classes_label: y_dataset[b*batch_size:(b+1)*batch_size], lr_placeholder: lr}

        if b == 5 and e == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, sum = sess.run([[g_optim, d_optim, kt_update_op], merge_summary_op], feed_dict, run_options, run_metadata)
            sum_file.add_run_metadata(run_metadata, 'step%d' % (e*n_batch+b))
            sum_file.add_summary(sum, e*n_batch+b)
        else:
            _, sum = sess.run([[g_optim, d_optim, kt_update_op], merge_summary_op], feed_dict)
            sum_file.add_summary(sum, e*n_batch+b)

        if b % 20 == 0:
            lr *= 0.95

        if b % 100 == 0:
            tl.files.save_npz(encoder.all_params, 'encoder.npz', sess)
            tl.files.save_npz(decoder.all_params, 'decoder.npz', sess)
            tl.files.save_npz(classifier_real.all_params, 'classifier.npz', sess)
            tl.files.save_npz(discriminator_real.all_params, 'discriminator.npz', sess)

            samples_z = np.random.normal(size=[36, 64])
            samples_c = np.random.randint(0, 10, [36,], np.int32)
            samples_imgs = sess.run(samples_decoder_output, {samples_placeholder: samples_z, classes_label: samples_c})
            samples_imgs = np.asarray((samples_imgs - np.min(samples_imgs)) / (np.max(samples_imgs) - np.min(samples_imgs)) * 255, np.uint8)
            tl.vis.save_images(samples_imgs, (6, 6), 'imgs/%d_%d.jpg' % (e, b))
