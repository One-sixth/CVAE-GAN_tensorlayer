import tensorflow as tf
import tensorlayer as tl
import numpy as np
import vae_net
import discriminator_net
import os
from progressbar import progressbar

tl.logging.set_verbosity('INFO')


classes_label = tf.placeholder(tf.int32, [None, ])
z_placeholder = tf.placeholder(tf.float32, [None, 64])

decoder, decoder_output = vae_net.get_decoder(z_placeholder, classes_label, False)

discriminator_fake, discriminator_fake_output = discriminator_net.get_discriminator(decoder_output, False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

tl.files.load_and_assign_npz(sess, 'decoder.npz', decoder)
tl.files.load_and_assign_npz(sess, 'discriminator.npz', discriminator_fake)

n_samples = 32

os.makedirs('test_output', exist_ok=True)

def tr_imgs(imgs_float):
    return np.asarray((imgs_float - np.min(imgs_float)) / (np.max(imgs_float) - np.min(imgs_float)) * 255, np.uint8)

for b in progressbar(range(n_samples)):
    samples_z = np.random.normal(size=[3*6, 64])
    samples_c = np.random.randint(0, 10, [3*6,], np.int32)
    feed_dict = {z_placeholder: samples_z, classes_label: samples_c}
    samples_imgs, recon_imgs = sess.run([decoder_output, discriminator_fake_output], feed_dict)
    samples_imgs = tr_imgs(samples_imgs)
    recon_imgs = tr_imgs(recon_imgs)
    output_imgs = np.concatenate([samples_imgs, recon_imgs], 0)
    tl.vis.save_images(output_imgs, (6, 6), 'test_output/%d.jpg' % b)
