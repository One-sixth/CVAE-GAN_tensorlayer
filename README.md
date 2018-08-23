# CVAE-GAN_tensorlayer
A CVAE-GAN implementation with tensorlayer.<br>

# Examples
## training VAE output
![0_0](https://github.com/One-sixth/CVAE-GAN_tensorlayer/blob/master/imgs/0_0.jpg)
![0_1600](https://github.com/One-sixth/CVAE-GAN_tensorlayer/blob/master/imgs/0_1600.jpg)
![1_1600](https://github.com/One-sixth/CVAE-GAN_tensorlayer/blob/master/imgs/1_1600.jpg)
![3_600](https://github.com/One-sixth/CVAE-GAN_tensorlayer/blob/master/imgs/3_600.jpg)
## testing VAE output and Recon output
![0](https://github.com/One-sixth/CVAE-GAN_tensorlayer/blob/master/test_output/0.jpg)
![1](https://github.com/One-sixth/CVAE-GAN_tensorlayer/blob/master/test_output/7.jpg)
![2](https://github.com/One-sixth/CVAE-GAN_tensorlayer/blob/master/test_output/14.jpg)
![3](https://github.com/One-sixth/CVAE-GAN_tensorlayer/blob/master/test_output/31.jpg)


# Dependent
tensorflow<br>
tensorlayer<br>
numpy<br>
progressbar2<br>

My test environment is tensorflow-gpu-1.10, tensorlayer-1.91, gtx970m-3g.<br>

# Some Problem and Attention
emmm... This implementation may have some differences with the page.<br>
I try WGAN-GP and LS-GAN 's loss, but the result is not good. Maybe my code is wrong.<br>
Then I change discriminator become autoencoder and try BEGAN 's loss, It is look good.<br>
<br>
The VAE output after training is still blurry, but I found that the reconstructed image of the discriminator can make the VAE output clearer.<br>
The main network structure is my simple custom resnet, which should be different from DCGAN.<br>

# Training Log
My training process output and logs can be found in the imgs and logs folders respectively.<br>
The training log has been compressed and only needs to be decompressed.<br>

# How to use
## Test
```
python3 test.py
```
I uploaded model weights in the repository. You can test with my weight right away.<br>
Test output can be found in the test_output folder.<br>
The upper three lines of each image are the VAE output, and the next three lines are the reconstructed output of the discriminator.<br>

## Train
```
python3 train.py
```
Just simple start train.py. The train will be start.<br>
It will automatically reuse the previous weights if they are still there.<br>
*.npz files is the networks weight. If you want a new training, just delete them simply.<br>
If you see OOM, try to reduce batch_size or train with cpu.<br>

# Page Link
CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training<br>
https://arxiv.org/abs/1703.10155
