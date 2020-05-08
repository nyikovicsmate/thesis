## Image Resolution Enhancement Using Multi-Step Reinforcement Learning 
_An MSc [thesis work](https://github.com/nyikovicsmate/thesis/blob/master/documentation/thesis_description.pdf "thesis description"). - 2020._  

As a quest to achieve super-resolution I've taken the novel approach of constructing a Reinforcement Learning 
based model using Tensorflow 2.0. But first, to better undersand the problem, I've studied and implemented 
several upsampling models which are detailed in the sections below. My contribution also extends to creating 
[easy-to-use notebooks](https://github.com/nyikovicsmate/thesis/tree/master/utils/datasets)
for downloading and preprocessing popular datasets (BSD68, BSD500, MIT-Adobe FiveK, MS COCO, BDD100k, Cityscapes, Kitti).
Based off of Tensorflow's dataset implementation, I've implemeted a custom, reusable, memory-efficient dataset class for
handling images (only HDF format is supported as of yet).

### Models

#### Supervised

---

##### Pre-upsampling model (SRCNN)

Reference: _[Image Super-Resolution Using Deep
Convolutional Networks](https://arxiv.org/abs/1501.00092) - Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang_

First, an LR image is upsampled using traditional Bicubic interpolation to get the HR* representation. Then this coarse 
image is fed to the model, which is aimed to reconstruct missing/lost details. Fast, but the achievable results are 
hidered by the upsampling algorithm.
 
![process](https://raw.githubusercontent.com/nyikovicsmate/thesis/master/documentation/img/pre0.png)

![results](https://raw.githubusercontent.com/nyikovicsmate/thesis/master/documentation/img/pre_r.png)

##### Post-upsampling model (FSRCNN)

Reference: _[Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367) - Chao Dong, Chen Change Loy, Xiaoou Tang_

A model largely similar to the pre-upsampling model with the difference of feeding the LR image to the model instead 
of the upsampled one. This is done in hope of trying to mitigate the effect of the upsamping algorithm by using the 
convolutional layers to extract features before they are potentionally lost. Fast, but the achievable results are 
still hidered by the upsampling.

![results](https://raw.githubusercontent.com/nyikovicsmate/thesis/master/documentation/img/post_r.png)

##### Progressive upsampling model (LapSRN)

Reference: _[Fast and Accurate Image Super-Resolution with
Deep Laplacian Pyramid Networks](https://arxiv.org/abs/1710.01992) - Wei-Sheng Lai, Jia-Bin Huang, Narendra Ahuja, Ming-Hsuan Yang_

The model uses a pyramid structure. At each pyramid level, the model consists of a feature embedding sub-network for 
extracting non-linear features, transposed convolutional layers for upsampling feature maps and images, and a 
convolutional layer for predicting the sub-band residuals. As the network structure at each level is highly similar, 
it shares the weights of those components across pyramid levels to reduce the number of network parameters. This way 
the upsampling can take place in multiple steps, which makes the network capable to preserve more of the original image's
features, especially for large (4x, 8x) scaling factors. 

![process](https://raw.githubusercontent.com/nyikovicsmate/thesis/master/documentation/img/prog0.png)

![results](https://raw.githubusercontent.com/nyikovicsmate/thesis/master/documentation/img/prog_r.png)

##### Iterative upsampling model (LapSRN)

Reference: _[Deep Back-Projection Networks for Single Image Super-resolution](https://arxiv.org/abs/1904.05677) - Muhammad Haris, Greg Shakhnarovich, Norimichi Ukita_

The model consists of iterative up- and down-sampling layers. These layers are formed as a unit providing an error 
feedback mechanism for projection errors. 

![process](https://raw.githubusercontent.com/nyikovicsmate/thesis/master/documentation/img/iter0.png)

![results](https://raw.githubusercontent.com/nyikovicsmate/thesis/master/documentation/img/iter_r.png)

#### Adversarial

---

TODO

#### Reinforcement

---

TODO     


### How to use

#### Predict

For prediction purposes, I've included pre-trained models. To use them, all you have to do is:
1. Instantinate a network e.g.: `network = IterativeSamplingNetwork()` 
2. Load the pre-trained state/weights: `network.load_state()`
3. Get your images as a 4D numpy array, shaped (count, height, width, channels) - **currently only grayscale (channels=1) images are supported**
4. Predict: `predicted_images = network.predict(your_4d_numpy_image_array)` The `predict` method supports an optional `scaling_factor: int` parameter. The default scaling is 2x, but most of the models suppot 4x, 8x as well.  

Complete example:

```
import numpy as np
import cv2

from src.networks.supervised.pre_upsampling_network import PreUpsamplingNetwork

network = PreUpsamplingNetwork()
network.load_state()

img = cv2.imread(<path to your image>, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (35, 35), cv2.INTER_CUBIC)    # network works with 35x35 images
img = img[np.newaxis, :, :, np.newaxis]   # get the 4D shaped image numpy array
img = img / 255.0	# normalize image
pred = network.predict(img, 2)  # upsample the image by 2x
pred = pred * 255.0   # revert normalization

# display the image
cv2.imshow("2x image", pred[0].astype(np.uint8))     # pred is also a 4D array, we want to display the first (and only) image
cv2.waitKey()
```

#### Train

 TODO