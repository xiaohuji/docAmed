# Deep Cluster

## ECCV 2018

### Deep Clustering for Unsupervised Learning of Visual Features

https://arxiv.org/pdf/1807.05520.pdf

#### 概况

![在这里插入图片描述](DeepCluster.assets/format,png#pic_center.png)

本文提出了一种端到端训练的无监督的视觉特征学习的方法. 传统的有监督的视觉特征学习的优化目标如下公式所示:
![image-20230515103201228](DeepCluster.assets/image-20230515103201228.png)

其中 f 是卷积网络, g 是多层感知分类器, x 是一个图片样本, y 是其对应的标签. 优化目标是最小化预测标签和真实标签之间的误差, 从而学习到视觉特征.

本文改进在使用 k 均值聚类的方法, 为每个样本分配一个标签, 从而实现无监督的端到端学习. 如下图所示, 对于卷积后生成的特征进行 k 均值聚类, 从而得到伪标签.
————————————————
https://blog.csdn.net/weipf8/article/details/105756217

#### 实现细节

**Convnet architectures.** For comparison with previous works, we use a standard AlexNet [54] architecture. It consists of five convolutional layers with 96, 256, 384, 384 and 256 filters; and of three fully connected layers. We remove the Local Response Normalization layers and use batch normalization [24]. We also consider a VGG-16 [30] architecture with batch normalization. Unsupervised methods often do not work directly on color and different strategies have been considered as alternatives [25,26]. We apply a fixed linear transformation based on **Sobel filters** to remove color and increase local contrast [19,39].

```java
 **Sobel filters：**

19：Bojanowski, P., Joulin, A.: Unsupervised learning by predicting noise. ICML (2017)

39：Paulin, M., Douze, M., Harchaoui, Z., Mairal, J., Perronin, F., Schmid, C.: Local convolutional features with unsupervised training for image retrieval. In: ICCV. (2015)
```

**Optimization.** We cluster the central cropped images features and perform data augmentation (random horizontal flips and crops of random sizes and aspect ratios) when training the network. This enforces invariance to data augmentation which is useful for feature learning [33]. The network is trained with dropout [62], a constant step size, an 2 penalization of the weights θ and a momentum of 0.9. Each mini-batch contains 256 images. For the clustering, **features are PCA-reduced to 256 dimensions, whitened and 2-normalized**. We use the k-means implementation of Johnson et al. [60]. Note that running k-means takes a third of the time because a forward pass on the full dataset is needed. One could reassign the clusters every n epochs, but we found out that our setup on ImageNet (updating the clustering every epoch) was nearly optimal. On Flickr, the concept of epoch disappears: choosing the tradeoff between the parameter updates and the cluster reassignments is more subtle. We thus kept almost the same setup as in ImageNet. We train the models for 500 epochs, which takes 12 days on a Pascal P100 GPU for AlexNet.

## CVPR 2020

### Online Deep Clustering for Unsupervised Representation Learning

https://arxiv.org/pdf/2006.10645.pdf

#### 背景

DC(Deep Clustering)在训练时交替进行“聚类”与“网络学习”，在无监督表示学习领域达到了很好的效果，但**其学习过程是不稳定的**

这主要是由于DC的**离线学习机制**，在不同的epoch中样本标签发生改变，导致网络学习不稳定。

![image-20230515133555272](DeepCluster.assets/image-20230515133555272.png)

#### 概况

为了解决这个问题，本文提出了ODC(Online Deep Clustering)的在线学习机制，同步执行“聚类”与“学习”而不是交替，这能保证分类器稳定更新的同时，簇心也稳步演变。通过使用两个存储器memory来实现，一个sample memory存储样本特征与标签，另一个centroids存储簇中心特征。
————————————————
https://blog.csdn.net/qq_36560894/article/details/114455070

![image-20230515133129762](DeepCluster.assets/image-20230515133129762.png)

# trick
