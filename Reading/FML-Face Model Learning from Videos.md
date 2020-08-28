## FML: Face Model Learning from Videos

***
### 摘要

基于单目图像的三维(人脸)重建是计算机视觉长期存在的问题。图像数据是三维人脸的二维投影，产生了深度模糊使问题不适定。现有的数据驱动的方法是基于有限的三维(人脸)扫描数据。本文提出了一种基于多帧视频的深度网络自监督训练方法

+ 学习面部识别模型的形状和外观

+ 共同学习重建三维人脸

我们的面部模型仅使用从互联网上收集的野生视频剪辑语料库进行学习。就能有无穷无尽的数据用于训练。提出了多帧一致性损失。在测试时，我们可以使用任意数量的帧，这样我们既可以执行单目重建，也可以执行多帧重建。

### 简介

不受控的场景下，场景定位、光照或介入设备都具有不确定性，这就会产生低分辨率噪声或运动模糊，相比受控场景重建更加困难。利用野外单目二维图像和视频数据进行三维人脸重建，处理面部形状识别(中性几何)、皮肤外观等问题(或反照率)和表达式，以及估计场景照明和相机参数。其中的一些属性，如反照率和光照，在单眼图像中是不容易分离的。

该人脸识别模型由两个部分组成:一个部分表示人脸识别的几何形状(模表达式)，另一个部分根据反照率表示人脸外观。与以前的大多数方法不同，我们不需要预先存在的形状标识和反照率模型作为初始化，而是从头开始学习它们的变化。

+ 一种深度神经网络，它可以从一个大的无约束图像数据库中学习面部形状和外观空间，该数据库包含每个受试者的多个图像，例如多视图序列，甚至是单目视频

+ 通过在blendshapes的零空间上的投影实现显式的blendshape和标识分离，从而实现多帧一致性损失

+ 一种新的基于Siamese网络的多帧身份一致性丢失，具有处理单目和多帧重建的能力

***
### 人脸与物体的异同

+ 基于单目图像的重建，对于人脸和物体，深度模糊产生的不适定问题都是相同的。对于三维数据方面，人脸的三维扫描数据很有限，对于物体 ShapeNet 提供了可供训练的数量的三维数据，但是希望能够不使用三维数据进行训练。

+ 只是用互联网上收集的wild视频剪辑进行学习，对于人脸能够进行面部识别以及68个特征点的定位且人脸形状具有先验且变化不大，对于物体变化就更为明显，则如果需要在wild的视频上训练进行前背景的分离是十分必要的，(1)使用分割方法对视频物体进行提取，(2)使用其他分割好的数据集。同时对于人脸需要的角度变化相较于物体更小。

+ 不受控场景下，分离出来的参数，对于人脸，现有的模型提供了可以用于预测光照参数，对于物体而言，如果表示方式为点云则光照较难表示，如果为网格则现有可微投影可以提供光照的变化

***
### 相关工作

* 多帧三维重建

多帧重建技术利用时间信息或多个视图来更好地估计三维几何。[]在多个关键帧上将多线性模型与三维地标进行拟合，并通过插值方法实现帧间的时间一致性。[]个人特定的面部形状是通过平均每帧估计参数面部模型。[]采用多视点束调整方法重建面部形状和细化表情使用特定的演员序列。[]采用多视点束调整方法重建面部形状和细化表情使用特定的演员序列。[]提出了一种无模型方法，从运动框架全局优化非刚性结构的稠密三维几何。

除了人脸，Tulsian等人[66]训练CNN预测单视图三维形状(用体素表示)使用多视图射线一致性

### 方法

解决了两个问题：参数化的面部几何和外观模型；一个面部形状，表情，反照率，刚性姿态和入射照明参数的估计器。

网络共同学习一个外观和形状识别模型(3.2节)。它还为刚体头部姿态、光照和表达式参数估计每帧的参数，以及所有帧共享的形状和外观标识参数。为此，我们提出了一套训练损失，考虑几何平滑度、光一致性、稀疏特征对齐和外观稀疏性。

在测试时，我们的网络从同一个人的任意数量的面部图像中共同重建形状、表情、反照率、姿势和光照。因此，同样的训练网络可以用于单目和多帧人脸重建

* 数据集

  404k段相同人物的视频。每段4帧，为了避免不必要的变化(例如老化和配饰)，同一个人可以在数据集中出现多次。生成由以下步骤组成

  + 由面部识别landmark裁剪

  + 我们丢弃裁剪区域小于阈值的图像或具有较低的地标检测置信度

  + 将图像变为240*240

  根据地标跟踪器获得的头部方向，我们确保头部姿态有足够的多样性

* Graph-based Face Representation

  我们提出了一种基于粗糙形状变形图和高分辨率表面网格的多层次的人脸表示方法，其中每个顶点都有一个颜色值来编码人脸的外观。这种表示法使我们能够学习基于多帧一致性的人脸几何和外观模型。下面，我们将详细解释这些组件

  - 可学习的基于图的身份模型

    对网格进行下采样简化网格，60k到521。我们的身份模型由变形图G表示，变形参数由网络回归，同时学习变形子空间基s。我们利用多帧一致性来正则化这个不适定学习问题。

  - Blendshape 表情模型

    变形表达式直接应用于高分辨率网格。高分辨率网格的顶点位置决定了形状的一致性和面部表情

  - 形状和表情的分离

    通过在习得的形状识别基和固定的blendshape基之间施加正交性来确保形状识别与面部表情的分离

  - 可学习的每个顶点外观模型

    平均的面部外观和外观的基是可学习的

* 多帧一致性面部模型学习

  每个信号塔由一个编码器组成，用于估计特定于框架的参数和标识特征映射。几何和外表参数在流中是共享的

  - 单帧参数估计

    使用卷积网络来提取底层特征。然后，应用一系列的卷积、ReLU和全连接层来回归每帧参数p[f]。

  - 多帧身份估计网络

    多帧输入的每一帧在不同的头部姿态和表情下都表现出相同的面部特征，利用这些信息并使用单一的身份估计网络，利用另外几个卷积层生成中层次特征。通过平均池的方法，将产生的中层次特征图融合到一个单一的多帧特征图中。使用平均池化可以适应任意数量的输入。然后将这个合并的特征图反馈给一个基于卷积层、ReLU和全连接层的身份参数估计网络。

* 损失函数

  - 多帧光度一致性

    方法的关键贡献之一是增强共享标识参数的多帧一致性。通过对帧施加以下光度一致性损失来做到这一点

  - 多帧landmark一致性

  - 图形级的几何平滑度

  - 外观稀疏

  - 表情正则化