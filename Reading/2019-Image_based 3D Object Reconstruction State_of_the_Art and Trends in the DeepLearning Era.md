## Image-based 3D Object Reconstruction:State-of-the-Art and Trends in the DeepLearning Era
> 基于图像的三维物体重建:最先进的技术和趋势在深度学习的时代

***
### 摘要

&emsp;&emsp;3D重建是一个计算机视觉、计算机图形学和机器学习邻域已经探索了几十年长期存在的不适定问题。从2015年开始，使用卷积神经网络(CNN)进行基于图像的三维重建引起了越来越多的关注，并展示了令人印象深刻的性能。考虑到这个快速发展的新时代，本文对这一领域的最新发展作了全面的综述。我们的工作重点是使用深度学习技术，通过单幅或多幅RGB图像来预测一般物体的三维形状。我们根据 **形状表示** 、**网络架构** 和它们使用的 **训练机制** 来组织文献。虽然这项调查的目的是为重建一般对象的方法，但是我们也回顾了一些最近的集中于特定的对象类的工作，如人体形状和人脸。我们对一些关键论文的表现进行了分析和比较，总结了该领域存在的一些问题，并对未来的研究方向进行了展望。

**关键词** - 三维重建、深度估计、SLAM、Sfm、CNN、深度学习、LSTM、三维人脸、三维人体、三维视频
> 3D Reconstruction, Depth Estimation, SLAM, SfM, CNN, Deep Learning, LSTM, 3D face, 3D Human Body, 3D Video

***
### <text id='section-1'>1 简介</text>

&emsp;&emsp;基于图像的三维重建的目标是从一个或多个二维图像中推断出物体和场景的三维几何和结构。这个长期存在的不适定问题是机器人导航、物体识别和场景理解、三维建模和动画、工业控制和医疗诊断等许多应用的基础。  

&emsp;&emsp;从二维图像中恢复丢失的维数一直是经典 **多视点立体视觉** 和 **shape-from-X** 方法的目标，已经被广泛研究了几十年。**第一代方法从几何的角度来处理这个问题**，他们专注于理解和形式化，数学上，3D到2D的投影过程，目的是设计数学或算法解决不适定的反问题。有效的解决方案通常需要使用 **精确标定** 的摄像机拍摄 **多张** 图像。例如，基于立体几何的[[1]](#cite-1)技术要求从稍微不同的视角捕获的图像之间匹配特征，然后使用三角测量原理来恢复图像像素的三维坐标。shape-of-silhouette(从轮廓中恢复形状)或shape-by-space-carving(通过空间雕刻形状)，方法[[2]](#cite-2)需要精确的分割二维轮廓。这些使三维重建得到了合理的质量的方法需要相同对象的通过精准标定的相机得到的多张图像。*然而，这在许多情况下可能不实际或不可行*。  

&emsp;&emsp;有趣的是，人类善于利用先验知识来解决这种不适定的逆问题。他们只用一只眼睛就能推断出物体的大致大小和大致的几何形状。他们甚至可以从另一个角度来猜测它的样子。我们可以做到这一点，因为所有之前看到的物体和场景都使我们能够建立先验知识，并建立物体外观的心里的模型。**第二代三维重建方法试图利用这一先验知识，将三维重建问题转化为识别问题**。深度学习技术的发展，更重要的是大型训练数据集的不断增加，导致了新一代的方法能够从单幅或多幅RGB图像中恢复物体的3D几何和结构，而无需复杂的相机校准过程。尽管这些方法是最近才出现的，但它们已经在与计算机视觉和图形相关的各种任务中显示出令人兴奋和有希望的结果。

&emsp;&emsp;在这片文章中，我们提供了一个全面和结构化的回顾，使用深度学习技术的三维物体重建的最新进展。我们首先关注 **一般形状**，然后讨论特定的情况，如 **人体形状、面部重建** 和 **3D场景解析**。我们已经收集了149篇论文，这些论文自2015年以来发表在计算机视觉、计算机图形学、机器学习会议和期刊上。其目标是帮助读者在这个过去几年获得了重要发展势头的新兴领域做一个导航。与现有文献相比，这篇文章的主要贡献如下：
> 不包括一些ICCV2019及许多CVPR2019的文章

  1. 据我们所知，这是第一篇调查以图像为基础利用深度学习进行三维物体重建的论文的文献
  2. 我们涵盖了与这邻域有关的当代文献。我们对自2015年以来出现了149种方法提出了一个全面的回顾
  3. 我们对使用深度学习进行三维重建的各个方面进行了全面的回顾和深入的分析，包括训练数据、网络架构的选择及其对三维重建结果的影响、训练策略和应用场景
  4. 我们提供了一个关于对一般物体三维重建方法的性能和表现的比较性的总结。我们介绍了88种通用的三维物体重建算法，11种与三维人脸重建相关的方法，6种三维人体形态重建方法。
  5. 我们用表格的形式对这些方法进行了比较总结

本文的其余部分组织如下：
  * **[第二节](#section-2)** 提出问题并制定分类
  * **[第三节](#section-3)** 回顾潜在空间和输入编码机制
  * **[第四节](#section-4)** 体素重建调查(volumetric reconstruction)
  * **[第五节](#section-5)** 侧重于基于表面的技术
  * **[第六节](#section-6)** 展示了一些最先进的技术如何使用额外的线索来提高三维重建的性能
  * **[第七节](#section-7)** 讨论训练过程
  * **[第八节](#section-8)** 专注于特定的物体，如人体形状和脸
  * **[第九节](#section-9)** 总结了最常用的数据集来训练、测试和评估各种基于深度学习的三维重建算法的性能
  * **[第十节](#section-10)** 比较和讨论了几种关键方法的性能
  * **[第十一节](#section-11)** 讨论未来可能的研究方向
  * **[第十二节](#section-12)** 以一些重要的评语结束论文。

<p id='cite-1'>[1] R. Hartley and A. Zisserman, Multiple view geometry in computer vision. Cambridge university press, 2003</p>
<p id='cite-2'>[2] A. Laurentini, “The visual hull concept for silhouette-based image understanding,” IEEE TPAMI, vol. 16, no. 2, pp. 150–162, 1994</p>

***
### <p id='section-2'>2 问题陈述和分类</p>

&emsp;&emsp;让$I=\{I_k,k=1,\cdots,n\}$ 为一组$n\geq1$的关于一个或多个物体$X$的RGB图片。三维重建可以被描述为学习一个预测器$f_\theta$能够推断出一个形状 $\hat{X}$ 与未知的形状 $X$足够接近的过程。换句话说，函数$f_\theta$是最小化重建目标$L(I)=d(f_\theta(I), X)$ 的结果。这里 $\theta$是一组$f$的参数，$d(\cdot,\cdot)$ 是一种目标形状$X$与重建形状$f(I)$ 之间特定的距离。重建目标$L$在深度学习文献中也称为损失函数。

&emsp;&emsp;这个调查根据 **输入$I$**、**输出的表示**、**在训练和测试中使用深度神经网络结构来近似预测因子$f$**、**使用的训练方法** 和 **他们的监督** 的性质来讨论和分类当前的技术状态，见[表1](#table-1)查看一个可视化的摘要。特别地，输入$I$可以为
  - 单幅图像
  - 使用RGB相机拍摄的多幅图像，其内部和外部参数可以是已知的，也可以是未知的
  - 一段视频序列(e.g.具有时间相关性的图像序列)  

第一种情况非常具有挑战性，因为三维重建的模糊性。当输入是视频流时，可以利用时间相关性来实现三维重建，同时确保重建视频流的所有帧是平滑和一致的。此外，输入可以描绘一个或多个属于已知或未知形状类别的3D对象。它还可以包含额外的信息，如轮廓、分割掩码和语义标签，作为引导重建的先验。

<center id='table-1'>
  <table>
    <tr>
      <td rowspan='2'><strong>输入</strong></td>
      <td>训练</td>
      <td>1 vs. muti RGB<br/>三维GT<br/>分割</td>
      <td rowspan='2'>单个 vs 多个物体<br/>一致的 vs 混乱的背景</td>
    </tr>
    <tr>
      <td>测试</td>
      <td>1 vs muti RGB<br/>分割</td>
    </tr>
    <tr>
      <td rowspan='3'><strong>输出</strong></td>
      <td>体素</td>
      <td colspan='3' align='center'>高 vs 低分辨率</td>
    </tr>
    <tr>
      <td>表面</td>
      <td colspan='3' align='center'>参数化、模板变形、点云</td>
    </tr>
    <tr>
      <td colspan='4' align='center'>直接 vs 间接</td>
    </tr>
    <tr>
      <td rowspan='3'><strong>网络结构</strong></td>
      <td colspan='2' align='center'><strong>训练结构</strong></td>
      <td colspan='2' align='center'><strong>测试结构</strong></td>
    </tr>
    <tr>
      <td colspan='2' align='center'>Encoder-Decoder<br/>TL-Net<br/>(Conditional)GAN</td>
      <td align='center'>Encoder-Decoder</td>
    </tr>
    <tr>
      <td colspan='2' align='center'>3D-VAE-GAN</td>
      <td align='center'>3D-VAE</td>
    </tr>
    <tr>
      <td rowspan='2'><strong>训练</strong></td>
      <td>监督程度</td>
      <td colspan='2'>2D vs 3D监督，弱监督</td>
    </tr>
    <tr>
      <td>训练过程</td>
      <td colspan='2'>对抗训练,对应2D-3D嵌入,与其他任务联合培训</td>
    </tr>
  </table>
  表1 - 基于深度学习的图像三维物体重建分类方法
</center><br/>

&emsp;&emsp;*输出的表示* 对网络体系结构的选择至关重要。这也影响了重建的计算效率和质量。特别地

  - **体素表示** 在早期基于深度学习的三维重建技术中被广泛采用，允许使用规则体素网格对三维形状进行参数化。因此，在图像分析中使用的2D卷积可以很容易地扩展到3D。然而，它们在内存需求方面非常昂贵，而且只有少数技术可以达到亚体素精度。
  - **基于表面的表示** 其他文章探索了基于表面的表现形式，如网格和点云。虽然这样的表示对于存储高效的，但它不是规则结构，因此不容易适合深度学习架构。
  - **中间媒介** 而一些三维重建算法则是从三维几何的角度通过RGB图像直接来预测物体的三维几何形状。其他的则将问题分解成连续的步骤，每一步都预测一个中间表示。

各种网络结构被用来实现预测器$f$。在训练和测试过程中可以使用的不同的骨干架构由一个编码器$h$跟随者一个解码器$g$构成，e.g.$f=g\circ h$。编码器使用一系列的卷积和池化操作，接着全连接层的神经元层，将输入映射到一个称为特征向量或编码的潜在变量$x$中。解码器，也称为生成器，通过使用 **全连接层** 或 **反卷积网络** *(卷积`convolution`和上采样`upsampling`操作的序列，也称为上卷积`upconvolution`)* 将特征向量解码成所需的输出。前者适用于 **非结构化的输出**，如三维点云；而后者用于重建 **体素网格或参数化表面**。自从引入这个基本架构以来，已经通过改变架构提出了几个扩展，或通过级联多个模块，每一个实现一个特定的任务。如ConvNet vs ResNet,CNN vs GAN,CNN vs VAE以及2D vs 3D卷积。

&emsp;&emsp;虽然网络的体系结构及其构建块很重要，但其性能在很大程度上取决于它的训练方式。在这个调查中，我们将着眼于:

  - **数据集** 目前有各种数据集可供训练和评估基于深度学习的三维重建。其中一些使用真实数据，另一些是CG生成的。
  - **损失函数** 损失函数的选择对重建质量影响很大。它还规定了监督的程度。
  - **训练过程以及监督程度** 有些方法需要用相应的3D模型标注真实图像，这是非常昂贵的。其他方法依赖于真实数据和合成数据的结合。剩下的通过损失函数利用容易获取的监督信号避免了完全的三维监督。

下面几节将详细讨论这些方面。

***
### <p id='section-3'>3 编码阶段</p>

&emsp;&emsp;基于深度学习的三维重建算法将输入$I$编码至一个特征向量$x=h(I)\in{\chi}$，其中 $\chi$是一个潜在空间。一个好的映射函数$h$需要满足以下的特性：

  - 两个表示相似三维物体的输入$I_1$和$I_2$被映射为$x_1和x_2\in\chi$需要在潜在空间中互相接近
  - 一个关于$x$的小波动 $\partial{x}$ 需要对应于输入形状的一个小扰动
  - 由$h$产生的潜在表示应该对摄像机姿态等外部因素保持不变
  - 三维模型及其对应的二维图像需要被映射到潜在空间的同一点上。这将确保表示是不模糊的，从而有助于重建。

前两个条件已经通过使用将输入映射到离散([Section 3.1](#section-3.1))或连续([Section 3.2](#section-3.2))的潜在空间的编码器解决。它们可以是平整的，也可以是层次化的([Section 3.3](#section-3.3))。第三点通过使用分离表示`disentangled representations`([Section 3.4](#section-3.4))解决。最后一点已经通过在训练阶段使用TL结构来解决。这在[Section 7.3.1](#section-7.3.1)中作为文献中使用的许多训练机制之一进行了介绍。[表2](#table-2)总结了这种分类法。

<center id='table-2'>
  <table>
    <tr>
      <th>潜在空间</th>
      <th>结构</th>
    </tr>
    <tr>
      <td>离散<a href='#section-3.1'>(3.1)</a> vs 连续<a href='#section-3.2'>(3.2)</a></td>
      <td rowspan='3'>ConvNet,ResNet,<br/>FC,3D-VAE</td>
    </tr>
    <tr>
      <td>平整 vs 层次化<a href='#section-3.3'>(3.3)</a></td>
    </tr>
    <tr>
      <td>分离表示<a href='#section-3.4'>(3.4)</a></td>
    </tr>
  </table>
  表2 - 编码阶段的分类。FC: 全连接层。VAE: 差分自编码器。
</center>

#### <p id='section-3.1'>3.1 离散潜在空间</p>

&emsp;&emsp;Wu等人在他们的开创性著作[[3]](#cite-3)中介绍了3D ShapeNet，这是一种映射3D形状的编码网络，表示尺寸为$30^3$的离散体积网格，映射到大小为4000*1的隐藏空间中。它的核心网络由$n_{conv}=3$个卷积层组成(每一个使用了三维卷积滤波器)，紧接着$n_{fc}=3$个全连接层。这个标准的香草架构已经被使用于三维形状分类和检索[[3]](#cite-3)，以及三维重建表示为体素网格[[3]](#cite-3)的深度图。它还被用于三维编码的分支，三维重建网络训练中的TL体系结构,详见[7.3.1节](#section-7.3.1)。

&emsp;&emsp;将输入图像映射到潜在空间的2D编码网络遵循与3D ShapeNet[[3]](#cite-3)相同的架构，但使用2D卷积[[4]](#cite-4), [[5]](#cite-5), [[6]](#cite-6), [[7]](#cite-7), [[8]](#cite-8), [[9]](#cite-9), [[10]](#cite-10), [[11]](#cite-11)。早期的工作层类型和层数不同。例如，Yan等人[[4]](#cite-4)使用$n_{conv}=3$个卷积层分别为64, 128, 256个通道，另外$n_{fc}=3$个全连接层分别包含1024, 1024, 512个神经元。Wiles和Zisserman[[10]](#cite-10)使用$n_{conv}=3$个卷积层分别为3, 64, 128, 256, 128, 160个通道。其他作品增加了池化层[[7]](#cite-7)，[[12]](#cite-12)，和leaky Rectified Linear Units(ReLU)[[7]](#cite-7)，[[12]](#cite-12)，[[13]](#cite-13)。比如，Wiles和Zisserman[[10]](#cite-10)在除了第一层后和最后一层前的每一对卷积层中间使用了最大池化层。ReLU层由于使梯度在反向传播时从不为0提高了训练效果。

&emsp;&emsp;二维和三维的编码网络都可以通过使用深度残差网络(ResNet)来实现，这种网络添加了卷积层之间的残差连接，如[[6]](#cite-6), [[7]](#cite-7), [[9]](#cite-9)。与VGGNet[[15]](#cite-15)等传统网络相比，ResNets改善并加快了深度网络的学习过程。

<p id='cite-3'>[3] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao, “3D shapenets: A deep representation for volumetric shapes,” in IEEE CVPR, 2015, pp. 1912–1920.</p>
<p id='cite-4'>[4] X. Yan, J. Yang, E. Yumer, Y. Guo, and H. Lee, “Perspective Transformer Nets: Learning single-view 3D object reconstruction without 3D supervision,” in NIPS, 2016, pp. 1696–1704.</p>
<p id='cite-5'>[5] E. Grant, P. Kohli, and M. van Gerven, “Deep disentangled representations for volumetric reconstruction,” in ECCV, 2016, pp. 266–279.</p>
<p id='cite-6'>[6] J. Wu, Y. Wang, T. Xue, X. Sun, B. Freeman, and J. Tenenbaum, “MarrNet: 3D shape reconstruction via 2.5D sketches,” in NIPS, 2017, pp. 540–550.</p>
<p id='cite-7'>[7] C. B. Choy, D. Xu, J. Gwak, K. Chen, and S. Savarese, “3DR2N2: A unified approach for single and multi-view 3D object reconstruction,” in ECCV, 2016, pp. 628–644.</p>
<p id='cite-8'>[8] S. Tulsiani, T. Zhou, A. A. Efros, and J. Malik, “Multi-view supervision for single-view reconstruction via differentiable ray consistency,” in IEEE CVPR, vol. 1, no. 2, 2017, p. 3.</p>
<p id='cite-9'>[9] X. Z. Xingyuan Sun, Jiajun Wu and Z. Zhang, “Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling,” in IEEE CVPR, 2018.</p>
<p id='cite-10'>[10] O. Wiles and A. Zisserman, “SilNet: Single-and Multi-View Reconstruction by Learning from Silhouettes,” BMVC, 2017.</p>
<p id='cite-11'>[11] S. Tulsiani, A. A. Efros, and J. Malik, “Multi-View Consistency as Supervisory Signal for Learning Shape and Pose Prediction,” in IEEE CVPR, 2018.</p>
<p id='cite-12'>[12]A. Johnston, R. Garg, G. Carneiro, I. Reid, and A. van den Hengel, “Scaling CNNs for High Resolution Volumetric Reconstruction From a Single Image,” in IEEE CVPR, 2017, pp. 939–948.</p>
<p id='cite-13'>[13]G. Yang, Y. Cui, S. Belongie, and B. Hariharan, “Learning singleview 3d reconstruction with limited pose supervision,” in ECCV, 2018.</p>
<p id='cite-14'>[14]K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in IEEE CVPR, 2016, pp. 770–778.</p>
<p id='cite-15'>[15]K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, 2014.</p>

#### <p id='section-3.2'>3.2 连续潜在空间</p>

#### <p id='section-3.3'>3.3 分层潜在的空间</p>

#### <p id='section-3.4'>3.4 分离表示</p>
