## Image-based 3D Object Reconstruction:State-of-the-Art and Trends in the DeepLearning Era
> 基于图像的三维物体重建:最先进的技术和趋势在深度学习的时代

***
### 摘要

&emsp;&emsp;3D重建是一个计算机视觉、计算机图形学和机器学习邻域已经探索了几十年长期存在的不适定问题。从2015年开始，使用卷积神经网络(CNN)进行基于图像的三维重建引起了越来越多的关注，并展示了令人印象深刻的性能。考虑到这个快速发展的新时代，本文对这一领域的最新发展作了全面的综述。我们的工作重点是使用深度学习技术，通过单幅或多幅RGB图像来预测一般物体的三维形状。我们根据 **形状表示** 、**网络架构** 和它们使用的 **训练机制** 来组织文献。虽然这项调查的目的是为重建一般对象的方法，但是我们也回顾了一些最近的集中于特定的对象类的工作，如人体形状和人脸。我们对一些关键论文的表现进行了分析和比较，总结了该领域存在的一些问题，并对未来的研究方向进行了展望。

**关键词** - 三维重建、深度估计、SLAM、Sfm、CNN、深度学习、LSTM、三维人脸、三维人体、三维视频
> 3D Reconstruction, Depth Estimation, SLAM, SfM, CNN, Deep Learning, LSTM, 3D face, 3D Human Body, 3D Video

***
### <p id='section-1'>1 简介</p>

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

#### <p id='section-3.2'>3.2 连续潜在空间</p>

&emsp;&emsp;使用前一节中介绍的编码器，潜在空间 $\chi$可能不是连续的，因此它不允许简单的插值。换句话说，如果$x_1=h(I_1),x_2=h(I_2)$，那么将不保证 $\frac{1}{2}(x_1+x_2)$ 能够被解码成一个有效的三维形状。同样，对于$x_1$的小扰动并不一定对应于输入的小扰动。变分自编码器(VAE)[[16]](#cite-16)及其3D扩展(3D- vae)[[17]](#cite-17)具有一个基本的独特特性，使其适合于生成建模:**通过设计，它们的潜在空间是连续的，允许简单的采样和插值**。它的关键思想是，将输入映射到一个平均向量 $\mu$和一个标准差为 $\sigma$的多元高斯分布的向量中，而不是一个特征向量。采样层随后取这两个向量，通过高斯分布的随机采样，生成一个特征向量$x$，作为后续解码阶段的输入。

&emsp;&emsp;这个结构被用来学习为了基于体素[[17]](#cite-17)[[18]](#cite-18)，深度[[19]](#cite-19)，表面[[20]](#cite-20)，点云[[21]](#cite-21)[[22]](#cite-22)重建的连续潜在空间。比如，在Wu[[17]](#cite-17)等人的工作中，图像编码器接收一个256*256的RGB图像并输出两个200维的向量表示在200维空间中的高斯分布的均值和标准差。相比标准编码器，3D-VAE能够在潜在空间中随机采样用来生成输入，从而由一个输入图像生成多个合理的3D形状[[21]](#cite-21)[[22]](#cite-22)。它很好地概括了在训练中没有看到的图像。

#### <p id='section-3.3'>3.3 分层潜在的空间</p>

&emsp;&emsp;Liu[[18]](#cite-18)等人将输入映射到单个潜在表示的编码器不能提取丰富的结构，从而可能导致模糊重建。为了提升重建的质量，Liu等人提出了一个更加复杂的内部变量结构`internal variable structure`，具体目标是鼓励学习一种分层排列的潜在特征检测器。该方法从一个全局潜在变量层开始，该层被硬连接到一组局部潜在变量层，每个层负责表示一个级别的特征提取。跳跃连接以自顶向下的定向方式将潜在编码连接在一起:离输入越近的局部编码代表低层特征，而离输入越远的局部编码代表高层特征。最后，将局部潜码连接到一个平整结构`flatten`上，并将其输入到特定于任务的模型中，如三维重建模型。

#### <p id='section-3.4'>3.4 分离表示</p>

&emsp;&emsp;图像中物体的外观受到多种因素的影响，如**物体的形状**、**相机的姿态** 和 **照明条件**。标准编码器将所有这些可变的部分表示为学习到的编码 $x$。这在识别和分类等应用程序中是不可取的，这些应用程序应该**不受外部因素(如姿态和照明[[23]](#cite-23))的影响**。三维重建还可以受益于分离表示，其中形状、姿态和照明用不同的编码表示。为此，Grant等人[[5]](#cite-5)提出了一种编码器将RGB图像映射成形状码和转换码。前者被解码成三维形状。编码为光照条件和姿态的后者被解码为(1)另一个80*80的RGB图像和(2)通过全连接层得到的相机姿态。 为了实现一种分离的表示，网络的训练方式为在前向过程中图像解码器接受形状码和转换码。在反向传播中，从图像解码器到形状码的信号被抑制，以迫使其仅表示形状。  
&emsp;&emsp;Zhu等人[[24]](#cite-24)遵循了同样的想法，将6DOF的姿态参数和形状参数解耦。网络通过2D输入重建标准姿态的3D形状。同时，姿态回归估计了6DOF的姿态参数，然后将这些参数应用于重建的标准形状。解耦字条和形状减少了网络中自由参数的数量，从而提高了效率。

***
### <p id='section-4'>4 体素解码</p>

&emsp;&emsp;体素表示将一个三维对象空间离散化为一个三维体素栅格$V$。离散化越精细表示越准确。随后目标为恢复一个栅格 $\hat{V}=f_\theta(I)$ 使其表示的三维形状 $\hat{X}$ 尽可能得接近未知的三维形状 $X$。使用体素网格的主要优势是许多现有的深度学习架构设计为2D图像的分析可以很容易地扩展到三维数据通过取代二维像素矩阵的三维模拟,然后处理网格使用3D卷积和池化操作。本节讨论不同的体素表示([Section 4.1](#section-4.1))，并回顾用于低分辨率([Section 4.2](#section-4.2))和高分辨率([Section 4.3]((#section-4.3)))3D重建的解码器架构。

<center>
  <img src='./imgs/2019zs-table3.png'/></br>
  表三:文献中使用的各种体素解码器的分类。圆括号中的数字是对应的节号。MDN:混合密度网络。BBX:边界框原语。Part.:分区(partitioning)
</center>

#### <p id='section-4.1'>4.1 三维形状的体素表示</p>

&emsp;&emsp;在文献中有四种主要的体素表示:  

  + **二进制占用栅格**(Binary occupancy grid) 在这个表示中，如果一个体素属于感兴趣的对象，那么它被设置为1，而背景体素被设置为0。
  + **概率占用栅格**(Probabilistic occupancy grid) 概率占用网格中的每个体素都对其属于感兴趣对象的概率进行编码。
  + **符号距离函数**(The Signed Distance Function: SDF) 每个体素都对其到最近表面点的符号距离进行编码。如果体素位于对象内部，则为负，否则为正。
  + **截断符号距离函数**(Truncated Signed Distance Function: TSDF) TSDF由Curless和Levoy[[37](#cite-37)]引入，首先估计距离传感器的瞄准线距离，形成射影符号距离场，然后在小的正负值处截断该场，计算TSDF

&emsp;&emsp;概率占用网格特别适合输出概率的机器学习算法。SDFs提供了**表面位置**和**法线方向**的明确估计。然而，从深度图等部分数据构造它们并非易事。TSDFs牺牲了从表面几何形状无限延伸的*全符号距离场*，但允许基于部分观察的场的局部更新。它们适用于从一组深度地图重建三维体素[[26]](#cite-26)[[31]](#cite-31)[[35]](#cite-35)[[38]](#cite-38)。  
&emsp;&emsp;总的来说，体素表示通过对对象周围的体素进行规律的采样来创建。Knyaz等人[[30]](#cite-30)介绍了一种称为Frustum的表示方法或体素模型，其将深度表示法和体素网格相结合。它使用了相机3D截体的切片构建体素空间，从而提供了体素切片与输入图像中的轮廓的精确对齐。  
&emsp;&emsp;同时，普通SDF和TSDF表示均离散成了一个规则的栅格。但是最近Park等人[[39]](#cite-39)提出了**深度SDF(deepSDF)** ，一个生成式深度学习模型，从输入点云生成连续的SDF场。与传统的SDF表示不同，DeepSDF可以处理有噪声和不完整的数据。它还可以表示整个类的形状。

#### <p id='section-4.2'>4.2 低分辨率三维体素重建</p>

&emsp;&emsp;一旦使用编码器学习了输入的向量表示，下一步即为学习解码函数 $g$ ，这一步被称为 `generator`(生成器)或 `generative model`(生成模型)，它将向量表示映射为一个体积像素网格。标准方法往往使用卷积解码器，也称为反卷积网络镜像了卷积编码器。Wu等人[[3]](#cite-3)是最早提出利用深度图三维体素重建方法的学者之一。Wu等人[[6]](#cite-6)提出了一种两阶段重建网络，称为MarrNet。第一阶段从输入图像使用了编解码结构来重建深度图、法线图和剪影图。这三张图被称为2.5草图，然后被用作另一种编解码架构的输入，该架构回归了一个3D体素形状。这个网络后来被Sun等人[[9]](#cite-9)扩展为也可以回归得到输入的姿态。这种两阶段方法的主要优点是，与完整的3D模型相比，深度图、法线图和轮廓图更容易从2D图像中恢复。同样地，3D模型从这三种模式中恢复要比仅从2D图像中恢复容易得多。然而，这种方法不能重建复杂的或是薄结构。

&emsp;&emsp;Wu等人的工作[[3]](#cite-3)推进了研究的进展 [[7]](#cite-7), [[8]](#cite-8), [[17]](#cite-17), [[27]](#cite-27), [[40]](#cite-40)。特别是，最近的研究试图在没有中间过程的情况下直接回归三维体素网格[[8]](#cite-8), [[11]](#cite-11), [[13]](#cite-13), [[18]](#cite-18)。Tulsiani等人[[8]](#cite-8)以及后来的[[11]](#cite-11)使用由三维反卷积层组成的解码器来预测体素的占用概率。Liu等人[[18]](#cite-18)使用三维反卷积神经网络，加上随后使用的元素级逻辑sigmoid，将学习到的潜在特征解码为三维网格占用概率。这些方法已经成功地实现了从单个或一组未经校准的相机捕获的图像进行三维重建。它们的主要优点是提出的用于二维图像分析的深度学习架构可以很容易地适应三维模型，将解码器中的二维反卷积替换为三维反卷积，并且可以在GPU上有效地实现。然而，考虑到计算复杂性和内存需求，这些方法产生的网格**分辨率较低**，通常大小为 $32^3$ 或 $64^3$ 。因此，他们无法找到细节。

#### <p id='section-4.3'>4.3 高分辨率三维体素重建</p>

&emsp;&emsp;已经有人尝试提升深度学习体系结构的高分辨率体素重建。例如，Wu等人[[6]](#cite-6)通过简单地扩展网络，就可以重建大小为 $128^3$ 的体素网格。然而，体素网格在内存需求方面非常高，且内存需求随着网格分辨率的增大而增大。本节将回顾一些用于推断高分辨率容量网格的技术，同时使计算和内存需求易于处理。我们根据这些方法是使用 **空间划分**、**形状划分**、**子空间参数化**还是**粗-精细分策略**，将它们分为四类。

##### <p id='section-4.3.1'>4.3.1 空间划分</p>

&emsp;&emsp;虽然常规的体素网格有助于卷积运算，但它们非常稀疏，因为表面元素包含在很少的体素中。一些论文利用这种稀疏性来解决分辨率问题[[32]](#cite-32), [[33]](#cite-33), [[41]](#cite-41), [[42]](#cite-42)。他们能够利用诸如八叉树之类的空间划分技术来重建尺寸为 $256^3$ 到 $512^3$ 的三维立体网格。然而，在使用八叉树结构进行基于深度学习的重建时，存在两个主要的问题。第一个是**计算性**的，因为在常规网格上操作时，卷积操作更容易实现(尤其是在gpu上)。为此，Wang等人[[32]](#cite-32)设计了O-CNN，一种新型的八叉树数据结构，有效地将八分区信息和CNN特征存储到图形内存中，并在GPU上执行整个训练和评估。O-CNN支持各种不同的CNN结构和工作与3D形状的不同表示。O-CNN的存储和计算成本随着八叉树深度的增加呈二次增长，通过约束对三维曲面所占用的八分位数的计算，使得三维CNN对于高分辨率的三维模型是可行的。

&emsp;&emsp;第二个问题源于**八叉树结构依赖于对象**这一事实。因此，理想情况下，深度神经网络需要学习如何推断八叉树的结构和内容。在本节中，我们将讨论这些问题是如何处理的。

<center>
  <img src='./imgs/2019zs-fig1.png'/>

  图1：空间划分。OctNet[[41]](#cite-41)是一个能够实现深度和高分辨率的3D CNNs的混合八叉树网格。高分辨率八叉树也能够以深度优先的[[36]](#cite-36)或宽度优先的[[33]](#cite-33)方式逐步生成。
</center>  

* **4.3.1.1**
  <p id='section-4.3.1.1'></p>

  **使用预定义的八叉树结构**：最简单的方法是假设在运行时八叉树的结构是已知的。这对于诸如语义分割这样的应用程序来说是很好的，因为在这种情况下，可以将输出八叉树的结构设置为与输入的结构相同。然而，在许多重要的场景中，如三维重建、形状建模和RGB-D融合，八叉树的结构并不预先知道，必须进行预测。为此，Riegler等人[[41]](#cite-41)提出了一种混合网格-八叉树结构，称为OctNet([图1](#fig-1)-(a))。关键思想是将一个八叉树的最大深度限制在一个很小的数上，例如3，并将几个这样的浅八叉树放在一个规则的网格上。这种表示方法实现了深度和高分辨率的三维卷积网络。然而，在测试时，Riegler等人[[41]](#cite-41)假设单个八叉树的结构是已知的。因此，尽管该方法能够以 $256^3$ 的分辨率重建三维体块，但由于不同类型的对象可能需要不同的训练，因此**缺乏灵活性**。

* **4.3.1.2**
  <p id='section-4.3.1.2'></p>

  **学习八叉树结构**：理想情况下，应该同时估计八叉树结构及其内容。可以这样做；
  + 首先，使用卷积编码器将输入编码成紧凑特征向量([Section 3](#section-3))

  + 然后，使用标准上卷积网络对特征向量进行解码，这导致输入的粗略体素重建，通常分辨率为 $32^3$ ([Section 4.2](#section-4.2))

  + 重建体素形成八叉树的根，被细分为八个部分。带有边界体素的子树被向上采样并进一步处理，以细化该八分区的区域的重建。

  + 子树被递归地处理，直到达到期望的分辨率

&emsp;&emsp;Hane等人[[36]](#cite-36)介绍了层级曲面预测`Hierarchical Surface Prediction`(HSP)，[图1](#fig-1)-(b)，它通过使用上述方法重建分辨率高达 $256^3$ 的体素栅格。在这个方法中，首先以深度优先的方式探索八叉树。另一方面，Tatarchenko等人[[33]](#cite-33)提出了八叉树生成网络(OGN)，它遵循相同的思想，但以广度优先的方式探索八叉树，见[图1](#fig-1)-(<text>c</text>)。因此，OGN生成三维形状的分层重建结果，该方法能够重建尺寸为 $512^3$ 的体素栅格。

&emsp;&emsp;Wang等人[[34]](#cite-34)引入了一种patch引导的剖分策略，核心思想是用八叉树表示三维形状，其中，每个叶节点近似一个平面曲面。为从潜在表示中推断出这样的结构，Wang等人[[34]](#cite-34)使用解码器级联，八叉树每层一个。在八叉树的每个层次，解码器预测每个细胞内的平面patch，且预测器(由全连接层组成)为每个子树预测patch近似状态，即:单元是否为'空'，是否用平面很好地近似曲面。未被很好近似的曲面patch的细胞被进一步细分并由下一层处理。这种方法将尺寸为 $256^3$ 的体素栅格的内存需求从 6.4GB[[32]](#cite-32) 减少到 1.7GB，计算时间从 1.39s 减少到 0.30s，同时保持相同的精度水平。它的主要缺陷是：**相邻的面片不能无缝重建**。另外，由于一个平面与每个八叉树细胞拟合，所以它不能很好地逼近曲面。

##### <p id='section-4.3.2'>4.3.2 占用网络</p>

&emsp;&emsp;虽然可以通过使用各种空间划分技术来减少内存占用，但是这些方法导致实现复杂，且现有的数据自适应算法仍然局限于相对较小的体素栅格($256^3到512^2$)。最近，有几篇论文提出用深度神经网络来学习三维形状的**隐式表示**(`implicit representation`)。例如，Chen和Zhang[[43]](#cite-43)提出了一个解码器来获取形状和三维点的潜在表示，并返回一个指示该点是在形状外部还是内部的值。该网络可用于重建高分辨率三维体素模型，但是当检索生成的形状时，体素CNN只需要一次学习(one-shot)就可以得到体素模型，而这种方法需要将体素栅格中的每一个点传递到网络中才可以得到其值。因此，生成样本所需的时间取决于采样分辨率。

&emsp;&emsp;Tatarchenko等人[[44]](#cite-44)引入了占有网络，它隐式地将物体的三维曲面表示为*深度神经网络分类器的连续决策边界*，该方法没有在固定分辨率下预测体素化表示，而是使用可以在**任意分辨率**下评估的神经网络来预测完全占用函数。这大大减少了训练期间的内存占用，在推理时可以使用简单的多分辨率等值面(iossurface)提取算法从学习模型中提取栅格。

##### <p id='section-4.3.3'>4.3.3 形状剖分</p>

&emsp;&emsp;另一种方法不是剖分嵌入三维形状的体素空间，而是 **将形状视为几何部件的排列**，独立地重构各个部分，然后将这些部分缝合在一起形成完整三维形状。。已经有一些工作尝试了这种方法。例如，Li等人[[42]](#cite-42)仅生成部件层次的体素表示，他们提出了一种用于形状结构的生成递归自编码器(GRASS)。这个想法将问题分成两个步骤。第一步使用递归神经网络(RvNN)编码器-解码器架构与生成对抗网络相结合，学习如何将形状结构最佳地组织为对称层次，以及如何合成部件排列。第二步，使用另一个生成模型，学习如何合成每个部件的几何图形，表示为大小为 $32^3$ 的体素栅格。因此，尽管部件生成网络仅以 $32^3$ 分辨率合成部件的三维几何图形，但单独处理单个部件的事实使其能以高分辨率重建三维形状。

&emsp;&emsp;Zou等人[[29]](#cite-29)使用称为3D-PRNN的生成递归神经网络将3D对象重建为基元(primitive)集，该结构使用编码器网络将输入转换为尺寸为 32 的特征向量；然后，由堆叠的长短期记忆模块(LSTM)和混合密度网络(MDN)组成的递归生成器从特征向量中依次预测形状的不同部分。在每个时间阶段，网络预测以特征向量和先前估计的单个基元为条件的基元组。然后，将预测部分组合在一起形成重建结果。这种方法只预测以长方体(cuboid)形式的抽象表示。将其与将重点放在单个长方体上的基于体素重建技术相结合，可以在部件层次上实现精确的三维重建。

##### <p id='section-4.3.4'>4.3.4 子空间参数化</p>

&emsp;&emsp;**所有可能的形状空间都可以使用一组正交基 $B=\{b_1,\cdots,b_n\}$ 参数化**。然后可以将每个形状 $X$ 表示为基的线性组合，即 $X = \sum^n_{i=1}a_ib_i$，其中 $a_i\in \Bbb{R}$ 。此公式简化了重建问题，不必尝试学习如何重建体素栅格 $V$ ，可以设计由全连接层组成的解码器来从潜在表示中估计系数 $a_i,i=1,\cdots,n$ ，然后恢复整个三维体。Johonston等人[[12]](#cite-12)使用离散余弦变换-II(DCT-II)定义 $B$ ，然后他们提出卷积编码器来预测低频DCT-II系数$a_i$，然后通过建的DCT(IDCT)线性逆变换替代解码网络，并将这些系数转换为实体3D像素。这对训练和推理的计算成本产生了较大的影响：使用 $n = 20^3$ 个DCT系数，网络能够在尺寸 $128^3$ 的体素栅格上重建曲面。

&emsp;&emsp;使用通用基(如 DCT基)的主要问题是：通常需要大量的基元素才能准确地表示复杂的3D对象。实践中，我们通常处理已知类别的对象，例如：人脸和三维物体，且通常可以获得训练数据，见[Section 8](#section-8)。为此，可以使用从训练数据中学习到的主成分(PCA)基来参数化形状空间[[31]](#cite-31)，这需要的基数量(大约10)比通用基数量(大约数千)少得多。

##### <p id='section-4.3.5'>4.3.5 由粗到精细化</p>

&emsp;&emsp;提高体素化方法分辨率的另一种方法是使用 **多阶段方法**[[26]](#cite-26)[[28]](#cite-28)[[35]](#cite-35)[[45]](#cite-45)[[46]](#cite-46)。第一阶段使用编解码架构恢复低分辨力的体素栅格，例如 $32^3$。后续有上采样功能的阶段，通过聚焦局部区域来细化重建。Yang等人[[46]](#cite-46)使用简单的由两个上卷积层组成的上采样模块。这个简单的上采样模块将输出的3D形状升级为 $256^3$ 的更高分辨率。

&emsp;&emsp;Wang等人[[28]](#cite-28)将重建的粗略体素栅格视为一些列图像(或切片)，然后以高分辨率逐片重建三维物体。虽然此方法允许使用二维上卷积进行有效地细化，但是用于训练的三维形状应一直对齐，以可以沿着第一个主方向对体素切片。此外，独立地重建各个切片可能会导致最终体积的*不连续和不连贯*。为捕捉切片之间的依赖关系，Wang等人使用由3D编码器、LSTM单元和2D解码器组成的长期递归卷积网络(LRCN)[[47]](#cite-47)。每次，3D编码器处理五个连续的切片以生成固定长度的矢量表示作为LSTM的输入，LSTM的输出被传递到2D卷积解码器以生成高分辨率图像，高分辨率二维图像再拼接成高分辨率三维体素。

&emsp;&emsp;其他论文没有使用 **体素切片**，而是使用 **额外的CNN模块**，这些模块关注需要细化的区域。例如，Dai等人[[26]](#cite-26)首先预测大小为 $32^3$ 的粗糙但完整的形状体素，然后通过迭代体块合成过程将其细化为 $128^3$ 的栅格，该合成过程复制粘贴从3D模型数据库中k近邻检索的体素。Han等人[[45]](#cite-45)通过引入一个执行patch级曲面优化的局部3DCNN来扩展Dai的方法。Cao等人[[35]](#cite-35)在第一阶段恢复尺寸为 $128^3$ 的体素栅格，取尺寸为 $16^3$ 的体素块并预测它们是否需要进一步细化；需要细化的块被重采样到 $512^3$ 中，并输入到另一个细化的编解码器中，并与初始粗略一起知道细化。两个子网络都采用 `U-net` 架构[[48]](#cite-48)，同时用 `OctNet` 的相应操作替换卷积和池化层[[41]](#cite-41)。

&emsp;&emsp;注意：这些方法在 **局部推理前需要单独的步骤且有时非常耗时**。例如，Dai等人[[26]](#cite-26)需要从3D数据库中尽心最近邻搜索；Han等人[[45]](#cite-45)需要3D边界检测；而Cao等人[[35]](#cite-35)需要评估一个块是否需要进一步细化。

#### <p id='section-4.4'>4.4 深度移动立方体</p>

&emsp;&emsp;虽然体素表示可以处理任意拓扑的三维形状，但它们需要一个后处理步骤。

### <p id='cite'>参考文献</p>

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
<p id='cite-16'>[16] D. P. Kingma and M.Welling, “Auto-encoding variational bayes,” ICLR, 2014.</p>  
<p id='cite-17'>[17] J. Wu, C. Zhang, T. Xue, B. Freeman, and J. Tenenbaum, “Learning a probabilistic latent space of object shapes via 3D generativeadversarial modeling,” in NIPS, 2016, pp. 82–90.</p>  
<p id='cite-18'>[18] S. Liu, C. L. Giles, I. Ororbia, and G. Alexander, “Learning a Hierarchical Latent-Variable Model of 3D Shapes,” International Conference on 3D Vision, 2018.</p>  
<p id='cite-19'>[19] A. A. Soltani, H. Huang, J. Wu, T. D. Kulkarni, and J. B. Tenenbaum, “Synthesizing 3D shapes via modeling multi-view depth maps and silhouettes with deep generative networks,” in IEEE CVPR, 2017, pp. 1511–1519.</p>  
<p id='cite-20'>[20] P. Henderson and V. Ferrari, “Learning to generate and reconstruct 3D meshes with only 2D supervision,” BMVC, 2018.</p>  
<p id='cite-21'>[21] P. Mandikal, N. Murthy, M. Agarwal, and R. V. Babu, “3DLMNet: Latent Embedding Matching for Accurate and Diverse 3D Point Cloud Reconstruction from a Single Image,” BMVC, pp. 662–674, 2018.</p>  
<p id='cite-22'>[22] M. Gadelha, R.Wang, and S. Maji, “Multiresolution tree networks for 3D point cloud processing,” in ECCV, 2018, pp. 103–118.</p>
<p id='cite-23'>[23] H. Laga, Y. Guo, H. Tabia, R. B. Fisher, and M. Bennamoun, 3D Shape Analysis: Fundamentals, Theory, and Applications. JohnWiley & Sons, 2019</p>
<p id='cite-24'>[24] R. Zhu, H. K. Galoogahi, C. Wang, and S. Lucey, “Rethinking reprojection: Closing the loop for pose-aware shape reconstruction from a single image,” in IEEE ICCV, 2017, pp. 57–65</p>
<p id='cite-25'>[25] R. Girdhar, D. F. Fouhey, M. Rodriguez, and A. Gupta, “Learning a predictable and generative vector representation for objects,” in ECCV, 2016, pp. 484–499</p>
<p id='cite-26'>[26] Ns and shape synthesis,” in IEEE CVPR, 2017, pp. 5868–5877</p>
<p id='cite-27'>[27] M. Gadelha, S. Maji, and R. Wang, “3D shape induction from 2D views of multiple objects,” in 3D Vision, 2017, pp. 402–411</p>
<p id='cite-28'>[28] W. Wang, Q. Huang, S. You, C. Yang, and U. Neumann, “Shape inpainting using 3D generative adversarial network and recurrent convolutional networks,” ICCV, 2017</p>
<p id='cite-29'>[29] C. Zou, E. Yumer, J. Yang, D. Ceylan, and D. Hoiem, “3D-PRNN: Generating shape primitives with recurrent neural networks,” in IEEE ICCV, 2017</p>
<p id='cite-30'>[30] V. A. Knyaz, V. V. Kniaz, and F. Remondino, “Image-to-Voxel Model Translation with Conditional Adversarial Networks,” in ECCV, 2018, pp. 0–0</p>
<p id='cite-31'>[31] A. Kundu, Y. Li, and J. M. Rehg, “3D-RCNN: Instance-Level 3D Object Reconstruction via Render-and-Compare,” in IEEE CVPR, 2018, pp. 3559–3568</p>
<p id='cite-32'>[32] P.-S. Wang, Y. Liu, Y.-X. Guo, C.-Y. Sun, and X. Tong, “OCNN: Octree-based convolutional neural networks for 3D shape analysis,” ACM TOG, vol. 36, no. 4, p. 72, 2017</p>
<p id='cite-33'>[33] M. Tatarchenko, A. Dosovitskiy, and T. Brox, “Octree generating networks: Efficient convolutional architectures for highresolution 3D outputs,” in IEEE CVPR, 2017, pp. 2088–2096</p>
<p id='cite-34'>[34] P.-S. Wang, C.-Y. Sun, Y. Liu, and X. Tong, “Adaptive O-CNN: a patch-based deep representation of 3D shapes,” ACM ToG, p. 217, 2018</p>
<p id='cite-35'>[35] Y.-P. Cao, Z.-N. Liu, Z.-F. Kuang, L. Kobbelt, and S.-M. Hu, “Learning to reconstruct high-quality 3D shapes with cascaded fully convolutional networks,” in ECCV, 2018</p>
<p id='cite-36'>[36] C. Hane, S. Tulsiani, and J. Malik, “Hierarchical Surface Prediction,” IEEE PAMI, no. 1, pp. 1–1, 2019</p>
