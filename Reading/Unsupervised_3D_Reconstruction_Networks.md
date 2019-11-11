## Unsupervised_3D_Reconstruction_Networks
> 无监督三维重建网络
***

* 提出了一个无监督三维重建网络**3D-URN**

  通过正交相机模型下的2D特征点重建给定目标类别实例  

  包含一个3D形状重建模块和一个旋转估计模块，分别负责

  - 从2D特征点重建

    引入了多个3D重建模块，最终三维模型由多个三维重建加权得到，这个权重也由神经网络得到。

  - 估计相机位姿

    旋转需满足正交性约束，提出了一个rotation refiner分解输出的旋转估计使其正交，并且每一步都是可微的

* SFM和NRSFM

  重建二维特征轨迹标记至三维轨迹，处理刚体以及非刚体目标。NRSFM是一个病态问题，它提升了自由度。

  shape space model和trajectory space model表明非刚体的变形可以用低秩矩阵表示，但是大多数NRSFM需要一个平滑的运动轨迹输入

* SFC

  从一个特定类别中重建结构，不需要平滑的特征点预测。不像NRSFM仅局限于人脸和人的肢体。

* LOSS

  projection loss

  the low-rank prior loss

* 文章贡献

  - 提出了一个基于神经网络的框架来解决类别结构问题

  - 提出了一种新的旋转细化器，它是一种可微层，解决了旋转估计矩阵之间的正交约束和反射模糊问题

  - 该方案在流行的基准数据集上显示了最新的性能

***

* 相关工作

  - NRSFM(Non-rigid structure from motion)

  - Structure from category
