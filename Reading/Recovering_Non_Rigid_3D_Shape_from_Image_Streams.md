## Recovering Non-Rigid 3D Shape from Image Streams
***
#### 概述  
  本文研究了从图像序列中恢复三维非刚性形状模型的问题。通过2D图像序列恢复3D形状，SFM技术通常假定三维物体是刚体。提出了一个独特的方法基于非刚体模型，每一帧的三维形状是一组基础形状的线性组合。在这个模型下，跟踪矩阵拥有更高的秩，可以被分解为一个三步的过程生成姿势、配置和形状。能够通过单个摄像机的图像序列生成三维非刚体模型。

#### 简介  
  之前的工作都是将从二维图像序列恢复三维形状和发现非刚体形状的形变参数化表示分开得看做两个问题。常用方法使用PCA。

  我们展示了如何在缩放正射投影下恢复三维非刚性形状模型。每一帧的三维形状都是由K个基础形状线性组合而成。二维跟踪矩阵的秩由3变为3K，可以被分解为三维形状、物体配置和三维基形状，通过SVD。

#### 相关工作  
  最有影响力的一个方法C. Tomasi and T. Kanade. Shape and motion from image streams under orthography: a factorization method. Int. J. of Computer Vision, 9(2):137–154, 1992.展示了对刚体和正射投影的分解方法。J. Costeira and T. Kanade. A multi-body factorization method for motion analysis多体分解方法放宽了刚体约束。这个方法中允许有K个独立运动的对象，结果得到一个秩为3K的跟踪矩阵和一个识别每个对象对应的子矩阵的排列算法。B. Bascle and A. Blake. Separability of pose and expression in facial tracking and animation在跟踪过程中分解面部表情和姿态，但基形状的寻找不在算法中。

  ...很多方法对于非刚体的二维估计使用了PCA。

  最令人印象深刻的是Volker Blanz and Thomas Vetter. A morphable model for the synthesis of 3d faces。建立了高分辨率三维模型的大型人脸数据库。使用手动初始化和形状、纹理和灯光的迭代匹配，可以从一张图像中恢复一个非常详细的3D人脸形状。

  所有存在的非刚体三维模型都需要一个先验或多视角。本文提出的方法不需要先验模型，仅需要单视角。

#### 分解算法
