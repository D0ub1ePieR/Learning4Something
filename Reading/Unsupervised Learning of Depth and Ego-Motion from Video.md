## Unsupervised Learning of Depth and Ego-Motion from Video
  > **基于视频的深度和帧间运动的无监督学习**

  >&emsp;人类有能力通过即使极短时间序列的景象来判断场景*三维结构*，计算机几何视觉研究无法对真实场景建模主要因为存在：  
  &emsp;&emsp;*非刚体、遮挡和纹理表现的缺失*  

  >&emsp;通过人类平时大量的观察已经形成了对世界充分的*几何信息*的理解，我们可以使用这些知识当我们接收一个新的场景，即使是一个*单视角*的图片

#### 简介

  + 成果  --  用于估计**单目深度**`monocular depth`和**摄像机位姿估计**`camera motion estimation`(6-DoF)

  + 结构  --  端到端end-to-end, 单视角深度网络, 多视角位姿网络, 利用计算得到的深度和位姿在目标视角下的图像变形计算损失，利用未标记视频序列训练

  <center><img src='./imgs/unsupervised-cvpr17-introduction.png'/></center>

  + 现有系统问题  --  一个几何视角合成系统只有在对于场景的几何和相机姿态的中间预测相对于*真实的物理场景*相匹配时才能表现得好，尽管不完美的几何或姿态估计可以对某些场景(如无纹理)用*似乎合理的合成视图*进行欺骗，但是如果呈现出另外一组布局和外观结构更加多样的场景，相同的模型将还原失败。因此**网络需要学习深度和相机姿态估计**的中间任务。

  + 数据集  --  KITTI

#### 相关工作  **[TODO]**

  + structure from motion*从运动中恢复结构*  
    需要依赖图像间准确的关联关系，会在低纹理、复杂的几何/光照、薄结构和遮挡等区域造成问题

  + warping-based biew synthesis*基于图像变形的视角合成*  
    通过新的照相机的观察视角合成场景,估计3D底层信息/在输入视图间建立对应关系,随后将输入视图的图像块合成新的视图

  + learning single-view 3D from registered 2D views从配准的2D视图中学习单视图3D  
    只是从世界的图像观测中学习，而不需要显式深度表示的训练方法

  + unsupervised/self-supervised learning from video*通过视频的无监督和自监督学习*  

#### 本文方法

  + **联合训练**,但是得到的位姿估计模型和深度模型可以*单独使用*  

    训练使用短视频序列由移动相机拍摄,**期望的场景是刚体**,运动由相机移动控制

  + 监督来自于**新视角合成任务**,训练集图像序列 $<I_1,\ldots,I_n>$ ,target view $I_t$,source views $I_s$  
    $$L_{v,s}=\displaystyle \sum_s \sum_p |I_t(p)-\hat I_s(p)|$$
    > photomatric reconstruction loss 光度重建误差  

    **其中p为图像像素坐标的索引, $\hat I_s$为源视角Is基于深度图像的渲染模块扭曲到目标帧**  

    **输入为：$I_s$, 深度预测 $\hat D_t$, 相机移动矩阵 $\hat T_{t\rightarrow s}^{R_{4*4}}$**

    <center><img src="./imgs/unsupervised-cvpr17-approach1.png"/></center>

    将目标帧$I_t$输入深度预测网络得到预测深度图 $\hat D_t(p)$，取临近视角 *$I_{t-1}及I_{t+1}$*，将它们以及目标帧输入至姿态估计网络得到两个相机位置变换 *$\hat T_{t\rightarrow t-1}及\hat T_{t\rightarrow t+1}$*。随后使用两个网络的输出将源视角扭曲至目标视角。

  + 可微深度图像渲染

    用于上一块的视角合成，由$I_s、\hat D_t、\hat T_{t\rightarrow s}$ 生成 $I_t$  
    $p_t$表示目标视角每一个像素的齐次坐标、K表示相机的内参矩阵，则$p_t$在$p_s$视角下的投影坐标为
    $$p_s\text{~} K\hat T_{t\rightarrow s}\hat D_t(p_t)K^{-1}p_t$$
    > <small><center>将pt转换到像素坐标系，加上深度预测乘上转移矩阵得到目标视角下像素在源视角中像素坐标系的位置，最后乘上相机内参矩阵得到源视角像素所在的相机坐标</center></small>

    随后使用可微双线性采样的空间变换网络得到由
    $$I_t(p_t)\rightarrow I_s(p_s)\rightarrow \hat I_s(p_t)$$
    i.e. 双线性插值`bilinear interpolation`
    $$\hat I_s(p_t)=I_s(p_s)=\Sigma_{i\in\text{{t,b}},j\in\text{{l,r}}}w^{ij}I_s(p^{ij}_s)$$

    <center><img src="./imgs/unsupervised-cvpr17-approach2.png"/></center>

  + 模型限制
    * 背景是静态的不包含运动物体的

    * 目标和源视角间没有遮挡和非遮挡情况

    * 表面是lambertian，使得光一致性误差具有一定的意义  

    为了提升鲁棒性，与深度和位姿网络一同训练一个可解释性的预测网络，其输出一个逐像素的soft mask $\hat E_s$对每一个目target-source对，展示对每个像素合成成功的概率
