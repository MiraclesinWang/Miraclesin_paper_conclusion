| 序号 | 论文标题                                                     | 更新时间   |
| ---- | ------------------------------------------------------------ | ---------- |
| 1    | 2021-ICCV-ViViT: A Video Vision Transformer                  | 2022.10.04 |
| 2    | 2022-arXiv-Ego4D: Around the World in 3,000 Hours of Egocentric Video@Hands and Objects Benchmark | 2022.11.09 |

template：

**20xx-where-title**

<font color='vornblue'>核心思想：</font>

xxxx

<font color='vornblue'>代码：</font>xxxx

<font color='vornblue'>相关细节：</font>

<font color='vornblue'>顺便吐个槽：</font>

<font color='vornblue'>启发：</font>

0. 数据集介绍：
   1. *Kinetics*：分为Kinetics400和Kinetics600，400和600分别表示类别数，前者大概有267,000段视频，后者大概有446,000段视频。视频均为25fps的10秒视频。需要注意的是这个数据集的视频都是存在YouTube上的，所以和flickr一样，会随着时间而资源慢慢失效。
   2. *Epic Kitchens-100*：在厨房中拍摄的第一人称视角视频，包含共计100小时的90,000个clip。该数据集的数据表示都是一个动词+一个名词。
   3. *Moments in Time*：包含800,000段3秒的YouTube clip。
   4. *Something-Something v2*：包含220,000段视频，每段持续时间为2-6秒。有别于其他数据集的是本数据集一般物体和背景保持不变，行动发生变化，所以更注重模型的动作细节甄别能力。

1. **2021-ICCV-ViViT: A Video Vision Transformer**

   <font color='vornblue'>核心思想：</font>

   xxxx

   <font color='vornblue'>代码：</font>[Google-research/scenic](https://github.com/google-research/scenic)

   <font color='vornblue'>相关细节：</font>

   1. ViT曾经提到了将transformer用于图像时需要采用大数据集训练才有明显的增益。ViT的作者将此解释为transformer对于卷积缺少一些归纳不变形（所谓归纳不变形，可以参考机器翻译的过程），因而需要更多的数据进行训练或者更强的正则化。
   
   1. 本文使用的MLP包含两层，通过GELU连接。（GELU这个激活函数相比于RELU还是有一定的理论优越性的，以后在transformer相关的结构列可以试试）
   
   1. 作者对video的patch embedding提出了两种方案：uniform frame sampling和tubelet embedding的概念。前者是把视频视作若干张图像（不一定每一帧都取），每张图像按照ViT的方式划分patch；后者其实就是把ViT中的patch增加了一个时间维度。不过加其实也有很多种方式，作者在实验部分对不同的时间维度添加方法进行了比较。
   
   1. 作者提出，视频的transformer可以有多种结构，具体来说如下：
   
      1. *token全排列*：和ViT一样将视频的所有token embedding和cls embedding一起输入模型。这种模式在transformer的时间复杂度为$O(n^2)$，对于视频来说，处理速度很难接受。
   
      1. *层级transformer*：具体来说就是两个transformer各司其职，第一个transformer先将所有视频帧按照图像编码，随后得到的所有帧的全局特征输入或所有局部特征的平均池化第二个transformer（两个transformer都是有positional/temporal embedding的）
   
         ![img](./video_base_assets/2.png)
   
      1. *层级attention*：结构和模型1大体上类似，但是每次attention操作不是所有token之间做，而是分为两级：空间的attention和时间的attention。复杂度和模型2一致。
   
         ![img](./video_base_assets/3.png)
   
      1. *层级点积注意力*：和思路2,3类似，不过这次是在attention的计算公式上入手，同一个特征有两种映射方式，分别是$K_s,V_s\in\mathbb{R}^{n_h\cdot n_w\times d}$和$K_t, V_t\in \mathbb{R}^{n_t\times d}$，对于一半的头，采用前一种计算方式，即$Y_s=Attention(Q, K_s, V_s)$，对于剩下的头，则采取后一种计算方式$Y_t=Attention(Q, K_t, V_t)$。最终将两种输出拼接起来，即$Y=Concat(Y_s, Y_t)\cdot W_o$。
   
      1. 从实验结果上来看，前两种模型效果整体较好，后两种模型则相对较差。但是参数量则是1、4模型较少，2、3相对较多，而处理宿舍则是1>3>2>4。
   
   1. 作者想要像ViT一样利用大量数据训练自己的模型，但是视频数据集目前还达不到那么大的规模，于是作者想要利用ViT的参数来初始化自己的模型。每个模型的初始化策略有所区别：
   
      1. positional embedding：考虑到视频模型除了空间关系外还有时序关系，不同帧的positional embedding未必需要相同，作者将每个帧的positional embedding先都加载为ViT的参数，后续各自训练。
      1. 三维卷积embedding权重：tubelet embedding需要对三维而不是二维的像素进行embedding，一个比较直观的解决方法是将patch内所有帧二维映射后求平均。作者还提供了一种替代思路："central frame initialisation"，即初始化只取一个卷积核$\frac{t}{2}$的帧的二维embedding，剩下的全部置零，用公式表示即为$E=[0,...,E_{central},...,0]$，初始时和上文的"Uniform frame sampling"类似，在训练过程中让模型自己调整$E$。（有个问题，既然你其他地方都置零了，那神经元后面训练不就无法激活了？何谈调整？）（不过从实验结果上来看，这种方式还是最好的，在Kinetics上达到了79.2的Rank-1准确率）
   
   1. 预测时模型输入是采样步长为2的32帧的clip，对于长视频，还会处理其不同的view（我的理解是取不同的clip）并将结果取均值作为最终结果。
   
   1. 作者使用了几种正则化方式，分别是：Kinetics 400初始化、随机深度（这个是啥？引用的论文Deep networks with stochastic depth）、随机数据增强、标签平滑（Rethinking the inception architecture for computer vision）、Mixup（Mixup: Beyond empirical risk minimization）。
   
   1. 作者还消融了tubelet的尺寸对准确率和运行速度的影响，在空间尺寸固定为$16\times 16$的情况下，时间尺寸从2提升到8，准确率和运行速度都在下降。这说明更小的tubelet效果更好。
   
   1. 如果增加输入模型的视频帧数，整体效果都会上升。但是增加在视频中采集的clip数，效果则会先增后降。而提升分辨率则会给准确率带来细微的提升，同时大幅降低处理速度。
   
   ![img](./video_base_assets/1.png)2. **2022-arXiv-Ego4D: Around the World in 3,000 Hours of Egocentric Video**
   
   <font color='vornblue'>核心思想：</font>
   
   xxxx
   
   <font color='vornblue'>代码：</font>xxxx
   
   <font color='vornblue'>相关细节：</font>
   
   1. 根据作者所说，FHO benchmark三个任务的含义分别是确定状态变换的时间(PNR)、空间(scod)、语义(oscc)。
   
   2. 相关工作：（PS：这一段提到了很多以往的论文，需要全部读一下作为参考）
      1. 之前也有object change类数据集，这类数据集分为两种：第一种将状态变化视作是attribute，比如MIT States dataset；第二种将状态变化视作是action。
      2. 之前也有human hand action类数据集，Yale human grasping dataset收集了27.7小时的未标注人类手抓取视频；Something-Something数据集包含220,847段标注了174种手和物品交互动作的短视频；Jester数据集包含148,092段标注了27种手势的短视频；Human Hands数据集包含100k帧视频，主要关注手和物体的交互过程。
      3. 之前也有第一视角视频数据集：Activities of Daily Living数据集标注了10个小时的行动识别数据；UT-Egocentric数据集标注了17小时的video summarization数据；UT Egocentric Engagement标注了14小时的拍摄者加入时刻数据；EGTEA+标注了28小时的44中厨房动作分类视频；EPIC-KITCHENS数据集标注了100小时的89977种厨房交互动作的视频，其中包含97种动词和330种名词，训练任务是目标视频、动作识别、预测交互；Charades-Ego数据集包含34小时的标注了156种动作的第一视角视频，并且有第三视角视频辅助。
   
   3. 数据分析
      1. 据作者所说，在8s的snippets中，PNR帧的分布整体上接近一个高斯分布，即中间部分的PNR帧比较多，但是post frame的边界是个例外，有很大一部分post frame在snippet的最后一帧。同样的，也有很大一部分pre frame落在snippet的第一帧。
      1. 四种目标：左手、右手、改变状态的目标、工具的bbox size分布基本一致
      1. 从动词的分布上来看，put, take两词出现的频率显著高于其他词
   
   4. 模型分析：
   
      1. PNR/oscc:
         1. I3D ResNet-50(PNR+oscc)常规的backbone+两个不同作用的head框架
         2. BMN(PNR)：将视频起始到PNR的一段视作是一个segment，从而将PNR定位视作是segment detection。
         3. SlowFast + Perceiver(PNR+oscc)：SlowFast用于提取特征，Perceiver用于进行PNR定位和oscc
   
      2. scod：作者说他们期望后面的工作能够利用pre/pnr/post三帧的信息综合训练，但是在baseline中只用了pnr的信息进行训练。baseline除了Faster-RCNN和DETR外的性能都很差，不展开介绍。
   
   5. 性能对比：
   
      1. oscc：
   
         <img src=./video_base_assets./2-1.png width=50% ></img>
   
      2. PNR:
   
         <img src=./video_base_assets./2-2.png width=50% ></img>
   
      3. scod：
   
         <img src=./video_base_assets./2-3.png width=50% ></img>
   
   
   6. 讨论：作者也指出，本任务的关键是理解人手的活动和物体状态的关系，并且PNR及其后方的信息也是很重要的，例如“劈木头”这一动作，就需要结合PNR后的若干帧才能精准确认状态的变化。