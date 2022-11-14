| 序号 | 论文标题                                                     | 更新时间   |
| ---- | ------------------------------------------------------------ | ---------- |
| 1    | 2022-arXiv-ReLER@ZJU-Alibaba Submission to the Ego4D Natural Language Queries Challenge 2022 | 2022.10.17 |
| 2    | 2022-arXiv-Ego4D: Around the World in 3,000 Hours of Egocentric Video | 2022.11.11 |
| 3    | 2022-arXiv-Structured Video Tokens @ Ego4D PNR Temporal Localization Challenge 2022 | 2022.11.14 |
|      |                                                              |            |

template：

**20xx-where-title**

<font color='vornblue'>核心思想：</font>

xxxx

<font color='vornblue'>代码：</font>xxxx

<font color='vornblue'>相关细节：</font>

<font color='vornblue'>顺便吐个槽：</font>

<font color='vornblue'>启发：</font>

1. **2022-arXiv-ReLER@ZJU-Alibaba Submission to the Ego4D Natural Language Queries Challenge 2022**

	<font color='vornblue'>核心思想：</font>

	浙江大学庄越挺组在Ego4d文本查询帧挑战中得到的第一名方案。该课题给出一个视频clip和一段查询文本，要求定位到视频clip中的一个包含查询问题答案的时序片段（span）。作者提出了一个多尺度的跨模态transformer、一个帧级别的对比学习损失、两种数据增强的方式（变长滑窗采样、视频剪切）。其实还是一个很经典的思路，即transformer+对比学习+扩充训练数据。论文其实还没写完，目前写了4页，估计后续是打算写完投稿CVPR。

	<font color='vornblue'>代码：</font>[Ego4d_NLQ_2022_1st_Place_Solution](https://github.com/NNNNAI/Ego4d_NLQ_2022_1st_Place_Solution)

	<font color='vornblue'>相关细节：</font>

	1. 作者提出，本课题主要面临两类挑战：一是定位要求很精确，Ego4D的视频clip平均时长达到7.5分钟，然而最终输出结果标注的平均时长仅为5秒；二是数据量比较缺乏，虽然Ego4D数据集里有10000个clip-text对，但是其中只有1200个视频片段，不足以支撑训练。
	
	 2. 作者使用了一个T层（默认是3层）的cross attention模块作为自己的backbone（PS：这么说其实也不准确，其实模型是利用了Ego4D官方提供的slow-fast和Omnivore提供的视频特征，文本特征则是用CLIP提取的，所以这里的backbone并不承担最低级的特征提取工作），并基于backbone设计了一个显著度判别器、关键区域判别器、有条件跨度判别器。为了增强特征，作者还利用CLIP来强化特征学习，作者对于视频特征的每一帧，随机选择其一个原始输入帧并输入进CLIP的image encoder（基于Vit-B/16）得到特征，这个视觉特征和视频特征拼接在一起，作为最终的输入特征。而文本特征则通过CLIP的文本编码器得到，不过作者不是CLIP传统的利用\<EOS\>的特征，而是翻转句子得到token级别的信息（？？？这是个甚么意思？等回头看代码再说吧）。
	
	  3. 作者的多尺度技术来源于另一篇论文VSLNet-L，该方案将视频分为$K$个视频片段，每个视频片段$V_k$各自输入跨模态编码器得到特征$F_k$，随后由Nil Prediction Module处理（该模块出自论文Natural language video localization: A revisit in span-based question answering framework）得到分数，该分数的含义是视频片段$V_k$和查询文本对应段的重合置信度。所有特征经过该分数的重新加权后在sequence维度上拼接起来并输入预测头。为了预测目标段，作者利用了VSLNet的条件跨度判别器和Moment DETR的显著度判别器。其中条件跨度判别器使用两层transformer encoder和两层线性层来预测输出span的开始和结束；而关键区域判别器和显著度判别器都是用两个线性层构成的，用于预测哪个视频帧属于输出span。
	
	  4. 损失函数的设计：一个直观的理念是，有关本文的NLQ任务，同一个视频里和文本对应的部分和文本的相似度应该高于其他部分。作者设计的计算视频帧和输入文本的函数为帧的特征$1\times d_v$和文本里每个词的特征$1\times d_t$点乘后求平均，再除以温度常量（默认为0.07）。基于上述两点，作者的对比学习函数则是很常规的NCE损失，限制同一个视频中和文本对应的片段为正样本，其他片段为负样本。除了NCE损失外，作者还使用了span损失、QGH损失、NPM损失、saliency损失，这些损失的设计可以参考Span-based localizing network for natural language video localization; A revisit in span-based question answering framework; Qvhighlights: Detecting moments and highlights in videos via natural language queries。
	
	  5. 数据增强：作者的数据增强方式是变长滑窗采样、视频剪切的结合，下图展示了其区别。前者首先确定一个采样范围$[r_s, r_e],r_s,r_e\in [0,1]$，在范围内随机采样一个数$\hat{r}$，滑窗的大小则为$\hat{r}*l_v$，其中$l_v$是视频的整体长度，利用此方法从视频中采样正样本clip，并确保其包含标注所对应的span。而后者则取另一段视频A并将其剪裁成两段，放至视频B的开头结尾，构成一个新的视频clip，特别地，有一个超参数$P_{vs}$控制同时slice两段视频的概率（这个我其实不是很懂，是二者都裁剪然后穿插拼接吗？作者在论文中阐述得比较模糊）。作者的结合方式如下：首先取视频$V_1,V_2$，然后利用随机滑窗采样得到正样本clip：$V_{1p}$，然后将slicing策略应用在$V_{1p}$和$V_2$上，获得最终的clip。
	
	  6. 作者的方法利用了两种预提取的特征：slowfast和omnivore，效果相差几乎为0。但是二者ensemble起来效果还是有明显提升的。关于ensemble策略就不展开介绍了，因为明显是一个trick。
	
	     <img src="./ego_video_hand_and_objection_assets/1-2.png" style="zoom:40%" />
	
	<font color='vornblue'>顺便吐个槽：</font>
	
	1. 作者的模型其实提升已经很大了，但是性能也还是很低。IoU=0.3的R@1仅10.79%，IoU=0.5的R@1仅6.74%，大胆预测，不训练编码器还是一个非常影响模型性能的操作，后续会继续观望。

<img src="./ego_video_hand_and_objection_assets/1-1.png" style="zoom:50%" />

2. **2022-arXiv-Ego4D: Around the World in 3,000 Hours of Egocentric Video**

​	<font color='vornblue'>相关细节：</font>

1. 根据作者所说，FHO benchmark三个任务的含义分别是确定状态变换的时间(PNR)、空间(scod)、语义(oscc)。

2. 相关工作：（PS：这一段提到了很多以往的论文，需要全部读一下作为参考）

   1. 之前也有object change类数据集，这类数据集分为两种：第一种将状态变化视作是attribute，比如MIT States dataset；第二种将状态变化视作是action。
   2. 之前也有human hand action类数据集，Yale human grasping dataset收集了27.7小时的未标注人类手抓取视频；Something-Something数据集包含220,847段标注了174种手和物品交互动作的短视频；Jester数据集包含148,092段标注了27种手势的短视频；Human Hands数据集包含100k帧视频，主要关注手和物体的交互过程。
   3. 之前也有第一视角视频数据集：Activities of Daily Living数据集标注了10个小时的行动识别数据；UT-Egocentric数据集标注了17小时的video summarization数据；UT Egocentric Engagement标注了14小时的拍摄者加入时刻数据；EGTEA+标注了28小时的44中厨房动作分类视频；EPIC-KITCHENS数据集标注了100小时的89977种厨房交互动作的视频，其中包含97种动词和330种名词，训练任务是目标视频、动作识别、预测交互；Charades-Ego数据集包含34小时的标注了156种动作的第一视角视频，并且有第三视角视频辅助。

3. 数据分析

   1. 据作者所说，在8s的snippets中，PNR帧的分布整体上接近一个高斯分布，即中间部分的PNR帧比较多，但是post frame的边界是个例外，有很大一部分post frame在snippet的最后一帧。同样的，也有很大一部分pre frame落在snippet的第一帧。
   2. 四种目标：左手、右手、改变状态的目标、工具的bbox size分布基本一致
   3. 从动词的分布上来看，put, take两词出现的频率显著高于其他词

4. 模型分析：

   1. PNR/oscc:
      1. I3D ResNet-50(PNR+oscc)常规的backbone+两个不同作用的head框架
      2. BMN(PNR)：将视频起始到PNR的一段视作是一个segment，从而将PNR定位视作是segment detection。
      3. SlowFast + Perceiver(PNR+oscc)：SlowFast用于提取特征，Perceiver用于进行PNR定位和oscc
   2. scod：作者说他们期望后面的工作能够利用pre/pnr/post三帧的信息综合训练，但是在baseline中只用了pnr的信息进行训练。baseline除了Faster-RCNN和DETR外的性能都很差，不展开介绍。

5. 性能对比：

   1. oscc：

      <img src=./ego_video_hand_and_objection_assets/2-1.png width=50% ></img>

   2. PNR:

      <img src=./ego_video_hand_and_objection_assets/2-2.png width=50% ></img>

   3. scod：

      <img src=./ego_video_hand_and_objection_assets/2-3.png width=50% ></img>

6. 讨论：作者也指出，本任务的关键是理解人手的活动和物体状态的关系，并且PNR及其后方的信息也是很重要的，例如“劈木头”这一动作，就需要结合PNR后的若干帧才能精准确认状态的变化。

7. **2022-arXiv-Structured Video Tokens @ Ego4D PNR Temporal Localization Challenge 2022**

   **->2022-arXiv-Bringing Image Scene Structure to Video via Frame-Clip Consistency of Object Tokens**

   <font color='vornblue'>核心思想：</font>

   提出了SViT模型，利用了一小部分图像的结构信息来提升视频模型性能。作者将图像和视频视作两个不同的模态，行文重心放在了利用图像来boost视频这件事上，利用了包括场景图在内的一系列技术。baseline选取了MViT。作者经过修改，目前已经达到了0.515的PNR错误率，是目前的新SOTA。

   <font color='vornblue'>代码：</font>本身的代码[elabd/SViT](https://eladb3.github.io/SViT/)，baseline MViT的代码如下：[facebookresearch/mvit](https://github.com/facebookresearch/mvit)

   <font color='vornblue'>相关细节：</font>

   1. 如上所说，作者的核心idea是如何通过图像来帮助视频理解，所以作者提出最直观的方法是让二者共享一个transformer模型，但是这种idea面临着两个问题：如何建模结构信息和如何处理domain gap。为此，作者提出了两个关键概念：一是目标token（从已学习embedding初始化而来的新token，作者也将其称为object prompts）用于捕捉视频/图像中的目标信息，<font color='red'>利用场景图来作为监督</font>；二是Frame-Clip一致性损失，该损失用于确保目标token在图像/视频中的一致性。
   2. 场景图只在训练时使用，测试时不再使用。从作者给出的示意图来看，场景图里似乎bbox是有角度标注的，可以旋转。但是作者给的文字描述里确实说bbox是$\mathbb{R}^4$，很奇怪。每张图像有4个bbox，分别表示左手、右手、和左手互动的物体、和右手互动的物体。还有一个4维的01变量表示每个bbox是否存在。另一方面，在手和物体之间建立有向边，有向边的值仅为1或0，表示手是否和物体有直接互动。
   3. 常规的video transformer做法是将视频编码为$T\times H\times W\times d$的张量，随后加上spatio-temporal position embedding，经过多层transformer编码后，进行平均池化得到全局特征。但是这么做的话无法同时适用于图像。于是作者提出使用目标token，假设每帧至多包含$n$个目标，则插入$n$个目标token，目标token的构建是$o_i+r_t$，其中前者是object prompt，后者是时序positional embedding，于是总计有$T\times H\times W+T\times n$个token一起输入transformer。
   4. 损失函数总共包含3项：
      1. Video Loss：二项交叉熵损失，意义没有明说，应该是PNR自带的任务损失
      2. HAOG Loss：transformer输出每张图像的$n$个object token，在本文中，$n=4$。每个object token有个作用：预测bbox的坐标、预测bbox是否存在、预测手和对应物体的联系变量，分别用L1损失+GIoU损失、二元交叉熵、交叉熵约束三个目标
      3. Frame-Clip一致性损失：为了避免模型只优化图像，作者额外提出了本损失。该损失将视频用clip形式和$T$帧图像形式输入两次（从我对作者上文的理解，区别仅仅是少了一个temporal positional embedding），然后约束两次输入的object token的$L_1$损失尽可能小。也可以理解为该项损失是在拟合场景图中的边信息。
   5. 作者训练时额外使用了100 Days of Hands数据集，但是作者在所有的消融实验里都没提到只用Ego4D pretrain的结果，感觉实验不是很可信。后来作者的论文经过一次大改版，预训练数据集改为K400，下游任务数据集增加到5个，并且模型主要做的任务也变成动作识别了。不过改版前后PNR效果分别为0.656和0.515，改版前没有classification的数据，改版后为74.1%，方法上感觉也没怎么大改，光是换个数据集能有这么大的提升吗？
   6. 作者提到了一个细节：作者尝试自己分析了一下自己的实验结果，成功的暂且不论，失败的大部分是预测的过早了，例如刚拿出刀靠近胡萝卜就被predict，而不是胡萝卜被切开的时候。

   <font color='vornblue'>启发：</font>

   1. 本文方法其实我不是很喜欢，一方面使用了额外的训练数据，另一方面还用了额外的特征图信息，再者将图像和视频的一致性感觉挺奇怪的，不知其义。不过既然能SOTA，而且是在多个数据集上，那相比方法一定是capture了某个关键痛点。
   1. 上文提到的作者分析自己模型定位错误的帧普遍是预测早了，所以大胆预测其实作者的模型核心是学到了”手靠近物体的帧是PNR“，并没有学到真正物体变化的信息。

   <img src=./ego_video_hand_and_objection_assets/3-1.png ></img>

