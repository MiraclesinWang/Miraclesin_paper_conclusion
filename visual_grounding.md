前排提示：该课题需要一定的目标检测基础，请至少先学会faster-rcnn的基本架构，否则会造成诸多困难

|      |                                                              |        |
| :--- | :----------------------------------------------------------- | :----- |
| 1    | 2021-AAAI-Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding | 王海光 |
| 2    | 2020-TPAMI-Discriminative Triad Matching and Reconstruction for Weakly Referring Expression Grounding | 王海光 |
| 3    | 2021-arXiv-Visual Grounding with Transformers                | 王海光 |
| 4    | 2020-ECCV-Contrastive learning for weakly supervised phrase grounding | 王海光 |
| 5    | 2021-arXiv-OVIS: Open-Vocabulary Visual Instance Search via Visual-Semantic Aligned Representation Learning | 王海光 |
| 6    | 2020-CVPR-Graph-Structured Referring Expression Reasoning in The Wild | 王海光 |
| 7    | 2021-arXiv-CPT: COLORFUL PROMPT TUNING FOR PRE-TRAINED VISION-LANGUAGE MODELS | 王海光 |
|      |                                                              |        |

前人整理的论文集：[![img](https://github.com/fluidicon.png)GitHub - TheShadow29/awesome-grounding: awesome grounding: A curated list of research papers in visual grounding](https://github.com/TheShadow29/awesome-grounding#paper-roadmap-chronological-order) 

\1. ***2021-AAAI-Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding***

王海光：

核心思想：grounding可以分为单阶段和双阶段两种（这个后续有空调研一下）。双阶段主要分为两个阶段：1. faster-rcnn等网络检测出bounding box。2. 将bounding boxes和文本进行匹配（为了便于大家理解，这里放个图）。作者认为，这个过程中，第一阶段的检测是完全由图像自身决定的，文本没有参与筛选，需要有文本来导向。所以作者在提取bounding box的NMS阶段，引入了bounding box和textual expression的相似度来做NMS。

![img](blob:https://deeplearning-suda.atlassian.net/2ae903f3-7dc8-4a2f-9a5b-bf2d6aff14e3#media-blob-url=true&id=511b0757-15b3-4260-a1fc-e0dfa5eff745&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=Z7%25C%7B%5D%24PIU1%407R%5BIJFZMKSA-20211002-140937.png&size=35481&height=212&width=641&alt=)

上面是传统的方法，下面的是作者的方法

代码：[GitHub - ChopinSharp/ref-nms: Official codebase for "Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding"](https://github.com/ChopinSharp/ref-nms)

一点细节：

1. 普通的two-stage grounding对于RPN的proposals一般根据检测的confidence取前k个，作者这边是根据阈值取confidence大于阈值的框，然后将每一个bbox和输入的一句Query中的每个词做attention。得到一个将词embeddding特征聚合得到的综合特征向量，再和bbox的特征进行相似度计算。

顺便吐个槽：

1. 仔细一想，作者的方法只是把他所抨击的传统two-stage方法的利用confidence取前k个改成取confidence阈值大于阈值的，然后把原来的匹配阶段换到NMS之前并将匹配相似度作为NMS的参考而已。换而言之，其实就是相当于其他方法做了NMS对框的个数做了删减，但是这边没做（也不能说没做，做的少，所以留的框多，自然把包含目标的框误筛掉的概率就低），直接做grouding，最后对grounding的结果补一次NMS（其实等于没补，反正单个文本的grounding只能取一个bounding box，所以最后总归是要取和文本匹配时相似度最大的bounding box的，且后面还有一次匹配，NMS没做也一样）。所以，其实按照我的理解，作者这篇文章效果好仅仅靠的是他多留了很多的bounding box，所以把其他方法误筛掉的框又捡回来了。（希望我理解的是错的，不然AAAI在我心里的地位-1）
2. 即便上述操作真的能到SOTA，按照我对文章的理解，它的复杂度也是不可接受的。每个bbox对query中每个词做attention，那这个复杂度是O(mn)，m表示稍微筛选后的bbox个数，n表示句子中的单词个数。而且bbox没有预筛选（或者说筛选后的量远远高于一般方法），整体复杂度肯定高得离谱（文章没提到时间的问题，也没说自己用了多少卡）
3. 虽然我有时也吐槽别人的写作，但是这篇论文是真的刷新我了的下限，各种意义不明的符号飞来飞去，而且有些操作明明毫无意义（比如对最终score进行归一化），但是作者还是给它加上了（当然，不排除这是业内规范的可能性，最后概率必须是0-1这样）

启发：

1. 作者本人宣称，各类课题“苦秦久矣”，并且他提出的这个只是一个即插即用的小模块（改进NMS），所以他下一步打算在视频groudning、VQA、场景图生成等方面尝试自己的方法。感兴趣的可以去截胡。
2. 这个idea其实和我现在（2021.10）在做的有点类似，也就是文本主导的图像局部搜索+在提取局部阶段优化，幸好他的方法做得比较烂，还有改进余地，他的方法对我重新构思和改进完善自己的idea还有提供了一些方向性的指导的。而且作者说自己是第一个优化局部提取的，如果他说的没错，那idea还是可以做下去的。

![img](blob:https://deeplearning-suda.atlassian.net/1153893e-eb8e-426b-a764-455a1fd7d4bb#media-blob-url=true&id=9b01f2ee-ceed-4832-8e90-80082df71a99&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=K%40K%7DZ%2522WXEKP71_KH%5D_P44-20211002-142748.png&size=77422&height=252&width=674&alt=)

\2. ***2020-TPAMI-Discriminative Triad Matching and Reconstruction for Weakly Referring Expression Grounding***

核心思想：方向是弱监督的grounding，实现方式是从输入的句子中提取若干个三元组<主体对象，关系，修饰对象>，例如<man, holding, plate>，主要过程分为：三元组生成->bbox特征+位置特征提取->匹配相似度计算->三元组重建（据说重建是该领域的传统方法）

代码：[GitHub - insomnia94/DTWREG: Preliminary code for reviewers](https://github.com/insomnia94/DTWREG) 

一点细节：

1. 关于三元组的提取，使用的stanford之前开源的Stanford CoreNLP分析语法树
2. 匹配过程主要是将所有的bbox proposal构建为对，然后寻找和三元组最匹配的bbox对。具体来说是构建了3个transformer，分别用于主体对象、关系、修饰对象的计算。（就是将图像特征和文本的embedding拼接起来，然后自注意力）这一步得到三个相似度矩阵。
3. 重建有两种：1)软重建：根据上述矩阵的权重对bbox特征进行加权，得到聚合特征；2)硬重建：利用Gumbel trick（关于利用Gumbel trick实现hard attention的方法我在报告*2020-ACMMM-Hierarchical Gumbel Attention Network for Text-based Person Search*时也曾提过）。这两种得到的都是根据特征解码得到的伪文本三元组。
4. 损失函数：和GAN比较类似，是计算重建三元组和原三元组特征的L2范数。

启发：

1. 这篇论文的意思说的还是很明白的，就是对之前普通的文本embedding做出改进，换成三元组，从而规范化地表达出文本中的关系（person search也曾有类似的工作，但是它还是提取了短语，没有真正做成三元组：*2020-ECCV-Improving deep visual representation for person re-identification by global and local image-language association*）相当于为grounding的关系表达提供了思路吧
2. 以前也text-image retrieval有篇论文以这种成对的方式表达图像局部的关系（*2020-TIP-MAVA: Multi-Level Adaptive Visual-Textual Alignment by Cross-Media Bi-Attention Mechanism*），但是这些论文最后对局部关系的表达都只剩下四维度的坐标向量和视觉特征的各种操作了，那么问题来了，比如像下图这种，“拿着球棒的男生”，”男生”的bbox和“球棒”的bbox其实关联不仅仅是坐标相近或者图像语义有任何联系。那么对图像中目标间关系的建模究竟应该是什么样的呢？是不是可以用attention机制去寻找”a boy holding”的热度图，再跟球棒的热度图结合呢？

![img](blob:https://deeplearning-suda.atlassian.net/c6cd7eca-70db-47ef-a892-b477e9ef5881#media-blob-url=true&id=ad53f153-053f-4cbd-8146-08919cef30c6&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=P66%251%5B3Z01%5B%25E(Q%5B%60%5B%40DBLP-20211003-103907.png&size=271527&height=338&width=945&alt=)

\3. ***2021-arXiv-Visual Grounding with Transformers***

核心思想：标题简约而不失格调，轻松点明了论文的主旨，论文目前还在arXiv上（截至2021.10.9），不清楚后续要投在哪（感觉像是CVPR）。具体来说，本文的贡献就是首次实现了完全由文本主导的图像目标检测，并且这个过程没有使用任何预训练模型。

代码：无

具体细节：

1. 图像流首先用CNN得到feature map，文本流则首先用RNN得到序列化特征。然后将这两个特征一起传进text-oriented的transformer（真的是transformer，结构图完全仿*attention is all you need*），经过解码得到一个四维向量——bbox的坐标。

启发：

1. emmm，怎么说呢？之前受本页面第一篇论文启发，我是打算做一个这种不采用bbox，从提取特征阶段就不计算proposal，直接利用文本信息在图像全局上搜索来回归出bbox坐标的工作，代码都开始敲了，但是后来就调研到了这篇，虽说还是在arXiv上的，但是还是只能感慨手慢了（笑哭）。

![img](blob:https://deeplearning-suda.atlassian.net/ed299c24-a492-4c42-9225-22a82a0d153e#media-blob-url=true&id=bb0bc98e-2d80-4cc8-a6ef-e4c6f7255eeb&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=_2(LRGI%7DJDXQDP_U%40L~%609XF-20211009-091750.png&size=96808&height=322&width=800&alt=)

![img](blob:https://deeplearning-suda.atlassian.net/dcd876d5-b1dd-4b02-8fde-baf97c490f30#media-blob-url=true&id=8598dd15-231b-453d-9160-985645982204&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=BY0%24_D%7D%7BWI2OE1NOD%60KQE8H-20211009-094714.png&size=34373&height=356&width=392&alt=)

\4. ***2020-ECCV-Contrastive learning for weakly supervised phrase grounding***

核心思想：主要应对的问题是弱监督的grounding。如标题所说，主要的方法是对比学习。这里对比学习生成正负样本的方法我觉得还是挺有意思的。

代码：[GitHub - BigRedT/info-ground: Learning phrase grounding from captioned images through InfoNCE bound on mutual information](https://github.com/BigRedT/info-ground) 

一些细节：

1. 作者整理的related work还不错，具体总结了以下几点：
   1. 现有信息聚合手段主要分为：池化、noisy OR（这个感兴趣的同学可以查一查，其实就是简化的概率图）、attention。
   2. 常用损失主要分为：binary loss、文本重建损失、triplet loss。
2. 关于InfoNCE：这个在对比学习中可能提及的比较多，出自*Representation Learning with Contrastive Predictive Coding*这篇论文，论文原文里提到了InfoNCE这个loss的下界

![img](blob:https://deeplearning-suda.atlassian.net/9b375f9c-6739-4a29-947a-f4c35d70662a#media-blob-url=true&id=4d8c93d7-b635-4877-8109-6645fded3d0b&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=50UI(D78BFVA1K8DPPIMLI8-20211011-121527.png&size=6032&height=42&width=254&alt=)

​     其实如果你关注对比学习，那这个损失函数你应该不陌生。所以这里并非什么创新点

\3. 关于对比学习负对的生成：文本：首先寻找句子中的名词，把它mask掉，然后用bert根据上下文c来预测这个被Mask的词是什么，取前30个候选。但是，考虑到这30个词中会有本来的词和它的同义词，这类词不能构成负对，所以再把原词x和上下文c拼接起来传入bert，计算和候选词的相似度，来进行reranking。最终获得25个候选词作为负对。图像：相比之下，图像的负对就显得很普通了，就是单纯换张图像而已。

\4. 这篇论文的涨点极其恐怖，在所有指标上都相对于前SOTA涨了5%以上的准确率。但是这不是关键，关键是消融实验相当有意思。从作者的消融实验来看，不引入对比学习和引入对比学习，但是使用随机负对相比，准确率几乎相同。真正带来巨变的是基于bert生成负对的模块，R@1和accuracy都因此涨了8%以上。而重排名则对准确率影响较小。那么这个现象有意思在哪呢？其实很简单：一个良好的负对选择算法对对比学习的性能有着巨大的影响。我知道这个结论可能听着像一句废话，但是私以为这就是一个很重要的方向，怎么样利用更高效的训练数据对模型产生四两拨千斤的效果？

启发：

1. 作者原文提到了这么一句话：“在计算文本和图像的互信息前，需要把这两个模态的共有信息提取出来”，私以为对于所有损失函数都需要注意这一点。也就是其他跨模态领域之前提到的一个观点：去除噪声。当然，本文中这是通过互信息的损失函数来实现的，为了最大化互信息，模型会关注和文本相关度较大的图像局部。
2. 说实话，这篇论文工作novelty感觉不咋地。核心贡献就是一个对比学习，其他的都是一些常用实现方法。如果对比学习以前有别的论文做过了，那这篇论文novelty其实就显得比较不足了。但是得承认这篇论文还是有很高的价值的。首先，他提供了我在“一些细节”第4点当中提到的一个启发；其次，这篇论文的写作还是很棒的，图标很清晰，而且也能够从各种角度突出工作的闪光点与价值，比如MI部分大量的公式，补足novelty的不足。

![img](blob:https://deeplearning-suda.atlassian.net/8e096663-fb3c-4179-98de-363b605c8470#media-blob-url=true&id=440570f1-b4dd-4e2c-a4ed-39da9d741ee9&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=image-20211011-082100.png&size=848905&height=889&width=2254&alt=)

\5. ***2021-arXiv-OVIS: Open-Vocabulary Visual Instance Search via Visual-Semantic Aligned Representation Learning***

核心思想：这和我之前组会汇报过的*Open Vocabulary Object Detection Using Captions*是一个课题。这篇论文的训练和测试模型结构相差比较大，后面会展开分析。

代码：无。

相关细节：

1. 对于缺词的处理：论文在分词的时候处理得更细一些，把词缀也分离出来了。比如reporter会被分成”report”和”er”。据作者所说，这种处理方式可以保证数据集所有的词都在词典中出现过（但是私以为这没办法处理拼错的词和特殊用法，例如talllllllll或face2face）
2. visual-semantic encoder：作者这个地方写得不太好，其实它就是第一个图的transformer模块，作者画图的时候测试模型图展开了这个模块，但是训练图没有展开，只说了”visual-semantic encoder”，私以为是个大败笔。如果能分成三张图我个人认为比较合适。除此之外，还有就是关于W，私以为训练模型图中的W是错的，应该用W^T，这应该算是一个写作失误。
3. 训练策略：这篇论文的训练策略比较特殊，使用的数据集没有grounding的标注，而是image caption的数据集，所以作者也没办法直接实现训练。为了解决这个问题，作者随机mask掉句子中的一些单词，然后根据图像和剩下的词的特征来预测这些单词。
4. 训练与测试：总结来说，训练所给测试提供的，包括visual backbone、visual-semantic encoder(transformer)、W(textual embedding)。
5. 这个方法在实验部分里visualize了一个很有意思的东西——对于多义词的定位。比如bass既可以指鲈鱼，又可以指低音吉他。作者给出了一个visualization成功把这两种含义的bass都给定位到了。

顺便吐个槽：

1. 这篇论文的SOTA对比实验简直是个迷，只有一个被对比SOTA方法（好像还是作者自己魔改的别人的方法），直接导致作者的模型性能完全吊打，很多mAP都能达到另一个方法的10倍。

启发：

1. 这个缺词的处理我觉得还蛮好玩的，如果能进一步分析，把所有词模块化，比如”versatile”分成”ver””sat””ile”，搞不好是文本embedding的新方向。
2. 其他就没啥了，都是一些被玩烂的方法，作者也就是赶上了该课题的首创红利，没啥现有方法。

| ![img](blob:https://deeplearning-suda.atlassian.net/3c2d720d-5e85-4798-83b9-4e22d38c85df#media-blob-url=true&id=dfd0647d-90c1-47f4-8904-627ebb5c12a1&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=image-20211016-120955.png&size=51872&height=326&width=377&alt=)测试模型图 | ![img](blob:https://deeplearning-suda.atlassian.net/d40f708f-e405-424b-b87d-ab177b883275#media-blob-url=true&id=80d641cf-f301-4d1f-906b-32d2b9530f73&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=4S%24S~%60%5BUV9XXYS66_BH%7BTA1-20211016-121026.png&size=49472&height=350&width=348&alt=)训练模型图 |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
|                                                              |                                                              |

\6. ***2020-CVPR-Graph-Structured Referring Expression Reasoning in The Wild***

核心思想：这个工作对文本和图像都建了个图，把两边的数据都结构化，最后做成一个图上点的匹配问题。对于图像建立场景图，图的节点表示各个局部对象，图的有向边表示关系（和知识图谱类似）；文本则建立语法图，节点表示名词短语，边表示介词。

代码：[GitHub - sibeiyang/sgmn: Graph-Structured Referring Expressions Reasoning in The Wild, In CVPR 2020, Oral.](https://github.com/sibeiyang/sgmn) 

具体细节：

1. 关于图像建图：建图的方式不使用任何成熟的场景图网络，而是用CNN提取的特征+坐标表示图像里每个目标（图上每个节点）。目标之间的关系用坐标变换和另一个节点的视觉特征的拼接来表示。
2. 关于文本建图：用的是一个现成的方法，与*Generating semantically precise scene graphs from textual descriptions for improved image retrieval*这篇论文用的相同。
3. 比较好玩的是这篇论文处理图的方式。对于文本图，首先选择图上出度为0的节点，即对其他节点没有影响的节点，然后从这些点开始执行广度优先搜索（这个过程图上边的方向会反过来，所以不用担心节点入度为0的事），并把访问到的节点放入栈中。然后从栈中不断弹出节点处理，从而实现更新任何节点的时候，有边指向该节点的节点（即对该节点有影响的节点）已经被更新过了（其实这里搞个拓扑排序就完事了，作者整得太复杂了）。

启发：

1. 关于具体细节第3点里提到的作者处理图的方式。这个地方虽然作者整得过于复杂，但是我觉得这个可能对以后图神经网络的更新方式有一定的启发，可以先留着坑。
2. 这篇论文总的来说挺奇怪的，很多地方让人想不通为什么要这样（也可能是我实力不够）。总之先留个坑吧，回头再来研究下，看看会不会有什么新发现。

![img](blob:https://deeplearning-suda.atlassian.net/0a9f495e-6145-4aeb-a4cf-0941ceaca883#media-blob-url=true&id=08bddce7-c7d7-43ff-87a3-ee5a51ffeee4&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=_4W%5DZY(7%7DNY7824S2FK1T4U-20211021-141806.png&size=257719&height=335&width=1097&alt=)

\7. ***2021-arXiv-CPT: COLORFUL PROMPT TUNING FOR PRE-TRAINED VISION-LANGUAGE MODELS***

核心思想：prompt在跨模态任务上的首次尝试，不得不说刘知远手下大佬的手速是真的快。这篇论文的核心思想很简单，单纯是把prompt用于Visual Grounding任务上，实现方法是给图像中不同局部上色，然后prompt的问题是：“[query]描述的局部在图像中是什么颜色的呢？”上色个人认为是一个很好的prompt。而且这篇论文虽然是赶热度做的，但是实验做得非常solid，实验结果列了整整半页的表格，塞得满满的，而且每个数据都是x加减y的形式，也就是都会给出误差。工作量很庞大。不过可能也是有点过于着急了，实验效果还没到SOTA（也可能是现在的SOTA过于难对付）

代码：无

相关细节：

1. 作者的这个框架很特殊，既可以做有监督学习，又可以做零样本学习。零样本学习其实就是对预训练模型的现有性能足够自信，所以不做微调了，直接上。
2. 作者的上色也不是随便上的，作者先准备很多纯色图像，然后做了另一个prompt learning，范式是“这张纯色图像是[MASK]色的”，然后预训练模型会返回一些文本单词及其得分。纵向来看，这个得分越高，说明模型越认为这张纯色图像应该用“x色”这个词来描述而不是别的颜色（比如会说这个网页是“白色”的，得分就比说这个网页是”黑色”的高）；横向来看，这个得分越高，越说明模型对该种颜色敏感，比如给出一张黄色图像，模型有0.9的置信度认为是“黄色”，而给出一张绿色图像，模型只有0.6的置信度认为是“绿色”，那就可以认为模型对黄色比对绿色更敏感，即对“黄色”建立的跨模态联系更加紧密。
3. 还有个微小的细节。如果你只看下面的模型图，你会觉得作者是只对bbox上色的，但其实作者提了两种方法，一种是对bbox上色，另一种是对segmentation扣出来的目标上色（关于segmentation，其实就是parsing，大家可以去搜下，不多赘述）。并且实验结果普遍是后者的准确率更高（但是复杂度也相对更大）。

启发：

1. 个人认为短期内，给图像局部上色然后文本询问颜色，仍将是跨模态prompt很重要的实现方式。（事实上，这也是目前跨模态prompt的唯一实现方式）有兴趣的可以把各大跨模态模型用在各类任务上。

![img](blob:https://deeplearning-suda.atlassian.net/3d2399a6-80ac-4808-82cf-b344e93ccfa5#media-blob-url=true&id=1c7882c4-a6f7-4603-927d-303811580562&collection=contentId-334135297&contextId=334135297&mimeType=image%2Fpng&name=MB2%40C_ZJB92%25GV0K(%40HT%600X-20211023-141451.png&size=224215&height=424&width=958&alt=)

 