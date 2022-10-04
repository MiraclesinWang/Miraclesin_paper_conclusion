| 序号 | 论文标题                                                     | 更新时间   |
| ---- | ------------------------------------------------------------ | ---------- |
| 1    | 2023-NeurIPS-VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training | 2022.09.30 |

template：

**20xx-where-title**

<font color='vornblue'>核心思想：</font>

xxxx

<font color='vornblue'>代码：</font>xxxx

<font color='vornblue'>相关细节：</font>

<font color='vornblue'>顺便吐个槽：</font>

<font color='vornblue'>启发：</font>



1. **2023-NeurIPS-VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**

   <font color='vornblue'>核心思想：</font>

   将MAE的思想迁移到视频中。除了达到SOTA外还有三个新发现：(1) 即便mask掉90%-95%的视频，依然不影响VideoMAE的重建性能。(2) VideoMAE在极小数据集上可以实现很高的性能。(3) 预训练数据的质量比数量更重要。由于视频本身存在一定的时序关系，单纯mask掉几个像素很容易造成信息泄漏，因而本文提出一种tube mask策略。

   <font color='vornblue'>代码：</font>[VideoMAE](https://github.com/MCG-NJU/VideoMAE)

   <font color='vornblue'>相关细节</font>：

   1. 降采样：视频输入模型后先进行降采样，从中选取包含t个连续帧的一个clip，随后通过时序采样将clip进一步压缩为T帧。本文的时序采样stride随数据集不同而不同，在Kinetics上位4，在Something-Something上则为2。
   2. token embedding：使用大小为$2\times 16\times 16$的cube来进行embedding。
   3. tube masking：根据作者的实验，VideoMAE相比于ImageMAE需要更高的掩码比例，因为视频的信息密度更低。为了防止不同帧之间的信息泄漏问题，同一个视频的不同帧mask的tube都是一样的。
   4. backbone采用ViT和space-time attention的结合。虽然space-time模块的复杂度是平方级的，但是由于输入时大部分的cube token都被mask了，所以该问题得到了一定程度上的改善。
   5. 实验表明，增加mask的比例，确实对模型的性能有所提升。（有一种猜想，可以通过mask语义信息更加明确的像素，或许对模型的性能有帮助）
   6. 模型在数据集之间迁移的性能很高，证明了模型的通用性。
   7. 作者注意到，UCF101 and HMDB51在自己的训练集上训练后在自己的测试集上测试的结果并不如在Something-Something V2预训练并在自己的测试集上测试的结果。作者猜测这有可能是Something-Something V2数据量更大导致的。然而即便削减了Something-Something V2的数据量并进行预训练，性能仍然高于UCF101 and HMDB51在自己的训练集上训练的结果，这验证了预训练数据的质量比数量更重要。
   
   ![img](./video_pretrain_assets/1-1.png)
   
   