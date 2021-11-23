# Make-Information-Extraction-Great-Again
An (incomplete) overview of information extraction

## Named Entity Recognition

## Relation Extraction

### Supervised Sentence-level Relation Extraction

#### What is it?
#### What are the challenges?

- 语义理解
- SPO 和 EPO
- entity提取跟relation提取之间的交互
- relation之间相互依赖

#### Mainstream methods?

- Sequence Labelling
  - [Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme](https://aclanthology.org/P17-1113/)(ACL2017): 序列标注，tag有三部分，第一部分表示实体开头/中间/结尾，第二部分表示所属关系，第三部分表示是头实体还是尾实体
  - [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction](https://aclanthology.org/2020.acl-main.136/) (ACL2020): 先序列标注出头实体，再根据头实体以及某个特定关系序列标注出尾实体
  - [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction](https://aclanthology.org/2021.acl-long.486/)(ACL2021): 同样先标头实体再标relation-specific的尾实体，改进在于先判断可能的relation，有可能出现的才去标对应的尾实体
- Sequence to Sequence
  - [Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism](https://aclanthology.org/P18-1047/)(ACL2018): 输入句子，输出提取结果序列，结果序列格式是 => r1, h1, t1, r2, h2, t2, ...
  - [CopyMTL: Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task Learning](https://arxiv.org/abs/1911.10438)(AAAI2020): 同上，做了改进
  - [Learning the Extraction Order of Multiple Relational Facts in a Sentence with Reinforcement Learning](https://aclanthology.org/D19-1035/)(EMNLP2019): 提取结构本来无序但是序列生来有序，用强化学习解决这个问题
  - [Minimize Exposure Bias of Seq2Seq Models in Joint Entity and Relation Extraction](https://aclanthology.org/2020.findings-emnlp.23/)(EMNLP2020 findings): Seq2Seq的方法time step过长导致exposure bias，所以尝试把sequential的decoder变成tree
  - [Effective Modeling of Encoder-Decoder Architecture for Joint Entity and Relation Extraction](https://arxiv.org/pdf/1911.09886.pdf)(AAAI2020):
- Question Answering
  - [Entity-Relation Extraction as Multi-turn Question Answering](https://aclanthology.org/P19-1129/)(ACL2019): 建模成多次问答
- Table
  - [Table Filling Multi-Task Recurrent Neural Network for Joint Entity and Relation Extraction](https://aclanthology.org/C16-1239/)(COLING2016): 使用RNN模型按预定义顺序遍历表格中cell捕捉cell之间依赖
  - [End-to-End Neural Relation Extraction with Global Optimization](https://aclanthology.org/D17-1182/)(EMNLP2017): 类似上文，但是引入syntactic信息 + 基于beam search的global optimization
  - [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://aclanthology.org/2020.coling-main.138/)(COLING2020): 表格中的cell不是直接表示两个词之间的关系，而变成更细粒度的比如两个词是否属于同一个entity的第一个词与最后一个词、两个词是否分别是某个关系中的头实体跟尾实体的第一个词等等
  - [Two are Better than One: Joint Entity and Relation Extraction with Table-Sequence Encoders](https://aclanthology.org/2020.emnlp-main.133/)(EMNLP2020): 提出了table encoder以及序列encoder，table encoder内部cell会交互，table跟sequence的encoder也会交互
  - [UniRE: A Unified Label Space for Entity Relation Extraction](https://aclanthology.org/2021.acl-long.19/)(ACL2021): 实体类型跟关系类型放到同一个label dpan，提出三个不同的预训练任务
- Others
  - [Relation Classification via Convolutional Deep Neural Network](https://aclanthology.org/C14-1220/)(COLING2014): 早期工作，CNN-based，给定实体对做关系分类
  - [Relation Classification via Recurrent Neural Network](https://arxiv.org/abs/1508.01006)(arXiv): 早起工作，RNN-based，给定实体对做关系分类
  - [A Frustratingly Easy Approach for Entity and Relation Extraction](https://aclanthology.org/2021.naacl-main.5/)(NAACL2021): pipeline，先找实体，再在实体左右加特殊符号做RE，加context会带来帮助
  - [Extracting Entities and Relations with Joint Minimum Risk Training](https://aclanthology.org/D18-1249/)(EMNLP2018): 使用joint minimum risk training更显式地考虑实体识别与关系识别两个任务的交互
  - [Recurrent Interaction Network for Jointly Extracting Entities and Classifying Relations](https://aclanthology.org/2020.emnlp-main.304/)(EMNLP2020): 实体识别跟关系识别模块像RNN在多个时刻不断地交互，增强他们之间的显式交互而不是简单共享encoder做multi-task

#### Datasets?

### Distant Supervised Relation Extraction

#### What is it?
#### What are the challenges?

- 如何去噪
#### Mainstream methods?

- Classical
  - [Distant Supervision for Relation Extraction via Piecewise
Convolutional Neural Networks](https://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf)(EMNLP2015): PCNN + bag里边选max probability的instance作为预测结果
- Better Selection or Aggregation Methods
  - [Relation Extraction with Multi-instance Multi-label
Convolutional Neural Networks](https://aclanthology.org/C16-1139/)(COLING2016): 提出跨句子max池化从而更好利用bag里边cross-sentence的信息来共同推断存在的关系
  - [Neural Relation Extraction with Selective Attention over Instances](https://aclanthology.org/P16-1200/)(ACL2016): bag里边的多个句子用attention聚合
  - [Large Scaled Relation Extraction with Reinforcement Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16257/16125)(AAAI2018): 使用强化学习来select sentences
  - [DSGAN: Generative Adversarial Training for Distant Supervision
Relation Extraction](https://aclanthology.org/P18-1046/)(ACL2018): 使用GAN来挑选positive sentence，Generator打分，Discriminator判断
  - [Self-Attention Enhanced CNNs and Collaborative Curriculum Learning for Distantly Supervised Relation Extraction](https://aclanthology.org/D19-1037/)(EMNLP2019): 使用两个网络互相约束和挑sentence
- External Knowledge
  - [RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information](https://aclanthology.org/D18-1157/)(EMNLP2018)
  - [GAN Driven Semi-distant Supervision for Relation Extraction](https://aclanthology.org/N19-1307/)(NAACL2019)
  - [Long-tail Relation Extraction via Knowledge Graph Embeddings and Graph Convolution Networks](https://aclanthology.org/N19-1306/)(NAACL2019)


#### Datasets?

- [Towards Accurate and Consistent Evaluation: A Dataset for
Distantly-Supervised Relation Extraction](https://aclanthology.org/2020.coling-main.566/)(COLING2020)

### Few-shot Relation Extraction

#### What is it?
#### What are the challenges?

- 编码support set与query的时候更好的感知交互，包括support set内部交互、support set与query的交互
- 更robust，更好地处理noise

#### Mainstream methods?

- [Hybrid Attention-Based Prototypical Networks for Noisy Few-Shot Relation Classification](https://ojs.aaai.org//index.php/AAAI/article/view/4604)(AAAI2019): 聚合prototype vector的时候考虑跟query的关系进行softmax聚合 + 不同维度计算距离的时候有不同重要性
- [Multi-Level Matching and Aggregation Network for Few-Shot Relation Classification](https://aclanthology.org/P19-1277/)(ACL2019): encode的时候考虑support set跟query的交互 + 聚合prototype vector的时候考虑query attention做softmax聚合 + 引入同类间距离要小的auxiliary loss约束
- [Meta-Information Guided Meta-Learning for Few-Shot Relation Classification](https://aclanthology.org/2020.coling-main.140/)(COLING2020): 基于MAML进行改进提出MIML，区别主要在于MAML是为了得到最佳快速adapt到某个任务的初始参数，MIML是得到最佳的meta-parameter，这个meta-parameter能够根据某个任务生成一个能快速adapt到这个任务的初始化参数
- [Few-shot Relation Extraction via Bayesian Meta-learning on Relation Graphs](https://proceedings.mlr.press/v119/qu20a)(ICML2020): 引入贝叶斯学习+图网络
- [Label Verbalization and Entailment for Effective Zero and Few-Shot Relation Extraction](https://aclanthology.org/2021.emnlp-main.92/)(EMNLP2021): formulate成NLI任务来做，这样可以先用大量NLI数据做pre-train，将知识迁移过来
- [ZS-BERT: Towards Zero-Shot Relation Extraction with Attribute
Representation Learning](https://aclanthology.org/2021.naacl-main.272/)(NAACL2021): 学习两个映射，一个映射句子，一个映射关系的文本描述，这样只需要关系的文本描述就可以分类实现zero-shot


#### Datasets?

- [FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation](https://aclanthology.org/D18-1514/)(EMNLP2018)
- [FewRel 2.0: Towards More Challenging Few-Shot Relation Classification](https://aclanthology.org/D19-1649/)(EMNLP2019)

### Document-level Relation Extraction

#### What is it?
#### What are the challenges?

- 每个entity有多个不同的mention
- 两个entity分布在不同的句子，跨句子关系
- 需要推理的关系

#### Mainstream methods?

- Graph
  - Word-level Graph (图的节点是每个token)
    - [Cross-Sentence N-ary Relation Extraction with Graph LSTMs](https://aclanthology.org/Q17-1008/)(EMNLP2017): dependency tree的边、相邻词的边、相邻句子root的边 + Graph LSTM
    - [Inter-sentence Relation Extraction with Document-level Graph Convolutional Neural Network](https://aclanthology.org/P19-1423/)(ACL2019): 每个句子基于dependency tree建图，之后还有相邻句子边、相邻词边、共指边和自环边
    - [Coarse-to-Fine Entity Representations for Document-level Relation Extraction](https://arxiv.org/abs/2012.02507)(arXiv): 基于dependency tree建图，同样有相邻句子、相邻词、共指、自环边 + 每个实体对在图上找各个mention之间路径再聚集起来预测
  - Non-word-level Graph (图的节点不是token而是mention/entity/sentence等)
    - [Connecting the Dots: Document-level Neural Relation Extraction with Edge-oriented Graphs](https://aclanthology.org/D19-1498/)(EMNLP2019): 图的节点包括mention、entity和sentence三种，启发式连边，之后与《[A Walk-based Model on Entity Graphs for Relation Extraction](https://aclanthology.org/P18-2014/)》类似基于其他边进行边信息聚合
    - [Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://aclanthology.org/2020.acl-main.141/)(ACL2020): 先基于dependency tree抽出关键节点以及mention node、entity node来构图，之后不断refine这个graph
    - [Double Graph Based Reasoning for Document-level Relation Extraction](https://aclanthology.org/2020.emnlp-main.127/)(EMNLP2020): Mention-level Graph解决跨句子依赖，Entity-level Graph解决推理
    - [Global-to-Local Neural Networks for Document-Level Relation Extraction](https://aclanthology.org/2020.emnlp-main.303/)(EMNLP2020): 同样图上有mention、entity和sentence，用图来捕捉跨句子交互 + 对于实体对更优的通过mention表示聚合成entity表示的方式 + 预测实体对relation的时候考虑其他实体对的关系表示
    - [The Dots Have Their Values: Exploiting the Node-Edge Connections in Graph-based Neural Models for Document-level Relation Extraction](https://aclanthology.org/2020.findings-emnlp.409/)(EMNLP2020 findings): 基于前面《Connceting the Dots》的改进，原来只考虑边，现在把节点表示也考虑上
    - [Document-Level Relation Extraction with Reconstruction](https://arxiv.org/abs/2012.11384)(AAAI2021): 图上有entity、mention、sentence节点 + 核心思路是对于存在某个关系的实体对在图上有一条meta path
  - Both
    - [Document-level Relation Extraction with Dual-tier Heterogeneous Graph](https://aclanthology.org/2020.coling-main.143/)(COLING2020): 首先一个基于dependency tree的syntactic graph，之后接一个以mention、entity为节点的semantic graph
- Non-graph
  - [Simultaneously Self-Attending to All Mentions for Full-Abstract Biological Relation Extraction](https://aclanthology.org/N18-1080/)(NAACL2018): Transformer+Convolution+Biaffine
  - [Entity and Evidence Guided Document-Level Relation Extraction](https://aclanthology.org/2021.repl4nlp-1.30/)(repl4nlp@ACL2021): 引入evidence prediction作为auxiliary task + 编码的时候输入concat上头实体
  - [Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling](https://ojs.aaai.org/index.php/AAAI/article/view/17717/17524)(AAAI2021):引入一个辅助类来做adaptive threshold处理multi-label问题 + 每个实体对更有针对性地提取context用于关系预测
  - [Entity Structure Within and Throughout: Modeling Mention Dependencies for Document-Level Relation Extraction](https://arxiv.org/abs/2102.10249)(AAAI2021): 魔改Transformer的self-attention模块，根据token跟token之间是否属于同一个句子、是否属于同一个实体、是否一个为实体一个为非实体之类的分为总共六类，每一类有不同的attention计算方法
  - [Multi-view Inference for Relation Extraction with Uncertain Knowledge](https://arxiv.org/abs/2104.13579)(AAAI2021): 引入外部knowledge base协助编码
  - [Document-level Relation Extraction as Semantic Segmentation](https://arxiv.org/abs/2106.03618)(IJCAI2021): 借鉴了CV中的U-Net
  - [Learning Logic Rules for Document-level Relation Extraction](https://aclanthology.org/2021.emnlp-main.95/)(EMNLP2021): 将关系的一些推理规则当成隐变量
  - May Be We Do NOT Need All Sentences
    - [Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction](https://aclanthology.org/2021.acl-short.126/)(ACL2021): 对于每个实体对只需要最多三个句子就可以提取出来关系了
    - [SIRE: Separate Intra- and Inter-sentential Reasoning for Document-level Relation Extraction](https://aclanthology.org/2021.findings-acl.47/)(ACL2021 findings): 如果实体对共现在同一个句子那么只需要用intra-sentence的表示即可，否则采用inter-sentence的表示 + 考虑多个关系的逻辑推理
    - [Discriminative Reasoning for Document-level Relation Extraction](https://aclanthology.org/2021.findings-acl.144/)(ACL2021 findings): 定义三种推理路径，对于每个实体对抽这些路径来获取hidden state进行预测
    - [Eider: Evidence-enhanced Document-level Relation Extraction](https://arxiv.org/abs/2106.08657)(arXiv): 使用整个文档预测的同时，引入另一个branch预测evidence sentences，然后只用这部分sentences组成伪文档进行预测，将两部分结果综合
    - [SAIS: Supervising and Augmenting Intermediate Steps for Document-Level Relation Extraction](https://arxiv.org/abs/2109.12093)(arXiv): Eider的基础上，前面引入共指等更多的中间auxiliary tasks

#### Datasets?

### Open Relation Extraction

#### What is it?
#### What are the challenges?
#### Mainstream methods?
#### Datasets?

## Event Extraction

### Supervised Sentence-level Event Extraction

#### What is it?
#### What are the challenges?
#### Mainstream methods?
#### Datasets?

### Few-shot Event Extraction

#### What is it?
#### What are the challenges?
#### Mainstream methods?
#### Datasets?

### Document-level Event Extraction

#### What is it?
#### What are the challenges?
#### Mainstream methods?
#### Datasets?

### Relations Among Events

#### What is it?
#### What are the challenges?
#### Mainstream methods?
#### Datasets?
