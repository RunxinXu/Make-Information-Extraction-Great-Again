# Make-Information-Extraction-Great-Again
An (incomplete) overview of information extraction

## Named Entity Recognition

## Relation Extraction

### Supervised Sentence-level Relation Extraction

#### What is it?
#### What are the challenges?

- 语义理解
- SPO
- EPO
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
  - [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking](https://aclanthology.org/2020.coling-main.138/)(COLING2020): 表格中的cell不是直接表示两个词之间的关系，而变成更细粒度的比如两个词是否属于同一个entity的第一个词与最后一个词、两个词是否分别是某个关系中的头实体跟尾实体的第一个词等等
  - [Two are Better than One: Joint Entity and Relation Extraction with Table-Sequence Encoders](https://aclanthology.org/2020.emnlp-main.133/)(EMNLP2020): 提出了table encoder以及序列encoder，table encoder内部cell会交互，table跟sequence的encoder也会交互
  - [UniRE: A Unified Label Space for Entity Relation Extraction](https://aclanthology.org/2021.acl-long.19/)(ACL2021): 实体类型跟关系类型放到同一个label space进行预测，以及更科学的从表格中分割出实体的策略
- Graph
  - [A Walk-based Model on Entity Graphs for Relation Extraction](https://aclanthology.org/P18-2014/)(ACL2018): 每个实体是一个点构成全连接图，两个实体之间的边表示关系，该边表示同样based on其它路径表示
  - [Leveraging Dependency Forest for Neural Medical Relation Extraction](https://aclanthology.org/D19-1020/)(EMNLP2019): 利用多棵independent的dependency parsing tree构图
  - [Graph Neural Networks with Generated Parameters for Relation Extraction](https://aclanthology.org/P19-1128/)(ACL2019): 实体做节点，生成边的表示
  - [GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction](https://aclanthology.org/P19-1136/)(ACL2019): 两阶段。第一阶段dependency tree构图，第二阶段用第一阶段预测结果构图做refinement
  - [AGGCN Attention Guided Graph Convolutional Networks for Relation Extraction](https://aclanthology.org/P19-1024)(ACL2019): 多个含有图的层，第一层用dependency tree构图，后面基于attention结果构图，最后把所有层结果利用起来
  - [Joint Type Inference on Entities and Relations via Graph Convolutional Networks](https://aclanthology.org/P19-1131/)(ACL2019): 二分图，实体在一边，关系在另一边
- Span-level
  - [Span-Level Model for Relation Extraction](https://aclanthology.org/P19-1525/)(ACL2019): 用span其实是为了解决nested NER
  - [Span-based Joint Entity and Relation Extraction with Transformer Pre-training](https://arxiv.org/abs/1909.07755)(ECAI2020)
- Pre-trained Model
- Others
  - [Relation Classification via Convolutional Deep Neural Network](https://aclanthology.org/C14-1220/)(COLING2014): 早期工作，CNN-based，给定实体对做关系分类
  - [Relation Classification via Recurrent Neural Network](https://arxiv.org/abs/1508.01006)(arXiv): 早起工作，RNN-based，给定实体对做关系分类
  - [A Frustratingly Easy Approach for Entity and Relation Extraction](https://aclanthology.org/2021.naacl-main.5/)(NAACL2021): pipeline，先找实体，再在实体左右加特殊符号做RE，加context会带来帮助
  - [Label Verbalization and Entailment for Effective Zero and Few-Shot Relation Extraction](https://aclanthology.org/2021.emnlp-main.92/)(EMNLP2021): formulate成NLI任务来做，这样可以先用大量NLI数据做pre-train，将知识迁移过来

#### Datasets?

### Distant Supervised Relation Extraction

#### What is it?
#### What are the challenges?
#### Mainstream methods?
#### Datasets?

### Few-shot Relation Extraction

#### What is it?
#### What are the challenges?
#### Mainstream methods?
#### Datasets?

### Document-level Relation Extraction

#### What is it?
#### What are the challenges?
#### Mainstream methods?
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
