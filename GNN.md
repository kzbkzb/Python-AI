***
**本文已加入 [**🚀 Python AI 计划**](https://github.com/kzbkzb/Python-AI)，从一个Python小白到一个AI大神，你所需要的所有知识都在 [这里](https://github.com/kzbkzb/Python-AI) 了。**
***

tf_geometric 是一个高效且友好的图神经网络库，同时支持TensorFlow 1.x 和 2.x。

受到  **rusty1s/pytorch_geometric** 项目的启发，我们为TensorFlow构建了一个图神经网络（GNN）库。
[tf_geometric](https://github.com/CrawlScript/tf_geometric) 同时提供面向对象接口（OOP API）和函数式接口（Functional API），你可以用它们来构建有趣的模型。


* **Github主页:** [https://github.com/CrawlScript/tf_geometric](https://github.com/CrawlScript/tf_geometric)
* **论文:** [Efficient Graph Deep Learning in TensorFlow with tf_geometric](https://arxiv.org/abs/2101.11552)



   <p align="center">
   <img src="https://raw.githubusercontent.com/CrawlScript/tf_geometric/master/TF_GEOMETRIC_LOGO.png" style="max-width: 400px; width: 100%;"/>
   </p>



# 高效且友好的API
----

tf_geometric使用消息传递机制来实现图神经网络：相比于基于稠密矩阵的实现，它具有更高的效率；相比于基于稀疏矩阵的实现，它具有更友好的API。
除此之外，tf_geometric还为复杂的图神经网络操作提供了简易优雅的API。
下面的示例展现了使用tf_geometric构建一个图结构的数据，并使用多头图注意力网络（Multi-head GAT）对图数据进行处理的流程：

```python3

   # coding=utf-8
   import numpy as np
   import tf_geometric as tfg
   import tensorflow as tf

   graph = tfg.Graph(
       x=np.random.randn(5, 20),  # 5个节点, 20维特征
       edge_index=[[0, 0, 1, 3],
                   [1, 2, 2, 1]]  # 4个无向边
   )

   print("Graph Desc: \n", graph)

   graph.convert_edge_to_directed()  # 预处理边数据，将无向边表示转换为有向边表示
   print("Processed Graph Desc: \n", graph)
   print("Processed Edge Index:\n", graph.edge_index)

   # 多头图注意力网络（Multi-head GAT）
   gat_layer = tfg.layers.GAT(units=4, num_heads=4, activation=tf.nn.relu)
   output = gat_layer([graph.x, graph.edge_index])
   print("Output of GAT: \n", output)
```

输出:

```python

   Graph Desc:
    Graph Shape: x => (5, 20)  edge_index => (2, 4)    y => None

   Processed Graph Desc:
    Graph Shape: x => (5, 20)  edge_index => (2, 8)    y => None

   Processed Edge Index:
    [[0 0 1 1 1 2 2 3]
    [1 2 0 2 3 0 1 1]]

   Output of GAT:
    tf.Tensor(
   [[0.22443159 0.         0.58263206 0.32468423]
    [0.29810357 0.         0.19403605 0.35630274]
    [0.18071976 0.         0.58263206 0.32468423]
    [0.36123228 0.         0.88897204 0.450244  ]
    [0.         0.         0.8013462  0.        ]], shape=(5, 4), dtype=float32)
```

# 入门教程
----


## 教程列表
- [安装](https://tf-geometric.readthedocs.io/en/latest/wiki_cn/installation.html)
	- [环境要求与依赖库](https://tf-geometric.readthedocs.io/en/latest/wiki_cn/installation.html#id2)
	- [使用pip一键安装tf_geometric及依赖](https://tf-geometric.readthedocs.io/en/latest/wiki_cn/installation.html#piptf-geometric)
- [快速入门](https://tf-geometric.readthedocs.io/en/latest/wiki_cn/quickstart.html)
	- [使用简单示例快速入门](https://tf-geometric.readthedocs.io/en/latest/wiki_cn/quickstart.html#id2)
	- [面向对象接口（OOP API）和函数式接口（Functional API）](https://tf-geometric.readthedocs.io/en/latest/wiki_cn/quickstart.html#oop-api-functional-api)


## 使用示例进行快速入门


强烈建议您通过下面的示例代码来快速入门tf_geometric：

### 节点分类

* [图卷积网络 Graph Convolutional Network (GCN)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gcn.py)
* [多头图注意力网络 Multi-head Graph Attention Network (GAT)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gat.py)
* [Approximate Personalized Propagation of Neural Predictions (APPNP)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_appnp.py)
* [Inductive Representation Learning on Large Graphs (GraphSAGE)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_graph_sage.py)
* [切比雪夫网络 Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (ChebyNet)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_chebynet.py)
* [Simple Graph Convolution (SGC)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sgc.py)
* [Topology Adaptive Graph Convolutional Network (TAGCN)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_tagcn.py)
* [Deep Graph Infomax (DGI)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_dgi.py)
* [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification (DropEdge)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_drop_edge_gcn.py)
* [基于图卷积网络的文本分类 Graph Convolutional Networks for Text Classification (TextGCN)](https://github.com/CrawlScript/TensorFlow-TextGCN)
* [Simple Spectral Graph Convolution (SSGC/S^2GC)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_ssgc.py)



### 图分类

* [平均池化 MeanPooling](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_mean_pool.py)
* [Graph Isomorphism Network (GIN)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gin.py)
* [自注意力图池化 Self-Attention Graph Pooling (SAGPooling)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sag_pool_h.py)
* [可微池化 Hierarchical Graph Representation Learning with Differentiable Pooling (DiffPool)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_diff_pool.py)
* [Order Matters: Sequence to Sequence for Sets (Set2Set)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_set2set.py)
* [ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations (ASAP)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_asap.py)
* [An End-to-End Deep Learning Architecture for Graph Classification (SortPool)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sort_pool.py)
* [最小割池化 Spectral Clustering with Graph Neural Networks for Graph Pooling (MinCutPool)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_min_cut_pool.py)


### 链接预测

* [图自编码器 Graph Auto-Encoder (GAE)](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_gae.py)

### 保存和载入模型


* [模型保存和载入](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_save_and_load_model.py)
* [使用tf.train.Checkpoint进行模型保存和载入](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_checkpoint.py)

### 分布式训练


* [分布式图卷积网络（节点分类）](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_distributed_gcn.py)
* [分布式平均池化（图分类）](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_distributed_mean_pool.py)



### 稀疏

* [稀疏节点特征](https://github.com/CrawlScript/tf_geometric/blob/master/demo/demo_sparse_node_features.py)



## API列表
-----------------

- [tf_geometric](https://tf-geometric.readthedocs.io/en/latest/modules/root.html)
	- [Graph (Data Structure for a Single Graph)](https://tf-geometric.readthedocs.io/en/latest/modules/root.html#graph-data-structure-for-a-single-graph)
	- [BatchGraph (Data Structure for a Batch of Graphs)](https://tf-geometric.readthedocs.io/en/latest/modules/root.html#batchgraph-data-structure-for-a-batch-of-graphs)
 - [tf_geometric.datasets](https://tf-geometric.readthedocs.io/en/latest/modules/datasets.html#)
 	- [Planetoid](https://tf-geometric.readthedocs.io/en/latest/modules/datasets.html#planetoid)
- [tf_geometric.layers (OOP API)](https://tf-geometric.readthedocs.io/en/latest/modules/layers.html)
- [tf_geometric.nn (Functional API)](https://tf-geometric.readthedocs.io/en/latest/modules/nn.html)

**申明：** 本文中部分文字、案例源于官网，将在后期的更新中不断丰富文中内容以及本文链接所指向的相关文章，如果侵犯了您的权益，可以联系我微.信（mtyjkh_）。
