# LightRAG个人学习笔记

## References
[1] [LightRAG](https://github.com/HKUDS/LightRAG)  

# 项目使用说明

### 运行环境
使用`pip`安装`requirements.txt`中的少数几个必备依赖即可  
原项目通过`pipmaster`依赖管理工具，可以在运行时根据需求自动安装依赖

### 运行与调试入口
项目根目录下`main.py`文件

### 测试数据目录
采用短篇小说《孔乙己》作为测试文本  
原文和建立后的索引数据库在`./kongyiji`目录下

### 知识图谱可视化
基于`pyvis`实现  
代码见`./visualize_graph/visualize_graph.ipynb`  
可视化结果可以直接在`Jupyter Notebook`界面查看，也可以用浏览器打开生成的html文件  
界面第一次显示有点慢，推测与某些必要的css与js文件在外网有关  
第一次显示后，当前路径下会出现`lib`文件夹，其中缓存了这些css与js文件  
确保`lib`文件夹与涉及`pyvis`的`ipynb`文件或`html`文件在同一文件夹下，之后的页面会自动加载这些文件，速度会变快  
鼠标放在节点或边上，会显示对应的详细信息  
节点可以拖动，拖动结束后会自动调整位置与图的形状

## 个人感想与思考
&emsp;&emsp;我在最开始做RAG的时候，就发现了这样一个问题：用户的问题与相关的参考文本，即`【query&corpus】`之间，存在明显的语义鸿沟（Semantic Gap）。  
&emsp;&emsp;比如：`红烧滩羊肉怎么做？`这个问题，与参考文本`红烧滩羊肉的菜谱\n原料：...... 烹饪过程：......`之间，无论是文本字面内容，还是语义，区别都很大。  
&emsp;&emsp;我们采用的Embedding模型性能越好，即越是能准确地实现语义到向量空间的映射，编码后的`【query&corpus】`的向量的相似度，反而有可能越低。
这会导致真正相关的参考文本，混在一堆相似度不高不低的无关文本中，难以区分。    
&emsp;&emsp;那么，如何跨越语义鸿沟？  
&emsp;&emsp;我之前使用过的一种常用方法，被称为`Q&A对编码`，其核心思路是：用LLM生成每个文本切片相关的、用户可能会问到的问题。
然后对这些问题进行embedding，实现`【query&question】`之间的语义相似度检索。检索器返回结果时将检索到的问题替换为关联的文本切片即可。  
&emsp;&emsp;该方法简单、低成本、且能显著弥合语义鸿沟，但也存在一些缺陷，比如：  
&emsp;&emsp;&emsp;①. 生成的问题只与来源文本切片关联，无法处理涉及多个文本切片的问题。比如法律条款间的互相引用、全局性概括性表述等。  
&emsp;&emsp;&emsp;②. 生成问题的过程是与LLM的单轮对话，且生成的问题一般都是短句。
在多轮对话场景下，尤其是用户对前文进一步追问时，需要对新的query进行压缩，在实现准确和不丢失关键信息的同时，又要与已有的`【query&question】`风格相似。这是一件不确定性很大的事情。  
&emsp;&emsp;&emsp;③. 在目前流行的Agentic RAG范式中，LLM自行决定需要查询数据库时，给出的query未必是一个问题，更常见的情况是一个陈述性短句，
比如`我需要查找红烧滩羊肉相关的信息`，或者干脆就是几个关键词。在这种情况下，语义鸿沟仍然存在。  
&emsp;&emsp;当然，以上问题并不是`Q&A对编码`方法独有的问题，而是几乎所有RAG方案都需要直面的普遍性问题。许多tricks都可以有效缓解，比如：父子文档检索、通过打标签记录文本切片间的引用关系实现伪知识图谱、
各种query增强方法、指代消岐、结合传统搜索引擎的关键词检索算法等等。  
&emsp;&emsp;在我看来，`Q&A对编码`方法比较适合语料库本身有一定程度的结构化、每个文本切片都是比较完整且独立的章节的场景。
如果语料库比较松散，最好还是根据具体文本内容与格式，定制文档预处理方法，在数据工程方面下一些功夫。  
&emsp;&emsp;那么，有没有哪种方法，既可以弥合语义鸿沟，又可以同时照顾到上述问题，还不需要复杂的数据工程呢？  
&emsp;&emsp;基于知识图谱的RAG方法符合这一要求。  
&emsp;&emsp;很容易可以想到这样一种方案：对所有文本切片进行实体与关系提取，构建一个知识图谱。
在问答时，对用户的query也进行NER，用提取到的实体名匹配知识图谱中已有的实体名，实现`【entity name&entity name】`之间的语义相似度检索。  
&emsp;&emsp;该方案过于简单了，完全就是传统图搜索引擎的关键词匹配方法。其并没有实现对语义鸿沟的弥合，而是彻底放弃了语义信息。
检索结果中会包含大量的无关文本切片，且无法区分出哪些切片是真正相关的。此外，指代消岐也会是个麻烦。  
&emsp;&emsp;那么，合理的Graph-based RAG方案应该如何实现呢？  
&emsp;&emsp;一个比较热门的项目，就是微软开源的[GraphRAG](https://github.com/microsoft/graphrag)。我之前也仔细阅读过该项目的源码，笔记分享在[GraphRAG-learn](https://github.com/YueZhengMeng/GraphRAG-learn)  
&emsp;&emsp;关于GraphRAG详细的解析，还请移步到笔记了解，这里我直接说结论：  
&emsp;&emsp;许多人认为GraphRAG是一个高准确度的、可以应用的RAG框架，实则是对“GraphRAG”这个项目名望文生义了。  
&emsp;&emsp;在我看来，GraphRAG正如其论文标题《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》所描述的那样，
是一次关于如何通过RAG实现文本宏观语义理解与总结性问答的探索性尝试，而非具备实际应用价值的高效算法或工程化框架。  
&emsp;&emsp;尤其是当我发现，每次添加新文本，都需要重建整个索引时，我更加确信了我的判断————毕竟，RAG技术最关键的实践价值，
就是通过动态、实时、低成本的增量式知识注入来扩展大语言模型的知识边界。  
&emsp;&emsp;我通过一段时间对思考与寻找，发现LightRAG项目解决了GraphRAG存在的问题，并且满足以上讨论的所有需求。  
&emsp;&emsp;那么，LightRAG相比于GraphRAG做了哪些改进呢？  
&emsp;&emsp;&emsp;①. GraphRAG使用用户的query与实体的描述进行语义相似度匹配，即`query&entity description`，仍存在语义鸿沟。
LightRAG会先调用LLM，基于query生成`high level keywords`与`low level keywords`两种关键词，每种关键词都有4个左右。  
&emsp;&emsp;`low level keywords`主要侧重于检索特定实体及其相关的属性或关系。面向细节，旨在提取有关图中特定节点或边的精确信息。  
&emsp;&emsp;`high level keywords`主要涉及更广泛的话题和总体性主题。聚合多个相关实体和关系的信息，提供了对更高层次概念和摘要的见解，而不是具体细节。  
&emsp;&emsp;在Local模式下，`low level keywords`会被拼接为字符串，然后用该字符串匹配实体的描述，实现`low level keywords&entity description`的检索。
得到topk个匹配的实体后，再获取与这些实体最相关的关系与文本切片，一起组装为参考上下文。  
&emsp;&emsp;在Global模式下，`high level keywords`会被拼接为字符串，然后用该字符串匹配关系的描述，实现`high level keywords&relation description`的检索。
得到topk个匹配的关系后，再获取与这些关系最相关的实体与文本切片，一起组装为参考上下文。  
&emsp;&emsp;虽然该方案未必能完全解决语义鸿沟，但其包含的语义信息丰富，检索结果充分。  
&emsp;&emsp;在生成关键词时，可以把多轮对话历史记录做为context，实现多轮对话场景下的检索器输入的对齐。  
&emsp;&emsp;&emsp;②. LightRAG在关键词检索完成，得到topk个匹配的实体或关系后，会再获取它们在图谱中一跳内的实体或关系，并获取与它们最相关的文本切片。
这就实现了涉及多个文本切片的相关信息的获取与汇总。相比于GraphRAG中，先划分社区再总结社区上下文，并检索到最相关的社区上下文作为参考上下文的方法，要更加简单且灵活。  
&emsp;&emsp;但是在更广泛的概括性的问答场景中，仅一跳的外延距离可能不够。LightRAG源码中对一跳的距离进行了硬编码，如果有相关需求，必须修改源码。  
&emsp;&emsp;&emsp;③. GraphRAG在构建索引时，对于多次被提取到的实体或关系，会记录并汇总每次的description，并在所有文本的实体与关系提取结束时，调用LLM进行总结，每个实体或关系只保留一条总结后的description。  
&emsp;&emsp;之后，GraphRAG会使用层次化莱顿社区划分算法，将知识图谱划分成多个多层次的社区，并根据每个社区内的实体和关系的description，调用LLM总结社区description。  
&emsp;&emsp;以上两个步骤，导致在GraphRAG中新增文档，需要重新总结实体或关系的description、重新划分社区、重新总结社区description。最终的结果就是，必须从头重建索引。  
&emsp;&emsp;LightRAG中不存在社区划分与总结相关的内容。对于多次被提取到的实体或关系，直接拼接每次的description，中间用分隔符`<SEP>`隔开。
只有当拼接后的description超过预设最大长度时，才会调用LLM进行总结。这就实现了文档的动态新增，无需重建索引。  
&emsp;&emsp;总结：在我看来，LightRAG兼顾了索引构建成本、检索速度、检索结果质量、灵活性与可扩展性。其最显著的优点是算法简单、快捷、高效。我在未来需要搭建RAG系统时，会首选LightRAG的思路作为基座。    