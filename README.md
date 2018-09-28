# document_classification

## 定义

这里的document_classification指的是长文本，一个文本由好几个句子组成。

典型的数据集包括yelp2013,2014，imdb

主要方法为嵌入attention信息，包括local，user、product、 cognition等

## 发展脉络

#### 最初：没有使用attention，提出了层次化针对document-level
1、duyu tang   Document modeling with gated recurrent neural network for sentiment classification.

#### 加入local attention信息
2、Zichao Yang,  Hierarchical  attention  networks  for  document  classification.

#### 加入cognition信息
3、yunfei long  A Cognition Based Attention Model for Sentiment Analysis

#### 加入user、product信息，沿着这个信息由多篇文章发表，duyu tang是第一个
4、duyu tang    learning semantic  representations  of  users  and  products  for document level sentiment classification 

5、Neural sentiment classification with user and product attention

attention计算方法a=vtanh（wd+wu+wp+b）

6、dou  capturing user and product information for document level sentiment analysis with deep memory network. 

文档之间进行交互，memory network是同一个u或者同一个p的所有文章dds，与d进行交互，得到dds的attention

7、aaai2018  Improving review representations with user attention and product attention for sentiment classification。

分别计算u和p，ud=vtanh(wd+wu+b)  pd=vtanh(wd+wp+b)   upd=vtanh(wd+wu+wp+b)  计算得到3个loss，然后每个loss有一个占比，加权相加得到总loss

8、yunfei long   Dual Memory Network Model for Biased Product Review Classification

2个memory network，u和p分开，u（d)属于同一个u的d，p(d)属于同一个p的d构成memory network，然后计算attention，多跳机制，得到最后的du和dp，然后
du和dp加权相加得到最后的d

9、Cold-Start Aware User and Product Attention for Sentiment Classification   acl20 8

这里是考虑冷启动情况，公式很复杂，我也不理解

## 结果比较acc
Model  | imdb | yelp2013 | yelp2014 | yelp2015
------------- | ------------- | ------------- | ------------- | -------------
 1  | 0.453   | 0.651 | 0.671  | 0.676 
 2  | 0.494 | 0.682 | 0.705 | 0.710 
 3  | 0.489 | 0.638 | 0.641 | 单元格内容 
 4  | 0.435 | 0.596 | 0.608 | / 
 5  | 0.533 | 0.650 | 0.667 | /
 6  | 0.465 | 0.639 | 0.613 | /
 7  | 0.550| 0.683 | 0.686 | /
 8  | 0.539 | 0.662 | 0.676 | / 
 9  | 0.542 |0.657 | /  | /

## 我的一些想法
关于user、product分别attention得到表示du、dp之后，可以使用一个gate进行选择，类似于8中最后要分别给一个权重，这里的权重通过gate获得，类似的gate想法参考Gated Neural Networks for Targeted Sentiment Analysis这篇文章。

试验有做，但是效果并不是很好，但我觉得这个想法应该不错吧。
