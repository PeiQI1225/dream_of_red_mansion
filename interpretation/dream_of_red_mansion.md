使用KMeans

<img src="C:\Users\佩棋\AppData\Roaming\Typora\typora-user-images\image-20230225102309053.png" alt="image-20230225102309053" style="zoom:80%;" />

# 实战项目概要

# 数据读取与整合

> 数据位于data目录下

数据集介绍如下:

- Dream_of_the_Red Mansion.txt为《红楼梦》小说的txt版本，编码格式为utf-8。
- Red_Mansion_Dictionary.txt为包含《红楼梦》中专有人物的词典，用于辅助分词。

> 避免一会使用jieba库进行分词时，把人名拆分开

- stop_words.txt为停用词表,包含数字、特殊符号等常见的停用词。

## 数据读取

```Python
import pandas as pd
#设置value的显示长度为200，默认为50
pd.set_option('max_colwidth',200)
#显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
#显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)

import warnings
warnings.filterwarnings('ignore')

## 读取文章、停用词和需要的词典
stop_words = pd.read_csv("D:\study\data\dream_of_red_mansion/stop_words.txt",header=None,names = ["stop_words"])
dictionary = pd.read_csv("D:\study\data\dream_of_red_mansion/Red_Mansion_Dictionary.txt",header=None, names=["dictionary"])
content = pd.read_csv("D:\study\data\dream_of_red_mansion/Dream_of_the_Red_Mansion.txt",header=None,names = ["content"])

print(content.head(),dictionary.head(),stop_words.head())
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=Yzg1MDQ4NWU2ZDQ0ZWY4MzlmMTZhOTFkNGJkZWVkZjRfeWhEWVJ5Q3BVb2hvM2tmYXBpTVpzTVhBN2lvdVk5bDJfVG9rZW46Ym94Y25PSUh6Qm1LbVhLSm5BZ2o0MlM3RXBiXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

接下来我们需要对文本数据进行预处理，整合显示格式。首先需要分析的是读取的数据是否存在缺失值，可以使用Pandas中的isnull() 函数进行判断。

```Python
import numpy as np
print(np.sum(pd.isnull(content)))
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NjE3YjBiODQyOGNhMWEwOTg5ODIxOGNiNGNkNzA5Y2FfcnVsTmRyeHJGa2V4anNlTmx5RUozVENPc2F1ME4xRVRfVG9rZW46Ym94Y25DR0J4Tm1XdE5kTlRCd3FTMHg0REZmXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

*可以看出数据中不存在空白行*

## 卷序号字段处理

为了观察的美观和简便，我们删除第1卷、第2卷 等文字占用的行，使用正则表达式进行匹配，将满足条件的索引进行筛选。

```Python
# 使用正则表达式，选取相应索引
index_of_juan = content.content.str.contains("^第+.+卷")

# 根据索引删除不需要的行，并重新设置索引
content = content[~index_of_juan].reset_index(drop=True)
print(content.head())
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDJiZTFlNjJkN2M4OWI1NWJiY2NlNDc1Y2U1YjM3ZjFfMUR1N29UN1RnSzZrWlVtNE1XNnZ5MnZwOEdsbWxUUnFfVG9rZW46Ym94Y243aFo5aVJFdHEyQkYwWUpuZk9uc2ZPXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

## 章节字段处理

接下来我们提取每个章节的标题，并进行字符的处理。

```Python
# 使用正则表达式，选取相应索引
index_of_hui = content.content.str.match("^第+.+回")

# 根据索引选取每一章节的标题
chapter_names = content.content[index_of_hui].reset_index(drop=True)
print(chapter_names.head())
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ODJmMmIxZWYzNDI5YWE1ZDcxMTJjMzQwYzQ1ODkxMzdfYzRoOGNORWZqekpITmhkcGZ5SURjc3Jyc1B2dTRlcEVfVG9rZW46Ym94Y25KYjM1cjhwUUFGdllrZDZMT201N0JjXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

把章节名进行拆分

```Python
chapter_names_split = chapter_names.str.split(" ").reset_index(drop=True)
print(chapter_names_split.head())
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGM5NjVlMzZmNzEyN2I2OTc2NmQxNjVlMjUzNzI0NDFfSHJ3S1lCaXZ2UUlsYkZhRDJsSGVxUnVmNDA4ZWdVT1JfVG9rZW46Ym94Y243VmltWE9HNVN5amNQZWNqcHdtYXRmXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

## 整合每章节的数据

处理好章节标题后，接着我们计算每一章含有多少行、 多少字，同时将每章节的内容进行整合，形成一个新的DataFrame对象。

> 便于我们了解那些章节的内容会比较多

```Python
# 建立保存数据的数据框
data = pd.DataFrame(list(chapter_names_split),columns=["chapter","left_name","right_name"])
# 添加章节序号和章节名称列
data["chapter_number"] = np.arange(1,121)
data["chapter_name"] = data.left_name+","+data.right_name
# 添加每章开始的行位置
data["start_id"] = index_of_hui[index_of_hui == True].index
# 添加每章结束的行位置
data["end_id"] = data["start_id"][1:len(data["start_id"])].reset_index(drop = True) - 1
data["end_id"][[len(data["end_id"])-1]] = content.index[-1]
# 添加每章的行数
data["length_of_chapters"] = data.end_id - data.start_id
print(data.head())
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NDJlYzhhM2YyNGYxZjA1NjAxYWUxN2JmNjg4M2QwOTFfMzR5YmV2bUVMVzVONGNSU2FzbU4zYnNIR3R1N01FdGJfVG9rZW46Ym94Y250TzVsQUJpd29xNnB1SWp1MWp0b2JmXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

接下来，我们构建每一个章节的内容，同时加入每个章节的字数。

```Python
data["content"] = ''
for i in data.index:
    # 将内容使用""连接
    chapter_id = np.arange(data.start_id[i]+1,int(data.end_id[i]))
    # 每章节的内容替换掉空格
    data["content"][i] = "".join(list(content.content[chapter_id])).replace(" ","")

# 添加每章字数
data["length_of_characters"] = data.content.apply(len)
print(data.head(2))
```

# 剧情趋势和人物词频分析

我们对每个章节数据整合完成后，我们对剧情趋势和人物词频进行分析

## 全文分词

首先，我们对《红楼梦》全文进行一个分词

```Python
import jieba
# 数据表的行列数
row,col = data.shape
# 预定义列表
data["cutted_words"] = ''
# 指定自定义的词典，以便包含jieba词库里没有的词，保证更高的正确率
jieba.load_userdict('D:\study\data\dream_of_red_mansion\Red_Mansion_Dictionary.txt')

for i in np.arange(row):
    # 分词
    cutwords = list(jieba.cut(data.content[i]))
    # 去除长度为1的词
    cutwords = pd.Series(cutwords)[pd.Series(cutwords).apply(len)>1]
    # 去停用词
    cutwords = cutwords[~cutwords.isin(stop_words)]
    data.cutted_words[i] = cutwords.values
# 添加每一章节的词数
data['length_of_words'] = data.cutted_words.apply(len)
print(data['cutted_words'].head())
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NTE2MjFiMzllYjEwZmM2MWFmNzFiZTcxZDVmYWMxNDRfaUNsbmtvNVdISFpDaGJUVHhtZldiOUxuS0RXMFE1UzJfVG9rZW46Ym94Y25FMXZ3cFNUYTBiWEVZTlhsVGtDdGNoXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

**小tips：**jieba库下载超时时，可以使用下面那句命令试试

```Python
pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NGI1NzA2MTA1MDNhNGI4Mjk0NDQ3NDIyY2I1MDI2Y2NfWnB5cjc0ZWFSQkhEYjR5STZiVHFuTUlLQ0tkTWdLYzdfVG9rZW46Ym94Y24zN2RCdkNFU0t3Q2x2d1k1UnhhdlhmXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

## 剧情发展趋势

我们可以绘制散点图展示每一章节的段数、字数，以此来大概观察情节发展的趋势。

```Python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] # 设置字体
plt.rcParams["axes.unicode_minus"]=False # 该语句解决图像中的乱码问题

plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(data.chapter_number,data.length_of_chapters,marker="o", linestyle="-",color = "tomato")

plt.ylabel("章节段数")
plt.title("红楼梦120回")
plt.hlines(np.mean(data.length_of_chapters),-5,125,"deepskyblue")
plt.vlines(80,0,100,"darkslategray")
plt.text(40,90,'前80回',fontsize = 15)
plt.text(100,90,'后40回',fontsize = 15)
plt.xlim((-5,125))

plt.subplot(3,1,2)
plt.plot(data.chapter_number,data.length_of_words,marker="o", linestyle="-",color = "tomato")
plt.xlabel("章节")
plt.ylabel("章节词数")
plt.hlines(np.mean(data.length_of_words),-5,125,"deepskyblue")
plt.vlines(80,1000,3000,"darkslategray")
plt.text(40,2800,'前80回',fontsize = 15)
plt.text(100,2800,'后40回',fontsize = 15)
plt.xlim((-5,125))

plt.subplot(3,1,3)
plt.plot(data.chapter_number,data.length_of_characters,marker="o", linestyle="-",color = "tomato")
plt.xlabel("章节")
plt.ylabel("章节字数")
plt.hlines(np.mean(data.length_of_characters),-5,125,"deepskyblue")
plt.vlines(80,2000,12000,"darkslategray")
plt.text(40,11000,'前80回',fontsize = 15)
plt.text(100,11000,'后40回',fontsize = 15)
plt.xlim((-5,125))

plt.show()
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=MWRiMjlkN2U0Y2FiY2IzNmJmN2UwOWVmYzYxY2EyMDFfbVJQbE1peWtLcXhDY3JHM3Y4REc2RDgwMDNWWU1mQjdfVG9rZW46Ym94Y25CM1NwNGxMM2RCYnhOTmZ2WWJleThiXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

- *蓝色线代表章节平均的段落数和字数，可以看到每一章 平均段落数为25左右，平均次数为1900左右，平均字数为7000左右，在60-80回篇幅最多。*
- *“红楼梦作者究竟是谁？”这个问题引起中国文学界的漫长争论，并持续至今。众多学者认为曹雪芹的原著仅存80回，现存后40回是清人高鄂所续。我们根据灰色线将前80回和后40回进行划分,从这些相互关系可以看出，前80章和后40章还是有一些差异的。*

## 词频统计

在分词完成后，我们可以统计全书的词频，计算每个词出现的频率并排序。

```Python
words = np.concatenate(data.cutted_words)
#统计词频
word_df = pd.DataFrame({"word":words})
word_frequency = word_df.groupby(by=["word"])["word"].agg({"count"}).reset_index()
word_frequency.columns = ["word","frequency"]
word_frequency = word_frequency.reset_index().sort_values(by="frequency",ascending=False)
print(word_frequency.head(10))
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NGE5ODkzNWZiOWZkYmEwMzE4MWI5NDYxNTU1Y2VkMWNfVHZaMVpjYzNORHNnMVhQcWUwbkNxVzFXNVRHbG80eEJfVG9rZW46Ym94Y25oSFJjUGF2Y1k2bDVNdjNWNXF3SUxPXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

使用条形图对上述数据进行一个简单展示

```Python
plt.figure(figsize=(8,10))

frequent_words = word_frequency.loc[word_frequency.frequency > 500].sort_values('frequency')
plt.barh(y = frequent_words["word"],width = frequent_words["frequency"])
plt.xticks(size = 10)
plt.ylabel("关键词")
plt.xlabel("频数")
plt.title("红楼梦词频分析")
plt.show()
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=M2Q1ZmU2OTE5Yzg4YmIyYmY1MzNhMzllNTMzZmViYTZfSHJ4V2x2ZGNscjBScW5lYzFaaFdGTnh2bzdqWWtUOXBfVG9rZW46Ym94Y24wUDlJMVJ4U3N0YlpVd1Npb3FPT2xlXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

从图中可以看出，宝玉出现的次数最多,是红楼梦中的主角。接下来我们通过Python中的wordcloud库进行词云绘制。

```Python
from wordcloud import WordCloud
plt.figure(figsize=(10,5))

wordcloud = WordCloud(font_path='D:\study\data\dream_of_red_mansion\Simhei.ttf', margin=5, width=1800,height=900)
wordcloud.generate("/".join(np.concatenate(data.cutted_words)))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTUyNDMxNGU0YzU5OTA2N2VkZjA0MmZjNWFhZWYwM2JfZFo4TWF0WHFMSEg5YTFGNXFacExSUkhDQnNPbEhiN3JfVG9rZW46Ym94Y25DWkdXMUUyRzBzTmZtR2NqRDdNQzNiXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

对应的词要是出现的频率比较高，在词云图中字体也是比较大的

# 章节聚类及可视化

## 构建词矩阵

在进行文本聚类之前，我们需要将词进行向量化，这里向量化的方式选用计算TF-IDF矩阵。

- TF-IDF含义是词频逆文档频率，如果某个词在一篇文章中出现的频率高，并在其他文章中很少出现，则该词的重要性较高。词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

> 即使一个词中某一行中没有出现也可以获取到一个对应的分数

- TfidfVectorizer模型建立后，可通过fit_transform()函数进行训练，将文本中的词语转换为词的TF-IDF矩阵；通过get_feature_names()可看到所有文本的关键字;通过vocabulary_属性查看关键词编号。TfidfVectorizer模型的输出为矩阵形式，通过toarray()函数可看到TF-IDF矩阵的结果。

我们借助sklearn库中一个提取特征的模块

```Python
from sklearn.feature_extraction.text import TfidfVectorizer

content = []
for cutword in data.cutted_words:
    content.append(" ".join(cutword))

# 构建语料库，并计算文档的TF－IDF矩阵
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(content)

# TF－IDF以稀疏矩阵的形式存储，将TF－IDF转化为数组的形式,文档－词矩阵
word_vectors = tfidf.toarray()
print(word_vectors)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NjM2OGM0ZDE5NWZjM2M4ZDJiMTg2NmNjZDczNWI3NTFfSXNoa3dNaVZxVk5hbDRnNXpQU3NvcmRMWjJpT0RMWkRfVG9rZW46Ym94Y243bFNsOTZjVlQ3VDBQbVVKU1A4aHFxXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

## 使用K-Means聚类

K-means聚类:对于给定的样本集A，按照样本之间的距离大小，将样本集A划分为K个簇$$$$A_1 $$$$,$$$$A_2$$$$, .., $$$$A_k$$$$.让这些簇内的点尽量紧密的连在一起， 而让簇间的距离尽量的大。

K-Means算法是无监督的聚类算法。目的是使得每个点都属于离它最近的均值(此即聚类中心)对应的簇A中。这里使用sklearn库中的K-means聚类算法对数据进行聚类分析，得到每一章所属的簇。

参数聚类数目n_clusters = 3，随机种子random_state = 0。

```Python
from sklearn.cluster import KMeans

# 对word_vectors进行k均值聚类
kmeans = KMeans(n_clusters=3, random_state=0).fit(word_vectors)
# 聚类得到的类别
kmean_labels = data[["chapter_name","chapter"]]
kmean_labels["cluster"] = kmeans.labels_
print(kmean_labels)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NmJkYTc5MTcxODc4ZGU2Yzg1MjkyMDZiMGMwMGNjODhfMllPaWxhTUJFUllnTVpHNHhRT3IzcFRnRmRtRU9qNThfVG9rZW46Ym94Y245NnFydXJXYnhFM0U4T1YyWDc5YmRnXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

> 看见看到我们根据每个章节的内容把全部章节分为了3个簇

查看每个簇有多少章节

```Python
count = kmean_labels.groupby('cluster')['chapter'].count()
print(count)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NTcyMjc1YjI0MWQ0OTIwNWI0ZDNhZGZkNzNmOGIzMjVfaXlNbTdkYkJLY3FGSExJdVRsb1E1cUhJQ3FhZlZodUxfVG9rZW46Ym94Y252UGJJSVgxVWtKU3BpV2Q4NVNsR0toXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

我们通过设置簇的个数为3，可以大致衡量哪些章节的文字内容较为接近，如簇编号为2的章节有第4、79、80、91、103回等，说明这些章节的文本内容距离较近。接下来使用降维技术将TF-IDF矩阵降维，并将K-means聚类的簇对比降维数据进行可视化。

## 拓展学习

### MDS

**多维标度**(Multidimensional scaling,缩写MDS,又译“多维尺度")也称作“相似度结构分析”(Similarity structure analysis)，属于多重变量分析的方法之一，是社会学、数量心理学、市场营销等统计实证分析的常用方法。MDS在降低数据维度的时候尽可能的保留样本之间的相对距离。

```Python
from sklearn.manifold import MDS

# 使用MDS对数据进行降维
mds = MDS(n_components=2,random_state=12)
mds_results = mds.fit_transform(word_vectors)
print(mds_results.shape)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=MWUwYTAwODA0ZGRjNWMwYzMwZTZiMzBiMTNmMjg3OTVfeG8zSEtkM0ZBZ1l6bHk3ZFBzV09QZTE2aUZyUzU5em5fVG9rZW46Ym94Y25uNndBQnJoOHpEM3VxSjI0c0trMFVnXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

```Python
# 绘制降维后的结果
plt.figure(figsize=(8,8))
plt.scatter(mds_results[:,0],mds_results[:,1],c = kmean_labels.cluster)

for i in (np.arange(120)):
    plt.text(mds_results[i,0]+0.02,mds_results[i,1],s = data.chapter_number[i])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means MDS")
plt.show()
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=YjdjNDViODE4YjRkNTcyNjIxYzZhNjJkZWQ4MDE2YjlfbXAxZ2JNTUNxdEljd3JQN1JPUFFEN0s0ejV1MDk3WFZfVG9rZW46Ym94Y25Mc1dmYjFlVFprdHVEUFdzY0lrRHZkXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

*使用MDS将各章的词向量降至2维后，将K-means聚类的簇对比降维数据进行可视化，可以大致验证聚类结果的有效性。如簇0(紫色点)和簇2(黄色)分别展示在图的四周，簇1 (绿色)的章节主要分布在图的中间，每个簇之间的章节相对距离较小。*

### PCA

PCA降维是一种常见的数据降维方法，其目的是在“信息”损失较小的前提下，将高维的数据转换到低维，从而减小计算量。PCA通常用于高维数据集的探索与可视化，还可以用于数据压缩，数据预处理等。

```Python
from sklearn.decomposition import PCA
# 使用PCA对数据进行降维
pca = PCA(n_components=2)
pca.fit(word_vectors)
print(pca.explained_variance_ratio_)
# 对数据降维
pca_results = pca.fit_transform(word_vectors)
print(pca_results.shape)
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=YTU3MjI1YTkwMTQ2YTY1MDhmZmRhNTQzZmZkM2ZjZGFfYkVCT0JhdmJDY3llRFVVUXlLbGtOY0wxZVJWUTh2ejdfVG9rZW46Ym94Y244cUlCellwcThGR3dPTm5lb0FzNVFOXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

```Python
# 绘制降维后的结果
plt.figure(figsize=(8,8))

plt.scatter(pca_results[:,0],pca_results[:,1],c = kmean_labels.cluster)

for i in np.arange(120):
    plt.text(pca_results[i,0]+0.02,pca_results[i,1],s = data.chapter_number[i])
plt.xlabel("主成分1")
plt.ylabel("主成分2")
plt.title("K-means PCA")
plt.show()
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=NjdhYWJhNzg5MGM0Njk4NmFmOWZiMjY1N2UxZTNmNzlfTkV5QUY3dHJwdDhWbWt2T0pFdjh1WXNHTWRBN3l4UUZfVG9rZW46Ym94Y25kdjdkOXpzRWNRQzR6R2ZkeHJOeENoXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

*使用PCA将各章的词向量降至2维后，将K-means聚类的簇对比降维数据进行可视化，可以得出相似的结论。簇2(黄色)的两个主成分相对较小，分布在图的左下部分较多，簇1(绿色)主成分相对较大，分布靠右，验证了聚类结果的有效性。*

### ward

上面已经成功地使用K-means对文档进行聚类和绘图，现在可以试一下另外一种聚类算法。

Ward聚类属于凝聚聚类算法，即每个处理阶段，将距离最小的两个对象分到一个类中。我使用预先计算的余弦距离矩阵计算出距离矩阵，然后将其绘制成树状图。

层次聚类(Hierarchical Clustering)是聚类算法的一种，通过计算不同类别数据点间的相似度来创建一棵有层次的嵌套聚类树。在聚类树中，不同类别的原始数据点是树的最低层，树的顶层是一个聚类的根节点。

```Python
from scipy.cluster.hierarchy import dendrogram,ward
from scipy.spatial.distance import pdist,squareform

# 标签，每个章节的标题
labels = data.chapter_number.values

#计算每章的距离矩阵
cos_distance_matrix = squareform(pdist(word_vectors,'cosine'))

# 根据距离聚类
ward_results = ward(cos_distance_matrix)

# 聚类结果可视化
fig, ax = plt.subplots(figsize=(10, 15))

ax = dendrogram(ward_results,orientation='right', labels=labels)
plt.yticks(size = 8)
plt.title("红楼梦各章节层次聚类")

plt.tight_layout()
plt.show()
```

![img](https://g7hwmghtbu.feishu.cn/space/api/box/stream/download/asynccode/?code=OWUyYTMwNTk5NGVhMDdhZjEzYWRjMzU4NTJkMmI4NzVfUm1rS2R2b0M5SHZJUDlCUmVueGl3QUZZclBEN2NnSmxfVG9rZW46Ym94Y25xUldkU3dvZ3FWdVBNbzVwTGE3enZkXzE2NzcyOTE2OTQ6MTY3NzI5NTI5NF9WNA)

*层次聚类可以清晰地表示章节之间的层次关系,章节和距离最近的章节进行合并，不断递归形成树。从层次聚类树形图中我们可以看出《红楼梦》哪些回最为接近，如114回和115回的距离最为接近,117回和118回的距离最为接近等等...*

# 总结

在本案例中，我们首先对红楼梦120回的文本数据进行了清洗和格式的整理;接着宏观了分析了其中的章节结构、次数和字数等基本情况;然后通过分词、分析词频并通过可视化的方法进行展示;最后通过两种聚类方法对各章节的文本进行聚类,使用两种降维方式对聚类结果进行验证和可视化。《红楼梦》被评为中国古典章回小说的巅峰之作，思想价值和艺术价值极高。关于红楼梦的研究一直是中国传统文学的热点，红学家们众说纷纭，思想观点百花齐放，在此案例中我们从统计分析和文本挖掘等角度对红楼梦进行了一些简单的分析。



[数据分析实战---红楼梦文本聚类 - 掘金 (juejin.cn)](https://juejin.cn/post/7192613612546949177)