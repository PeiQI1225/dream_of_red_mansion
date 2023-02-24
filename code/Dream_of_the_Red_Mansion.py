import pandas as pd

# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)

import warnings

warnings.filterwarnings('ignore')

## 读取文章、停用词和需要的词典
stop_words = pd.read_csv("D:\study\data\dream_of_red_mansion/stop_words.txt", header=None, names=["stop_words"])
dictionary = pd.read_csv("D:\study\data\dream_of_red_mansion/Red_Mansion_Dictionary.txt", header=None,
                         names=["dictionary"])
content = pd.read_csv("D:\study\data\dream_of_red_mansion/Dream_of_the_Red_Mansion.txt", header=None, names=["content"])

# print(content.head(),dictionary.head(),stop_words.head())

import numpy as np

# print(np.sum(pd.isnull(content)))

# 使用正则表达式，选取相应索引
index_of_juan = content.content.str.contains("^第+.+卷")

# 根据索引删除不需要的行，并重新设置索引
content = content[~index_of_juan].reset_index(drop=True)
# print(content.head())
# 使用正则表达式，选取相应索引
index_of_hui = content.content.str.match("^第+.+回")

# 根据索引选取每一章节的标题
chapter_names = content.content[index_of_hui].reset_index(drop=True)
# print(chapter_names.head())

chapter_names_split = chapter_names.str.split(" ").reset_index(drop=True)
# print(chapter_names_split.head())

# 建立保存数据的数据框
data = pd.DataFrame(list(chapter_names_split), columns=["chapter", "left_name", "right_name"])
# 添加章节序号和章节名称列
data["chapter_number"] = np.arange(1, 121)
data["chapter_name"] = data.left_name + "," + data.right_name
# 添加每章开始的行位置
data["start_id"] = index_of_hui[index_of_hui == True].index
# 添加每章结束的行位置
data["end_id"] = data["start_id"][1:len(data["start_id"])].reset_index(drop=True) - 1
data["end_id"][[len(data["end_id"]) - 1]] = content.index[-1]
# 添加每章的行数
data["length_of_chapters"] = data.end_id - data.start_id
# print(data.head())

data["content"] = ''
for i in data.index:
    # 将内容使用""连接
    chapter_id = np.arange(data.start_id[i] + 1, int(data.end_id[i]))
    # 每章节的内容替换掉空格
    data["content"][i] = "".join(list(content.content[chapter_id])).replace(" ", "")

# 添加每章字数
data["length_of_characters"] = data.content.apply(len)
# print(data.head(2))

import jieba

# 数据表的行列数
row, col = data.shape
# 预定义列表
data["cutted_words"] = ''
# 指定自定义的词典，以便包含jieba词库里没有的词，保证更高的正确率
jieba.load_userdict('D:\study\data\dream_of_red_mansion\Red_Mansion_Dictionary.txt')

for i in np.arange(row):
    # 分词
    cutwords = list(jieba.cut(data.content[i]))
    # 去除长度为1的词
    cutwords = pd.Series(cutwords)[pd.Series(cutwords).apply(len) > 1]
    # 去停用词
    cutwords = cutwords[~cutwords.isin(stop_words)]
    data.cutted_words[i] = cutwords.values
# 添加每一章节的词数
data['length_of_words'] = data.cutted_words.apply(len)
# print(data['cutted_words'].head())

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的乱码问题

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(data.chapter_number, data.length_of_chapters, marker="o", linestyle="-", color="tomato")

plt.ylabel("章节段数")
plt.title("红楼梦120回")
plt.hlines(np.mean(data.length_of_chapters), -5, 125, "deepskyblue")
plt.vlines(80, 0, 100, "darkslategray")
plt.text(40, 90, '前80回', fontsize=15)
plt.text(100, 90, '后40回', fontsize=15)
plt.xlim((-5, 125))

plt.subplot(3, 1, 2)
plt.plot(data.chapter_number, data.length_of_words, marker="o", linestyle="-", color="tomato")
plt.xlabel("章节")
plt.ylabel("章节词数")
plt.hlines(np.mean(data.length_of_words), -5, 125, "deepskyblue")
plt.vlines(80, 1000, 3000, "darkslategray")
plt.text(40, 2800, '前80回', fontsize=15)
plt.text(100, 2800, '后40回', fontsize=15)
plt.xlim((-5, 125))

plt.subplot(3, 1, 3)
plt.plot(data.chapter_number, data.length_of_characters, marker="o", linestyle="-", color="tomato")
plt.xlabel("章节")
plt.ylabel("章节字数")
plt.hlines(np.mean(data.length_of_characters), -5, 125, "deepskyblue")
plt.vlines(80, 2000, 12000, "darkslategray")
plt.text(40, 11000, '前80回', fontsize=15)
plt.text(100, 11000, '后40回', fontsize=15)
plt.xlim((-5, 125))

plt.show()

words = np.concatenate(data.cutted_words)
# 统计词频
word_df = pd.DataFrame({"word": words})
word_frequency = word_df.groupby(by=["word"])["word"].agg({"count"}).reset_index()
word_frequency.columns = ["word", "frequency"]
word_frequency = word_frequency.reset_index().sort_values(by="frequency", ascending=False)
# print(word_frequency.head(10))

plt.figure(figsize=(8, 10))

frequent_words = word_frequency.loc[word_frequency.frequency > 500].sort_values('frequency')
plt.barh(y=frequent_words["word"], width=frequent_words["frequency"])
plt.xticks(size=10)
plt.ylabel("关键词")
plt.xlabel("频数")
plt.title("红楼梦词频分析")
plt.show()

from wordcloud import WordCloud

plt.figure(figsize=(10, 5))

wordcloud = WordCloud(font_path='D:\study\data\dream_of_red_mansion\Simhei.ttf', margin=5, width=1800, height=900)
wordcloud.generate("/".join(np.concatenate(data.cutted_words)))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer

content = []
for cutword in data.cutted_words:
    content.append(" ".join(cutword))

# 构建语料库，并计算文档的TF－IDF矩阵
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(content)

# TF－IDF以稀疏矩阵的形式存储，将TF－IDF转化为数组的形式,文档－词矩阵
word_vectors = tfidf.toarray()
# print(word_vectors)

from sklearn.cluster import KMeans

# 对word_vectors进行k均值聚类
kmeans = KMeans(n_clusters=3, random_state=0).fit(word_vectors)
# 聚类得到的类别
kmean_labels = data[["chapter_name", "chapter"]]
kmean_labels["cluster"] = kmeans.labels_
# print(kmean_labels)

count = kmean_labels.groupby('cluster')['chapter'].count()
# print(count)

from sklearn.manifold import MDS

# 使用MDS对数据进行降维
mds = MDS(n_components=2, random_state=12)
mds_results = mds.fit_transform(word_vectors)
# print(mds_results.shape)

# 绘制降维后的结果
plt.figure(figsize=(8, 8))
plt.scatter(mds_results[:, 0], mds_results[:, 1], c=kmean_labels.cluster)

for i in (np.arange(120)):
    plt.text(mds_results[i, 0] + 0.02, mds_results[i, 1], s=data.chapter_number[i])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means MDS")
plt.show()

from sklearn.decomposition import PCA

# 使用PCA对数据进行降维
pca = PCA(n_components=2)
pca.fit(word_vectors)
print(pca.explained_variance_ratio_)
# 对数据降维
pca_results = pca.fit_transform(word_vectors)
print(pca_results.shape)

# 绘制降维后的结果
plt.figure(figsize=(8, 8))

plt.scatter(pca_results[:, 0], pca_results[:, 1], c=kmean_labels.cluster)

for i in np.arange(120):
    plt.text(pca_results[i, 0] + 0.02, pca_results[i, 1], s=data.chapter_number[i])
plt.xlabel("主成分1")
plt.ylabel("主成分2")
plt.title("K-means PCA")
plt.show()

from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial.distance import pdist, squareform

# 标签，每个章节的标题
labels = data.chapter_number.values

# 计算每章的距离矩阵
cos_distance_matrix = squareform(pdist(word_vectors, 'cosine'))

# 根据距离聚类
ward_results = ward(cos_distance_matrix)

# 聚类结果可视化
fig, ax = plt.subplots(figsize=(10, 15))

ax = dendrogram(ward_results, orientation='right', labels=labels)
plt.yticks(size=8)
plt.title("红楼梦各章节层次聚类")

plt.tight_layout()
plt.show()
