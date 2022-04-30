import random
import sys

import jieba

sys.path.append('..')
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from keyword_extraction.utils import get_all_docs, get_stopwords, get_test_docs


# 分词后的docs
def extract(doc, topk=10, cut_all=False):
    stopwords = get_stopwords()
    # 第一步，先进行分词
    words = jieba.lcut(doc['content'], cut_all=cut_all)
    # 第二步，去除停用词
    words = " ".join([i for i in words if (i not in stopwords and i != '')])
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    X = vectorizer.fit_transform([words])  # 将文本转为词频矩阵
    tfidf = transformer.fit_transform(X)  # 计算tf-idf，
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    keyword = [(wd, score) for wd, score in zip(word, weight[0])]
    keyword = sorted(keyword, key=lambda x: x[1], reverse=True)[:topk]
    return keyword


if __name__ == '__main__':
    # 准备数据
    docs = get_test_docs()
    len_docs = list(range(len(docs)))
    # 随机获取一个文档
    doc = docs[random.choice(len_docs)]
    print("真实的关键词：")
    print(doc['keyword'])
    extract(doc)
    keywords = extract(doc)
    print('提取的关键词：')
    for keyword, score in keywords:
        print(keyword, score)
