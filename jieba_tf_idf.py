import sys

sys.path.append('..')
import jieba
import jieba.analyse as analyse
import random

from keyword_extraction.utils import get_test_docs, get_stopwords


def extract(doc, topK=10, cut_all=False):
    stopwords = get_stopwords()
    # 第一步，先进行分词
    words = jieba.lcut(doc['content'], cut_all=cut_all)
    # 第二步，去除停用词
    words = " ".join([i for i in words if (i not in stopwords and i != '')])
    # 第三步，提取关键词
    keywords = jieba.analyse.extract_tags(words, topK=topK, withWeight=True)
    return keywords


if __name__ == '__main__':
    # 准备数据
    docs = get_test_docs()
    len_docs = list(range(len(docs)))
    # 随机获取一个文档
    doc = docs[random.choice(len_docs)]
    print("真实的关键词：")
    print(doc['keyword'])
    keywords = extract(doc)
    print('提取的关键词：')
    for keyword, score in keywords:
        print(keyword, score)
