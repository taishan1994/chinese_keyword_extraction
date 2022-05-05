import sys
sys.path.append('..')
import random

from smoothnlp.algorithm.phrase import extract_phrase
from keyword_extraction.utils import get_test_docs, get_stopwords

"""
pip install smoothnlp>=0.4.0
"""


def extract(doc, topk=10):
    keyword = extract_phrase(doc, top_k=topk)
    return keyword  # 返回的只有词语


if __name__ == '__main__':
    # 准备数据
    docs = get_test_docs()
    stopwords = get_stopwords()
    len_docs = list(range(len(docs)))
    # 随机获取一个文档
    ind = random.choice(len_docs)
    doc = docs[ind]
    text = doc['content']
    print(ind)
    print(text)
    print("真实的关键词：")
    print(doc['keyword'])
    print(extract(text))
