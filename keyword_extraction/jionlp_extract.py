import sys
sys.path.append('..')
import random

import jionlp as jio
from keyword_extraction.utils import get_test_docs, get_stopwords

"""
pip install jionlp
jionlp=1.3.53
默认会下载：
https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip
https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip
最后还提示需要安装jdk8，可以去这里下载：
https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
"""


def extract(doc, topk=10):
    keyword = jio.keyphrase.extract_keyphrase(doc)[:topk]
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
