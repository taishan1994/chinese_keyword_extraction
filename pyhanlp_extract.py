# coding=utf-8
import sys
sys.path.append('..')
import random

from pyhanlp import *
from keyword_extraction.utils import get_test_docs, get_stopwords
"""
pyhanlp=0.1.84
导入后会自动下载：
https://file.hankcs.com/hanlp/hanlp-1.8.3-release.zip
https://file.hankcs.com/hanlp/data-for-1.7.5.zip
停用词位于自己安装的python目录下的Lib\site-packages\pyhanlp\static\data\dictionary
"""

def extract(doc, topk=10):
    """关键词提取"""
    keyword = HanLP.extractKeyword(doc, topk)
    return keyword


def extract_phrase(doc, topk=10):
    """短语提取 """
    phrase_list = HanLP.extractPhrase(doc, topk)
    return phrase_list


def extract_sentence(doc, topk=3):
    """关键句(摘要)抽取"""
    sentence_list = HanLP.extractSummary(doc, topk)
    return sentence_list


if __name__ == "__main__":
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
    print(extract_phrase(text))
    print(extract_sentence(text))