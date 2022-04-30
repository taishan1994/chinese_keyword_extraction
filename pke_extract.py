import sys

sys.path.append('..')
import random
import pke
from keyword_extraction.utils import get_test_docs, get_stopwords

"""
git clone https://github.com/boudinfl/pke.git
cd pke-master
python setup.py install
依赖：
pke=2.0.0
spacy=3.3.0
nltk=3.6.7
networkx=2.5.1
transformers>=3.4.0（就安装3.4.0）
torch>=1.6.0（我这里安装了1.10.2）
tensorflow=1.14.0
去https://github.com/explosion/spacy-models/releases下载：
zh_core_web_trf-3.3.0.tar.gz
然后pip install zh_core_web_trf-3.3.0.tar.gz
"""

def extract_by_tf_idf(doc, topk=10):
    extractor = pke.unsupervised.TfIdf()  # initialize a keyphrase extraction model, here TFxIDF
    extractor.load_document(input=doc, language='zh', normalization='none',  stoplist=stopwords)
    extractor.candidate_selection()  # identify keyphrase candidates
    extractor.candidate_weighting()  # weight keyphrase candidates
    keyword = extractor.get_n_best(n=topk)
    return keyword

def extract_by_kpminer(doc, topk=10):
    extractor = pke.unsupervised.KPMiner()  # initialize a keyphrase extraction model, here TFxIDF
    extractor.load_document(input=doc, language='zh', normalization='none',  stoplist=stopwords)
    extractor.candidate_selection()  # identify keyphrase candidates
    extractor.candidate_weighting()  # weight keyphrase candidates
    keyword = extractor.get_n_best(n=topk)
    return keyword

def extract_by_yake(doc, topk=10):
    extractor = pke.unsupervised.YAKE()  # initialize a keyphrase extraction model, here TFxIDF
    extractor.load_document(input=doc, language='zh', normalization='none',  stoplist=stopwords)
    extractor.candidate_selection()  # identify keyphrase candidates
    extractor.candidate_weighting()  # weight keyphrase candidates
    keyword = extractor.get_n_best(n=topk)
    return keyword

def extract_by_firstrstphrases(doc, topk=10):
    extractor = pke.unsupervised.FirstPhrases()  # initialize a keyphrase extraction model, here TFxIDF
    extractor.load_document(input=doc, language='zh', normalization='none',  stoplist=stopwords)
    extractor.candidate_selection()  # identify keyphrase candidates
    extractor.candidate_weighting()  # weight keyphrase candidates
    keyword = extractor.get_n_best(n=topk)
    return keyword


def extract_by_textrank(doc, topk=10):
    extractor = pke.unsupervised.TextRank()
    extractor.load_document(input=doc, language='zh', normalization='none',  stoplist=stopwords)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyword = extractor.get_n_best(n=topk)
    return keyword


def extract_by_positionrank(doc, topk=10):
    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=doc, language='zh', normalization='none',  stoplist=stopwords)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyword = extractor.get_n_best(n=topk)
    return keyword


def extract_by_topicrank(doc, topk=10):
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=doc, language='zh', normalization='none',  stoplist=stopwords)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyword = extractor.get_n_best(n=topk)
    return keyword


def extract_by_multipartiterank(doc, topk=10):
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=doc, language='zh', normalization='none', stoplist=stopwords)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyword = extractor.get_n_best(n=topk)
    return keyword


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
    # extract_by_textrank(text)
    # extract_by_positionrank(text)
    # extract_by_topicrank(text)
    # extract_by_multipartiterank(text)
    # print(extract_by_tf_idf(text))
    print(extract_by_yake(text))
    # print(extract_by_kpminer(text))
    # print(extract_by_firstrstphrases(text))

