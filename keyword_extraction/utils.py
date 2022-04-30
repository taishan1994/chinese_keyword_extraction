import json
import re


def cut_sentences(sent):
    """
    将文档划分为句子
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")


def get_all_docs():
    """或测一千条带关键词的文档"""
    with open('../data/all_docs.txt', 'r', encoding='utf-8') as fp:
        docs = fp.read().strip().split('\n')
    docs = [i.split('\x01')[2] for i in docs]
    print(docs[0])
    return docs


def get_test_docs():
    """或测一千条带关键词的文档"""
    with open('../data/test.json', 'r', encoding='utf-8') as fp:
        docs = json.loads(fp.read())
    return docs


def get_stopwords():
    with open('../data/stopwords.txt', 'r', encoding='utf-8') as fp:
        stopwords = fp.read().strip().split('\n')
    return stopwords


def get_seg_txt():
    docs = get_all_docs()
    stopwords = get_stopwords()
    import jieba
    from tqdm import tqdm
    res = []
    for doc in tqdm(docs, ncols=100):
        doc_jieba = jieba.lcut(doc, cut_all=True)
        doc_stop = [i for i in doc_jieba if (i not in stopwords
                                             and i != ''
                                             and i != '\n'
                                             and i != '\t')]
        res.append(" ".join(doc_stop))
    with open('../data/segment.txt', 'r') as fp:
        fp.write("\n".join(res))


if __name__ == '__main__':
    # get_all_docs()
    get_seg_txt()
