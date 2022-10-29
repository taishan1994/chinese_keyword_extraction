"""
train_docs_keywords.txt里面只有一千条记录，也就是说只有一千条数据是有关键词的，
因此我们先要取出这一千条数据
"""
import json


def get_test_data():
    with open('all_docs.txt', 'r', encoding='utf-8') as fp:
        docs = fp.read().strip().split('\n')
    docs = {i.split('\x01')[0]: {"title": i.split('\x01')[1], "content": i.split('\x01')[2]} for i in docs}
    test_data = []
    with open('train_docs_keywords.txt', 'r', encoding='utf-8') as fp:
        labels = fp.read().strip().split('\n')
    for label in labels:
        label = label.strip()
        label = label.split('\t')
        data = docs.get(label[0])
        tmp = {}
        tmp['titile'] = data['title']
        tmp['content'] = data['content']
        tmp['keyword'] = label[1]
        test_data.append(tmp)

    with open('test.json', 'w', encoding='utf-8') as fp:
        json.dump(test_data, fp, ensure_ascii=False)


if __name__ == '__main__':
    get_test_data()
