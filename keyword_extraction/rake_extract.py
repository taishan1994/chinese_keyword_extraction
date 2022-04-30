import sys
sys.path.append('..')
import random

import jieba
import jieba.posseg as pseg
import operator
import json
from collections import Counter
from keyword_extraction.utils import get_test_docs, get_stopwords


"""
代码来源：https://blog.csdn.net/qq_25439417/article/details/82995374
"""
# Data structure for holding data
class Word():
    def __init__(self, char, freq=0, deg=0):
        self.freq = freq
        self.deg = deg
        self.char = char

    def returnScore(self):
        return self.deg / self.freq

    def updateOccur(self, phraseLength):
        self.freq += 1
        self.deg += phraseLength

    def getChar(self):
        return self.char

    def updateFreq(self):
        self.freq += 1

    def getFreq(self):
        return self.freq


# Check if contains num
def notNumStr(instr):
    for item in instr:
        if '\u0041' <= item <= '\u005a' or ('\u0061' <= item <= '\u007a') or item.isdigit():
            return False
    return True


# Read Target Case if Json
def readSingleTestCases(testFile):
    with open(testFile) as json_data:
        try:
            testData = json.load(json_data)
        except:
            # This try block deals with incorrect json format that has ' instead of "
            data = json_data.read().replace("'", '"')
            try:
                testData = json.loads(data)
                # This try block deals with empty transcript file
            except:
                return ""
    returnString = ""
    for item in testData:
        try:
            returnString += item['text']
        except:
            returnString += item['statement']
    return returnString


def extract(rawText, topk=10):
    # Construct Stopword Lib
    swLibList = [line.rstrip('\n') for line in open(r"../data/stopwords.txt", 'r', encoding='utf-8')]
    # Construct Phrase Deliminator Lib
    # conjLibList = [line.rstrip('\n') for line in open(r"spw.txt", 'r', encoding='utf-8')]
    conjLibList = []
    # Cut Text
    rawtextList = pseg.cut(rawText)

    # Construct List of Phrases and Preliminary textList
    textList = []
    listofSingleWord = dict()
    lastWord = ''
    poSPrty = ['m', 'x', 'uj', 'ul', 'mq', 'u', 'v', 'f']
    meaningfulCount = 0
    checklist = []
    for eachWord, flag in rawtextList:
        checklist.append([eachWord, flag])
        if eachWord in conjLibList or not notNumStr(
                eachWord) or eachWord in swLibList or flag in poSPrty or eachWord == '\n':
            if lastWord != '|':
                textList.append("|")
                lastWord = "|"
        elif eachWord not in swLibList and eachWord != '\n':
            textList.append(eachWord)
            meaningfulCount += 1
            if eachWord not in listofSingleWord:
                listofSingleWord[eachWord] = Word(eachWord)
            lastWord = ''

    # Construct List of list that has phrases as wrds
    newList = []
    tempList = []
    for everyWord in textList:
        if everyWord != '|':
            tempList.append(everyWord)
        else:
            newList.append(tempList)
            tempList = []

    tempStr = ''
    for everyWord in textList:
        if everyWord != '|':
            tempStr += everyWord + '|'
        else:
            if tempStr[:-1] not in listofSingleWord:
                listofSingleWord[tempStr[:-1]] = Word(tempStr[:-1])
                tempStr = ''

    # Update the entire List
    for everyPhrase in newList:
        res = ''
        for everyWord in everyPhrase:
            listofSingleWord[everyWord].updateOccur(len(everyPhrase))
            res += everyWord + '|'
        phraseKey = res[:-1]
        if phraseKey not in listofSingleWord:
            listofSingleWord[phraseKey] = Word(phraseKey)
        else:
            listofSingleWord[phraseKey].updateFreq()

    # Get score for entire Set
    outputList = dict()
    for everyPhrase in newList:

        if len(everyPhrase) > 5:
            continue
        score = 0
        phraseString = ''
        outStr = ''
        for everyWord in everyPhrase:
            score += listofSingleWord[everyWord].returnScore()
            phraseString += everyWord + '|'
            outStr += everyWord
        phraseKey = phraseString[:-1]
        freq = listofSingleWord[phraseKey].getFreq()
        if freq / meaningfulCount < 0.01 and freq < 3:
            continue
        outputList[outStr] = score

    sorted_list = sorted(outputList.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_list[:topk]


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
    result = extract(text)
    print(result)
