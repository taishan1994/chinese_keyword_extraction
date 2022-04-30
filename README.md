# chinese_keyword_extraction
中文关键词提取<br>

数据下载：<br>

链接：https://pan.baidu.com/s/1Chpv_N1I1nK2_h37x1a3sA?pwd=st02 <br>
提取码：st02<br>

# 目录结构

```
--data：数据存储相关

----all_docs.txt：文本数据，以\x01分隔，分别为id，title，content

----process.py：数据处理生成test.json

----segment.txt：文本数据分词后的结果

----stopwords.txt：停止词

----test.json：带关键词的文本，有一千条

----train_docx_keywords.txt：不带关键词的文本10万+

--keyword_extraction：主代码

----jieba_textrank.py：基于jieba+textrank的提取

----jieba_tf_idf.py：基于Jieba+tf_idf的提取

----jionlp_extract.py：基于jionlp的提取

----pke_extract.py：基于pke的提取，包含textrank、positionrank、topicrank、multipartiterank、tf_idf、yake、kpminer、firstrstphrases

----pyhanlp_extract：基于pyhanlp提取，包含关键词提取、关键短语提取、关键句提取（文本摘要）

----sklearn_tf_idf：基于sklearn+tf_idf的关键词提取

----rake_extract.py：基于rake的提取

----seg_utils.py：基于多进程+tqdm显示的分词

----utils.py：一些辅助函数
```

# 运行

进入到keyword_extraction下面，不同的提取方式下都有可运行的例子。相关的依赖也有说明，

# 最后

参考了一些github上面的代码以及博客，在相关文件中都有说明。<br>

欢迎补充其它的一些例子。
