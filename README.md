# QQ语义相似度匹配模型

## 数据

接口机100.88.66.85

数据所在文件路径：/ceph/qbkg2/aitingliu/qq/src/data

数据文件简介：

| file                                | 备注                                                   | 格式                     |
| ----------------------------------- | ------------------------------------------------------ | ------------------------ |
| data/qq_simscore/merge_20190508.txt | 分词后的标注数据                                       | query\tquestion\tlabel\n |
| data/qq_simscore/train.txt          | 对merge_20190508.txt文件进行切分得到的训练集（120135） | query\tquestion\tlabel\n |
| data/qq_simscore/dev.txt            | 验证集（10000）                                        | query\tquestion\tlabel\n |
| data/qq_simscore/test.txt           | 测试集（10000）                                        | query\tquestion\tlabel\n |
| data/qq_simscore/word.txt           | word level 词典                                        | word\n                   |
| data/qq_simscore/char.txt           | char level 词典                                        | char\n                   |



## 模型

APLSTM/APCNN ：[Attentive Pooling Networks](https://arxiv.org/pdf/1602.03609.pdf)

ARCII : [Convolutional Neural Network Architectures for Matching Natural Language Sentences](http://papers.nips.cc/paper/5550-convolutional-neural-network-architectures-for-matching-natural-language-sentences.pdf)

Siamese LSTM/CNN : [Semi-supervised Clustering for Short Text via Deep Representation Learning](https://arxiv.org/pdf/1602.06797.pdf)

BiMPM : [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/pdf/1702.03814.pdf)

FastText :  [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

TextCNN : [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)

