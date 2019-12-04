# README
山东大学信息检索与数据挖掘大作业
##  BACKGROUND
在许多应用中，同名消歧 (Name Disambiguation - aiming at disambiguating WhoIsWho) 一直被视为一个具有挑战性的问题，如科学文献管- 人物搜- 社交网络分析等，同时，随着科学文献的大量增长，使得该问题的解决变得愈加困难与紧迫。尽管同名消歧已经在学术界和工业界被大量研究，但由于数据的杂乱以及同名情景十分复杂，导致该问题仍未能很好解决。

## INTRODUCTION

收录各种论文的线上学术搜索系统(例Google Scholar, Dblp和AMiner等)已经成为目前全球学术界重要且最受欢迎的学术交流以及论文搜索平台。然而由于论文分配算法的局限性，现有的学术系统内部存在着大量的论文分配错误；此外，每天都会有大量新论文进入系统。故如何准确快速的将论文分配到系统中已有作者档案以及维护作者档案的一致性，是现有的线上学术系统亟待解决的难题。

由于学术系统内部的数据十分巨大（AMiner大约有130，000，000作者档案，以及超过200，000，000篇论文），导致作者同名情景十分复杂，要快速且准确的解决同名消歧问题还是有很大的障碍。

竞赛希望提出一种解决问题的模型，可以根据论文的详细信息以及作者与论文之间的联系，去区分属于不同作者的同名论文，获得良好的论文消歧结果。而良好的消歧结果是确保学术系统中，专家知识搜索有效- 数字图书馆的高质量内容管理以及个性化学术服务的重要前提，也可影响到其他相关领域。

## DESCRIPTION
线上系统每天会新增大量的论文，如何准确快速的将论文分配到系统中已有作者档案，这是线上学术系统最亟待解决的问题。所以问题抽象定义为：给定一批新增论文以及系统已有的作者论文集，最终目的是把新增论文分配到正确的作者档案中。

---
## DATASET

> 训练集已经公开在https://www.aminer.cn/diambiguate-data
 
### 训练集（10.1比赛开始公布）

- train_author.json

> 数据格式：此文件中的数据组织成一个字典（dictionary, 记为dic1），存储为JSON对象。 dic1的键（key）是作者姓名。 dic1的值（value）是表示作者的字典（记为dic2）。 dic2的键（key）是作者ID， dic2的值（value）是该作者的论文ID列表。

- train_pub.json

> 此文件包含train_author.json所有论文的元数据，数据存储为JSON对象；

> 数据格式：此文件的数据表示为一个字典（dictionary），其键（key）是论文ID，其值是相应的论文信息。 每篇论文的数据格式如下：

![PIC1](https://github.com/lyj201002/OAG-WhoIsWho2/blob/master/Screenshot%20from%202019-12-04%2023-32-46.png)

### 已有用户档案（10.1号开始公布）

- whole_author_profile.json

> 二级字典，key值为作者id，value分为俩个域: ‘name’域代表作者名，’papers’域代表作者的所拥有的论文(作者的profile), 测试集与验证集使用同一个已有的作者档案；

- whole_author_profile_pub.json

> whole_author_profile.json中涉及的论文元信息，格式同train_pub.json;

### 验证集（10.1号开始公布）

- cna_valid_unass_competition.json

> 论文列表，代表待分配的论文list，列表中的元素为论文id + ‘-’ + 需要分配的作者index(从0开始)；参赛者需要将该文件中的每篇论文的待分配作者对应分配到已有作者档案中(whole_author_profile.json).

- valid_example_evaluation_continuous.json

> 示例提交文件。二级字典，key值为作者 ID，value 值代表分配到该作者的论文id（来自cna_valid_unass_competition.json）。

- cna_valid_pub.json

> cna_valid_unass_competition.json中所涉及的论文元信息，格式同train_pub.json。

### 测试集（11.30号发布）

- cna_test_unass_competition.json

> 论文列表，代表待分配的论文list，列表中的元素为论文id + ‘-’ + 需要分配的作者index(从0开始)；参赛者需要将该文件中的每篇论文的待分配作者对应分配到已有作者档案中(whole_author_profile.json).

- test_example_evaluation_continuous.json 

> 示例提交文件。二级字典，key值为作者 ID，value 值代表分配到该作者的论文id（来自cna_valid_unass_competition.json）。

- cna_test_pub.json

> cna_test_unass_competition.json中所涉及的论文元信息，格式同 train_pub.json。
---

## SOLUTION

- 二分类
- RandomForests

---
## Features

|               |                                                            |
|       ---     |    :----:                                                     |
| scoreTitle    |待查询论文题目长度/待查询作者论文题目平均长度                       |
| scoreAuthor1  |重合的合作者数                                                |
| scoreAuthor2  |待查询论文的合作者数/待查询作者的平均合作者数                       |
| scoreAuthor3  |重合的合作者数/待查询论文的合作者数                               |  
| scoreAuthor4  |重合的合作者数/待查询作者的合作者数                              |
| scoreAuthor5  |重合的合作者的总合作次数                                       |
| scoreAuthor6  |重合的合作者的总合作次数/所有合作者的总合作次数                    |
| scoreVenue1   |重合期刊的发表次数                                             |
| scoreVenue2   |重合期刊的发表次数/总发表期刊次数                                |
| scoreOrg1     |待查询论文合作机构                                             |
| scoreOrg2     |待查询论文合作机构/待查询作者平均合作机构                          |
| scoreOrg3     |待查询论文合作机构总合作次数                                     |
| scoreOrg4     |待查询论文合作机构总合作次数/待查询作者所有合作机构总合作次数          |
| scoreOrg5     |待查询论文合作机构散度                                          |
| scoreOrg6     |待查询论文合作机构散度/待查询作者所有合作机构平均散度                |
| scoreYear     |我们认为同一作者不同年份发表数量均匀则对偏离平均发表数量的年份做惩     |
| scoreKeyword1 |重合关键词总次数                                               |
| scoreKeyword2 |总关键词次数                                                  |
| scoreAbstract1|待查询论文摘要长度/平均摘要长度                                  |
| scoreAbstract2|待查询论文摘要关键词重合个数                                     |
| scoreAbstract3|关键词重合总次数                                               |
| scoreAbstract4|关键词重合总次数/总关键词出现次数                                |

## 特征重要度
![PIC2](https://github.com/lyj201002/OAG-WhoIsWho2/blob/master/download%20(1).png)
![PIC3](https://github.com/lyj201002/OAG-WhoIsWho2/blob/master/download%20(2).png)
![PIC4](https://github.com/lyj201002/OAG-WhoIsWho2/blob/master/download%20(3).png)
![PIC5](https://github.com/lyj201002/OAG-WhoIsWho2/blob/master/download%20(4).png)
![PIC6](https://github.com/lyj201002/OAG-WhoIsWho2/blob/master/download%20(5).png)
![PIC7](https://github.com/lyj201002/OAG-WhoIsWho2/blob/master/download%20.png)
