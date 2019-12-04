import re
import gc
import os
import json
import jieba
import joblib
import random
import psutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from collections import defaultdict



with open("../cna_data/train_pub.json")as f:
    public = json.load(f)
with open("../cna_data/train_author.json")as f:
    Author = json.load(f)


negNum = 1

# 选取同名人数大于指定值的作者作为训练集
paperAndAuthor = []
authorPaper = {}
for name in Author:
    if len(Author[name].keys()) > negNum:
        for person in Author[name]:
            authorPaper[person] = Author[name][person]
            for paper in Author[name][person]:
                temp = paper + '-' + person + '-' + name
                paperAndAuthor.append(temp)
print(len(paperAndAuthor))


zuozhe = set()
for artical in public:
    for author in public[artical]["authors"]:
        temp = author["name"].encode('utf-8')
        item = temp.decode('utf-8')
        if item not in zuozhe:
            zuozhe.add(item)


def cleanName(dirtyName):
    
    name = dirtyName.lower()
    name = name.replace('\xa0', ' ')
    name = name.replace('.', ' ')
    name = name.replace('dr.', '')
    name = name.replace('dr ', '')
    name = name.replace(' 0001', '')
    temp = name.split(' ')
    if len(temp) == 2:
        if '-' in temp[1] and '-' not in temp[0]:
            a = temp[1]
            temp[1] = temp[0]
            temp[0] = a
    k = []
    for t in temp:
        if t != '' and t != ' ':
            k.append(t)
    name = '_'.join(k)
    name = name.replace('-', '')
    return name

def getSample(item):
    sample = []
    paper, author, name = item.split('-')
    sample.append('-'.join([paper, author, name, '1']))

    candidateList = list(Author[name].keys())
    candidateList.remove(author)
    negPeople = random.sample(candidateList, negNum)
    for negPerson in negPeople:
        sample.append('-'.join([paper, negPerson, name, '0']))
    return sample
        
def entropy(dic):
    count = []
    total = 0
    ans = 0
    for item in dic:
        count.append(dic[item])
        total += dic[item]
    prob = [item/total for item in count]
    for item in prob:
        ans += -1 * item * round(np.log(item), 6)
    return ans

def cleanYear(dic):
    count = []
    largest = 0
    smallest = 3000
    err = 0
    ans = defaultdict(int)
    for item in dic:
        if item == '' or item > 2019 or item < 1950 :
            err += dic[item]
        else:
            count.append(dic[item])
            ans[item] = dic[item]
            if item > largest:
                largest = item
            if item < smallest:
                smallest = item
    mean = np.mean(count)
    if smallest < largest:
        add = np.random.randint(smallest, largest, err)
    else:
        add = [smallest] * err
    for data in add:
        ans[data] += 1
    count = []
    for item in ans:
        count.append(ans[item])
    mean = np.mean(count)
    for item in ans:
        ans[item] = 1/(1 + np.abs(mean - ans[item]))
    
    return ans

def getFeature(item):
    stopWord = ['', 'a', 'an', ':', 'the', '.', ',', '\n', '(', ')', ]
    paperId, authorId, _, label = item.split('-')
    
    paperInfo = public[paperId]
    
    title = paperInfo["title"]
    authors = paperInfo["authors"]
    venue = paperInfo["venue"]
    year = paperInfo.get("year", 0)
    keyword = paperInfo.get("keywords", "")
    if keyword == None:
        keyword = [""]
    abstract = paperInfo.get("abstract", "")
    if abstract == None:
        abstract = ""
    #这里不对待查询作者进行删除
    #对待查询论文进行预处理，获取其相关特征
    #特征包括titleLength、words、coauthors、coauthorsLength、coauthorsDiffLength、
    #        orgs、orgsDiffLength、venu、year、keywords、abstractLength
    #题目长度、题目词频
    titles = title.split()
    titleLength = 0
    words = defaultdict(int)
    for item in titles:
        if item.lower() not in stopWord:
            titleLength += 1
            words[item.lower()] += 1
            
    #合作者个数、频率
    coauthors = defaultdict(int)
    coauthorsLength = 0
    for item in authors:
        if item["name"] != '':
            coauthors[cleanName(item["name"])] += 1
            coauthorsLength += 1
    coauthorsDiffLength = len(coauthors)
    
    #合作机构频率
    orgs = defaultdict(int)
    for item in authors:
        orgs[item.get("org", "").lower()] += 1
    orgsDiffLength = len(orgs)
            
    #发表年份可能不存在
            
    #关键词词频
    keywords = defaultdict(int)
    for item in keyword:
        if item != '':
            keywords[item.lower()] += 1
    
    #摘要长度、词频
    abstractLength = 0
    abstracts = abstract.split()
    for item in abstracts:
        if item not in stopWord:
            abstractLength += 1
            words[item.lower()] += 1
    
    
    relatedTitleLength = 0               #题目平均长度 1
    relatedCoauthorLength = 0            #合作者数量1
    relatedCoauthorDiffLength = 0        #合作者人数1
    relatedOrgsDiffLength = 0            #合作机构个数1
    relatedAbstractLength = 0            #摘要长度
    relatedCoauthors = defaultdict(int)  #合作者1
    relatedOrgs = defaultdict(int)       #合作机构1
    relatedVenue = defaultdict(int)      #发表期刊1
    relatedYear = defaultdict(int)       #发表年份1
    relatedKeyword = defaultdict(int)    #关键词1
    relatedWords = defaultdict(int)      #题目、摘要出现词
    relatedOrgEntropy = 0
    
    relatedPapers = authorPaper[authorId]
    #去除待分类论文
    if paperId in relatedPapers:
        relatedPapers.remove(paperId)
    for relatedPaper in relatedPapers:
        relatedPaperInfo = public[relatedPaper]
        
        tempAuthors = relatedPaperInfo["authors"]
        tempOrgs = defaultdict(int)
        for tempName in tempAuthors:
            #机构、合作者频率
            #合作者数量
            relatedOrgs[tempName.get("org", "")] += 1
            relatedCoauthors[cleanName(tempName["name"])] += 1
            tempOrgs[tempName.get("org", "")] += 1
            relatedCoauthorLength += 1
        #机构平均散度
        relatedOrgEntropy += entropy(tempOrgs)
        #年份
        relatedYear[relatedPaperInfo.get("year", 0)] += 1
        #期刊
        relatedVenue[relatedPaperInfo.get("venue", "none")] += 1
        #关键词
        for item in relatedPaperInfo.get("keywords", ""):
            if item != '':
                relatedKeyword[item.lower()] += 1
        #题目长度
        tempTitle = relatedPaperInfo["title"].split()
        for item in tempTitle:
            if item not in stopWord:
                relatedTitleLength += 1
                relatedWords[item.lower()] += 1
        #摘要
        tempAbstracts = relatedPaperInfo.get("abstract", " ")
        if tempAbstracts == None:
            tempAbstracts = ""
        tempAbstracts = tempAbstracts.split()
        
        for item in tempAbstracts:
            if item not in stopWord:
                relatedAbstractLength += 1
                relatedWords[item.lower()] += 1
        
    relatedTitleLengthAve = relatedTitleLength/(len(relatedPapers) + 1)
    relatedCoauthorLengthAve = relatedCoauthorLength/(len(relatedPapers) + 1)
    relatedAbstractLengthAve = relatedAbstractLength/(len(relatedPapers) + 1)
    relatedCoauthorDiffLength = len(relatedCoauthors)
    relatedOrgsDiffLength = len(relatedOrgs)
    relatedOrgsDiffLengthAve = len(relatedOrgs)/(len(relatedPapers) + 1)
    relatedOrgEntropyAve = relatedOrgEntropy/(len(relatedPapers) + 1)
    #根据待查询论文与待查询作者的相关信息生成特征
    #1、题目长度 待查询论文题目长度/待查询作者论文题目平均长度
    scoreTitle = titleLength/(relatedTitleLengthAve + 1)
    
    #2、合作者相关
    # -1重合的合作者数
    # -2待查询论文的合作者数/待查询作者的平均合作者数(relatedCoauthorLengthAve)
    # -3重合的合作者数/待查询论文的合作者数(coauthorsLength)
    # -4重合的合作者数/待查询作者的合作者数(relatedCoauthorDiffLength)
    # -5重合的合作者的总合作次数
    # -6重合的合作者的总合作次数/所有合作者的总合作次数
    
    relatedCoauthorsSet = set(relatedCoauthors.keys())
    coauthorsSet = set(coauthors.keys())
    scoreAuthor1 = len(relatedCoauthorsSet & coauthorsSet)
    scoreAuthor2 = coauthorsLength /(relatedCoauthorLengthAve + 1)
    scoreAuthor3 = scoreAuthor1 /(coauthorsLength + 1)
    scoreAuthor4 = scoreAuthor1 /(relatedCoauthorDiffLength + 1)
    scoreAuthor5 = 0
    for item in coauthorsSet:
        scoreAuthor5 += relatedCoauthors[item]
    scoreAuthor6 = scoreAuthor5/(relatedCoauthorLength + 1)
    
    #3、期刊相关
    # -1 重合期刊的发表次数
    # -2 重合期刊的发表次数/总发表期刊次数
    
    scoreVenue1 = relatedVenue.get(venue, 0)
    temp = 0
    for item in relatedVenue:
        temp += relatedVenue[item]
    scoreVenue2 = scoreVenue1/(temp + 1)
    
    #4、合作机构相关
    # -1 待查询论文合作机构(orgsDiffLength)
    # -2 待查询论文合作机构/待查询作者平均合作机构
    # -3 待查询论文合作机构总合作次数
    # -4 待查询论文合作机构总合作次数/待查询作者所有合作机构总合作次数
    # -5 待查询论文合作机构散度
    # -6 待查询论文合作机构散度/待查询作者所有合作机构平均散度
    scoreOrg1 = orgsDiffLength
    scoreOrg2 = orgsDiffLength/(relatedOrgsDiffLengthAve + 1)
    scoreOrg3 = 0
    for item in orgs:
        scoreOrg3 += relatedOrgs[item]
    temp = 0
    for item in relatedOrgs:
        temp += relatedOrgs[item]
    scoreOrg4 = scoreOrg3/(temp + 1)
    scoreOrg5 = entropy(orgs)
    scoreOrg6 = scoreOrg5/(relatedOrgEntropyAve + 1)
    
    #5、发表时间
    #我们认为同一作者不同年份发表数量均匀
    #则对偏离平均发表数量的年份做惩罚
    
    if year == 0:
        scoreYear = 0
    else:
        scoreYear = cleanYear(relatedYear)[year]
        
    #6、关键词相关
    # -1 重合关键词总次数
    # -2 总关键词次数
    
    originKeywords = set(keywords.keys())
    relatedKeywords = set(relatedKeyword.keys())
    scoreKeyword1 = len(originKeywords & relatedKeywords)
    scoreKeyword2 = 0
    for item in relatedKeyword:
        scoreKeyword2 += relatedKeyword[item]
    
    
    #7、摘要相关
    # -1 待查询论文摘要长度/平均摘要长度
    # -2 待查询论文摘要关键词重合个数
    # -3 关键词重合总次数
    # -4 关键词重合总次数/总关键词出现次数
    
    scoreAbstract1 = abstractLength/(relatedAbstractLengthAve + 1)
    originWord = set(words.keys())
    relatedWord = set(relatedWords.keys())
    scoreAbstract2 = len(originWord & relatedWord)
    scoreAbstract3 = 0
    for item in words:
        scoreAbstract3 += relatedWords.get(item, 0)
    temp = 0
    for item in relatedWords:
        temp += relatedWords[item]
    scoreAbstract4 = scoreAbstract3/(temp + 1)
    
    # 所有特征
    # 1、scoreTitle
    # 2、scoreAuthor1
    # 3、scoreAuthor2 
    # 4、scoreAuthor3
    # 5、scoreAuthor4
    # 6、scoreAuthor5
    # 7、scoreAuthor6
    # 8、scoreVenue1
    # 9、scoreVenue2 
    # 10、scoreOrg1
    # 11、scoreOrg2
    # 12、scoreOrg3 
    # 13、scoreOrg4
    # 14、scoreOrg5
    # 15、scoreOrg6 
    # 16、scoreYear
    # 17、scoreKeyword1
    # 18、scoreKeyword2 
    # 19、scoreAbstract1
    # 10、scoreAbstract2
    # 21、scoreAbstract3 
    # 21、scoreAbstract4
    
    feature = [round(scoreTitle, 6),
               round(scoreAuthor1, 6),
               round(scoreAuthor2, 6),
               round(scoreAuthor3, 6),
               round(scoreAuthor4, 6),
               round(scoreAuthor5, 6),
               round(scoreAuthor6, 6),
               round(scoreVenue1, 6),
               round(scoreVenue2, 6),
               round(scoreOrg1, 6),
               round(scoreOrg2, 6),
               round(scoreOrg3, 6),
               round(scoreOrg4, 6),
               round(scoreOrg5, 6),
               round(scoreOrg6, 6),
               round(scoreYear, 6),
               round(scoreKeyword1, 6),
               round(scoreKeyword2, 6),
               round(scoreAbstract1, 6),
               round(scoreAbstract2, 6),
               round(scoreAbstract3, 6),
               round(scoreAbstract4, 6)
              ]
    return [feature, label]   

pickleText = []
trainSet = paperAndAuthor
posAndNegSample = []
random.shuffle(trainSet)
for item in trainSet:
    posAndNegSample.extend(getSample(item))

result = []
for item in tqdm(posAndNegSample[:1000]):
    result.append(getFeature(item))
pickleText.extend(result)

print(len(pickleText))
print(len(paperAndAuthor))

joblib.dump(pickleText, "./samples.pkl")

samples = joblib.load("./samples.pkl")

joblib.dump(paperAndAuthor, "./paperAndAuthor.pkl")

sample = []
for item in tqdm(samples):
    f, l = item
    l = int(l)
    f.append(l)
    sample.append(f)

test = np.array(sample)
random.shuffle(test)
features = DataFrame(test, columns=["Title", "Author1", "Author2", "Author3",
                                   "Author4", "Author5", "Author6", "Venue1",
                                   "Venue2", "Org1", "Org2", "Org3",
                                   "Org4", "Org5", "Org6", "Year",
                                   "Keyword1", "Keyword2", "Abstract1", "Abstract2",
                                   "Abstract3", "Abstract4",
                                   "label",
                                   ])

tempFeature = (features - features.mean())/features.std()

labelSample = features["label"]
col = [item for item in features.columns if item != "label"]
featureSample = features[col]
tempFeature = (featureSample - featureSample.mean())/featureSample.std()
joblib.dump(tempFeature, "./standardSample.pkl")
joblib.dump(labelSample, "./labelSample.pkl")

joblib.dump(np.array(labelSample), "./labelarray.pkl")
joblib.dump(np.array(tempFeature), "./featurearray.pkl")
joblib.dump(np.array(features.mean().tolist()[:-1]), "./mean.pkl")
joblib.dump(np.array(features.std().tolist()[:-1]), "./std.pkl")
