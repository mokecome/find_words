# -*- coding:utf-8 -*-  
import json
import math
import random
hanzi_left_num={}
hanzi_right_num={}
hanzi_count={}
def normal(word_num,word_count):
	results={}
	for word,num_list in word_num.items():
		count=word_count[word]
		results[word]=dict([[h,num/count] for h,num in num_list.items()])#取平均
	return results
def get_top_words(contents,word_length):
	contents=random.sample(contents,50*10000)#取句
	word_count={}
	for content in contents:#紀錄詞出現次數
		hanzi_list=["start"]+[s for s in content]+["end"]#['start','a','b','end']
		for i in range(1,len(hanzi_list)-word_length):
			word="".join(hanzi_list[i:i+word_length])#word_length一組  遍歷
			word_count[word]=word_count.get(word,0)+1.0
	word_count=word_count.items()
	word_count=sorted(word_count,key=lambda s:s[1],reverse=True)[0:500000]#由高到低排序
	return set([s[0] for s in word_count])#不重複的字  s[0]詞  s[1]次數

def extract_info(contents,word_length):
	count=0
	word_left_num={}
	word_right_num={}
	word_count={}
	if word_length>1:
		all_words=get_top_words(contents,word_length)
	for content in contents:
		if count%100000==0:  #進度條
			print(count,len(contents))
		count+=1
		hanzi_list=["start"]+[s for s in content]+["end"]
		for i in range(1,len(hanzi_list)-word_length):
			word="".join(hanzi_list[i:i+word_length])
			if word_length>1 and word not in all_words:
				continue
			l=hanzi_list[i-1]
			r=hanzi_list[i+word_length]
			if word not in word_count:
				word_left_num[word]={}#左邊有哪些 字:次數
				word_right_num[word]={}
			word_count[word]=word_count.get(word,0)+1.0
			word_left_num[word][l]=word_left_num[word].get(l,0)+1.0 
			word_right_num[word][r]=word_right_num[word].get(r,0)+1.0
	word_right_num=normal(word_right_num,word_count)
	word_left_num=normal(word_left_num,word_count)
	return word_right_num,word_left_num   
def cal_info(word_num,w1,w2):
	score=word_num[w1][w2]
	return -1*math.log(score)/len(word_num)#取log比較平滑/字數   內聚的分數
def extract_feature(hanzi_right_num,hanzi_left_num,word_right_entropy,word_left_entropy):
	all_words=word_right_num.keys()
	results=[]
	for word in all_words:
		right_entropy=word_right_entropy[word]
		left_entropy=word_left_entropy[word]
		right_inner=[ cal_info(hanzi_right_num,word[i],word[i+1]) for i in  range(0,len(word)-1)]  #概率計算
		left_inner=[ cal_info(hanzi_left_num,word[i],word[i-1]) for i in range(1,len(word))]
		feature=right_inner+left_inner+[right_entropy]+[left_entropy]
		results.append(word+"\t"+str(feature))
	return results
def cal_entropy(word_direction_num):#信息墒
	results={}
	for word,l in word_direction_num.items():
		score=sum([-1*math.log(s[1]) for s in l.items()])/len(l)#所有發生的
		results[word]=score
	return results
with open("data/weibo_train_data.txt","r",encoding='utf-8') as f:  #遍歷所有語料  統計全局特徵
	lines=f.readlines()
contents=[line.strip().split("\t")[-1] for line in lines]  #取最後一個
hanzi_right_num,hanzi_left_num=extract_info(contents,1)#詞的長度為1
for l in range(2,3):
	print(l)
	word_right_num,word_left_num=extract_info(contents,l)  #詞的長度(2-3)  左邊有哪些   右邊有哪些
	word_right_entropy=cal_entropy(word_right_num)
	word_left_entropy=cal_entropy(word_left_num)
	results=extract_feature(hanzi_right_num,hanzi_left_num,word_right_entropy,word_left_entropy)
	with open("train_data/feature_{}".format(l),"w",encoding='utf-8') as f:
		f.writelines("\n".join(results))





