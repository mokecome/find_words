# -*- coding:utf-8 -*-
import json
import math
def add_dict(hanzi,hanzi_dict):
	if hanzi not in hanzi_dict:
		hanzi_dict[hanzi]=len(hanzi_dict)
	return hanzi_dict[hanzi]
with open("legal_words") as f:#已有詞表 
	lines=f.readlines()
legal_words=set([line.strip().split("\t")[0] for line in lines])
for l in [2,3]:
	count=0
	results=[]
	hanzi_dict={'unknow':0}
	with open("train_data/feature_{}".format(l),'r',encoding='utf-8') as f:
		lines=f.readlines()
	lines=[line.strip().split("\t") for line in lines]
	lines=[s for s in lines if len(s)==2] #s每次取兩個 s[0]詞 s[1]特徵
	for word,feature in lines:
		feature=eval(feature)
		index=[ add_dict(hanzi,hanzi_dict) for hanzi in word]
		if len(index)!=l:#字典裡字長度=1時為字才往下 #條件成立下一個循環
			continue 
		label=1 if word in legal_words else 0  #三元表达式#任意取--如果在詞表裡就label設1  正樣本  其他設0  負樣本
		if label==1:
			count+=1
		results.append(str([word,index,feature,label]))  
	with open("train_data/train_data_{}".format(l),"w",encoding='utf-8') as f:#[詞 字 特徵  label]   正負樣本
		f.writelines("\n".join(results))
	with open("train_data/hanzi_index_{}".format(l),"w",encoding='utf-8') as f:
		json.dump(hanzi_dict,f,ensure_ascii=False)
	print(l,count)
	




