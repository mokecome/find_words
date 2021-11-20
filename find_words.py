#!/usr/bin/python
# -*- coding: UTF-8 -*
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json
import numpy as np
from keras.models import load_model  

def read_data(path):
	with open(path) as f:
		lines =f.readlines()
	lines=[eval(line.strip()) for line in lines]
	lines=[line for line in lines if line[3]==0]
	words,x1_data,x2_data,y_data=zip(*lines)
	x1_data=np.array(x1_data)
	x2_data=np.array(x2_data)
	y_data=np.array(y_data)
	return  words,x1_data,x2_data,y_data

for max_length in [2,3]:
	with open("train_data/hanzi_index_{}".format(max_length)) as f:
		hanzi_dict=json.load(f)
	words,x1_data,x2_data,_=read_data("train_data/train_data_{}".format(max_length))
	model = load_model("model/model_{}.h5".format(max_length))
	scores=model.predict([x1_data,x2_data])
	results=zip(words,scores)
	results=sorted(results,key=lambda s:s[1],reverse=True)
	results=["{}\t{}".format(s[0],s[1]) for s in results]
	with open("results/{}".format(max_length),"w") as f:
		f.writelines("\n".join(results))




