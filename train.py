#!/usr/bin/python
# -*- coding: UTF-8 -*

from keras.models import Model
from keras.layers import Input, LSTM, Dense,concatenate,Embedding,Bidirectional,TimeDistributed
from keras import callbacks
import numpy as np
import json
from keras_multi_head import MultiHeadAttention
from keras.utils.np_utils import to_categorical
import keras.preprocessing.sequence
import math
import random
def encode(y):
	s=bin(y).replace('0b','').zfill(1)#原字符串右对齐，前面填充0
	result=[ int(i) for i in s]
	return result

def read_data(path):
	with open(path,'r',encoding='utf-8') as f:
		lines =f.readlines()
	lines=[eval(line.strip()) for line in lines]
	_,x1_data,x2_data,y_data=zip(*lines)
	x1_data=np.array(x1_data)
	x2_data=np.array(x2_data)
	y_data=np.array(y_data)
	return  x1_data,x2_data,y_data

for max_length in [2,3]:
	with open("train_data/hanzi_index_{}".format(max_length),'r',encoding='utf-8') as f:
		hanzi_dict=json.load(f)
	hanzi_num=len(hanzi_dict)
	x1_data,x2_data,y_data=read_data("train_data/train_data_{}".format(max_length))

	input1 = Input(shape=(max_length,))#有些字容易成詞   有些字不容易成詞    還需要字本身信息
	input2 = Input(shape=(2*max_length,))
	hanzi_emb=Embedding(hanzi_num,32, input_length=max_length)(input1)#輸出維度32
	hanzi_rep= Bidirectional(LSTM(5,return_sequences=False))(hanzi_emb)#雙向LSTM

	stat_feature=Dense(10,activation='relu')(input2)
	out=keras.layers.concatenate([stat_feature,hanzi_rep],axis=1)
	out=Dense(10,activation='relu')(out)
	out=Dense(1,activation='sigmoid')(out)#2分類
	model = Model([input1,input2],out)
	model.compile(optimizer='adam', loss='binary_crossentropy')
	model.fit([x1_data,x2_data],y_data,batch_size=128,epochs =20,shuffle=True,class_weight='auto')
	model.save('model/model_{}.h5'.format(max_length))



