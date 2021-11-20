#遍歷語料 取得全局特徵2*n維
python get_data.py
#建構正(遍歷出來的在已有詞表label設1)負樣本(遍歷出來的隨機抽)
python get_train_data.py
#模型訓練
python train.py
#將所有2字搭配的字符串進行預測 得分高的當作新詞
python find_words.py