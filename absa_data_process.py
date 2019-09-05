import pickle
import numpy as np
import pandas as pd
from keras import Input
from keras.layers import Embedding

from absa_config import Config


def list_flatten(l):
    result = list()
    for item in l:
        if isinstance(item, (list, tuple)):
            result.extend(item)
        else:
            result.append(item)
    return result


def build_vocabulary(corpus, start_id=1):
    corpus = list_flatten(corpus)
    return dict((word, idx) for idx, word in enumerate(set(corpus), start=start_id))

train_file = pd.read_csv("/Users/duty/downloads/zhijiang_race/TRAIN/Train_reviews.csv")
label_file = pd.read_csv("/Users/duty/downloads/zhijiang_race/TRAIN/Train_labels.csv")
new_data = pd.merge(train_file, label_file, on='id')
filter_data = pd.DataFrame(new_data, columns=['id', 'Reviews', 'Categories', 'Polarities'])
drop_duplicate_data = filter_data.drop_duplicates()
pd.DataFrame.to_csv(drop_duplicate_data.loc[0:4000], "absa_train_data.csv", header=True, index=False)
pd.DataFrame.to_csv(drop_duplicate_data.loc[4000:5000], "absa_valid_data.csv", header=True, index=False)
pd.DataFrame.to_csv(drop_duplicate_data.loc[5000:], "absa_test_data.csv", header=True, index=False)

##处理原始数据分成训练数据和测试数据，将content的字转换为id，将aspect的词组转换为id
##训练集
train_data = pd.read_csv('absa_train_data.csv', header=0, index_col=None)
train_data['content_char_list'] = train_data['Reviews'].apply(lambda x: list(x))##每个char取出来
train_data['aspect_char_list'] = train_data['Categories'].apply(lambda x: list(x))
train_data['aspect_term_list'] = train_data['Categories'].values
##验证集
valid_data = pd.read_csv('absa_valid_data.csv', header=0, index_col=None)
valid_data['content_char_list'] = valid_data['Reviews'].apply(lambda x: list(x))##每个char取出来
valid_data['aspect_char_list'] = valid_data['Categories'].apply(lambda x: list(x))
valid_data['aspect_term_list'] = valid_data['Categories'].values
##测试集
test_data = pd.read_csv('absa_test_data.csv', header=0, index_col=None)
test_data['content_char_list'] = test_data['Reviews'].apply(lambda x: list(x))##每个char取出来
test_data['aspect_char_list'] = test_data['Categories'].apply(lambda x: list(x))
test_data['aspect_term_list'] = test_data['Categories'].values
###构造字和id映射 dic形式
total_chars = np.concatenate((train_data['content_char_list'].values, valid_data['content_char_list'].values, test_data['content_char_list'].values)).tolist()
content_char_corpus = build_vocabulary(total_chars, start_id=1)
total_aspect_chars = np.concatenate((train_data['aspect_char_list'].values, valid_data['aspect_char_list'].values, test_data['aspect_char_list'].values)).tolist()
aspect_char_corpus = build_vocabulary(total_aspect_chars, start_id=1)
total_aspect_terms = np.concatenate((train_data['aspect_term_list'].values, valid_data['aspect_term_list'].values, test_data['aspect_term_list'].values)).tolist()
aspect_term_corpus = build_vocabulary(total_aspect_terms, start_id=0)
pickle.dump(content_char_corpus, open("./absa_data/content_char_corpus.pkl", "wb"))
pickle.dump(aspect_char_corpus, open("./absa_data/aspect_char_corpus.pkl", "wb"))
pickle.dump(aspect_term_corpus, open("./absa_data/aspect_term_corpus.pkl", "wb"))

###根据



print("dd")
