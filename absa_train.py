import pickle

from keras.utils import plot_model, to_categorical
from keras_preprocessing.sequence import pad_sequences

from absa_config import Config
from absa_model import AbsaModel
import pandas as pd
config = Config()
absa_model = AbsaModel(config)
atae_model = absa_model.atae_lstm()
atae_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
plot_model(atae_model, to_file='absa_atae_model')
def category_process(x):
    if x=='正面':
        return 0
    elif x=='中性':
        return 1
    elif x=='负面':
        return 2
    return 1
##读取训练数据
train_data = pd.read_csv('absa_train_data.csv', header=0, index_col=None)
train_data['content_char_list'] = train_data['Reviews'].apply(lambda x: list(x))
char_vocab = pickle.load(open("./absa_data/content_char_corpus.pkl","rb"))
train_char_input = train_data['content_char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
aspect_term_vocab = pickle.load(open("./absa_data/aspect_term_corpus.pkl", "rb"))
train_aspect_input = train_data["Categories"].apply(lambda x: [aspect_term_vocab[x]]).values.tolist()
train_label = train_data['Polarities'].apply(category_process).values.tolist()
##读取验证集
valid_data = pd.read_csv('absa_valid_data.csv', header=0, index_col=None)
valid_data['content_char_list'] = valid_data['Reviews'].apply(lambda x: list(x))
# char_vocab = pickle.load("./absa_data/content_char_corpus.pkl", "rb")
valid_char_input = valid_data['content_char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
# aspect_term_vocab = pickle.load("./absa_data/aspect_term_corpus.pkl", "rb")
valid_aspect_input = valid_data["Categories"].apply(lambda x: [aspect_term_vocab[x]]).values.tolist()
valid_label = valid_data['Polarities'].apply(category_process).values.tolist()

##读取测试集
test_data = pd.read_csv('absa_test_data.csv', header=0, index_col=None)
test_data['content_char_list'] = test_data['Reviews'].apply(lambda x: list(x))
# char_vocab = pickle.load("./absa_data/content_char_corpus.pkl", "rb")
test_char_input = test_data['content_char_list'].apply(
        lambda x: [char_vocab.get(char, len(char_vocab)+1) for char in x]).values.tolist()
# aspect_term_vocab = pickle.load("./absa_data/aspect_term_corpus.pkl", "rb")
test_aspect_input = test_data["Categories"].apply(lambda x: [aspect_term_vocab[x]]).values.tolist()
test_label = test_data['Polarities'].apply(category_process).values.tolist()

##处理输入数据和label
x_train_content = pad_sequences(train_char_input, maxlen=config.max_len)
x_train_aspect = pad_sequences(train_aspect_input, 1)
y_train = to_categorical(train_label, config.n_classes)
x_valid_content = pad_sequences(valid_char_input, maxlen=config.max_len)
x_valid_aspect = pad_sequences(valid_aspect_input, 1)
y_valid = to_categorical(valid_label, config.n_classes)
print("开始训练。。。。。。。。。。。。。。")
atae_model.fit(x=[x_train_content, x_train_aspect], y=y_train, batch_size=32, epochs=50,
                validation_data=([x_valid_content, x_valid_aspect],y_valid))
atae_model.save("atae_model.h5")
# atae_model.predict()
print("训练结束。。。。。。。。。。。。。。。。。")
print("ddd")