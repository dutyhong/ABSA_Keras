# import os
# import pickle
#
# use_text_input = True
# def pickle_load(file_path):
#     return pickle.load(open(file_path, 'rb'))
#
# def pickle_dump(obj, file_path):
#     pickle.dump(obj, open(file_path, 'wb'))
# input_data =[]
# if use_text_input:
#     input_data.append(pickle_load(os.path.join("./data/laptop/term/train_char_input.pkl")))
# print("dd")

from keras.models import Input, Model
from keras.layers import Dense, Conv2D, TimeDistributed
# from keras.utils import plot_model
from keras.utils import plot_model

input_ = Input(shape=(12, 8))
out = Dense(units=10)(input_)
# out = Dense(units=10)(input_)
model = Model(inputs=input_, outputs=out)
model.summary()
plot_model(model, to_file='test.png', show_shapes=True)
