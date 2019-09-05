from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, Flatten, RepeatVector, concatenate, CuDNNLSTM, LSTM, Lambda, K, \
    multiply, Dense, Activation, add

from custom_layers import Attention


class AbsaModel(object):
    def __init__(self, config):
        self.max_len = config.max_len
        self.max_content_vocab_size = config.max_content_vocab_size
        self.content_embed_dim = config.content_embed_dim
        self.max_aspect_vocab_size = config.max_aspect_vocab_size
        self.aspect_embed_dim = config.aspect_embed_dim
        self.lstm_units = config.lstm_units
        self.is_cudnn = config.is_cudnn
        self.dense_units = config.dense_units
        self.n_classes = config.n_classes
    #####构造atae——lstm模型
    def atae_lstm(self):
        input_content = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,))
        ###先将每个字进行embed，然后将aspect进行embed，然后根据content中字的个数重复，然后重复后的每个aspect embedding跟每个字的进行串联
        content_embed = Embedding(input_dim=self.max_content_vocab_size, output_dim=self.content_embed_dim)
        aspect_embed = Embedding(input_dim=self.max_aspect_vocab_size, output_dim=self.aspect_embed_dim)
        content_embedding = content_embed(input_content)
        content_embedding = SpatialDropout1D(0.2)(content_embedding)
        aspect_embeddibg = aspect_embed(input_aspect)
        aspect_flatten = Flatten()(aspect_embeddibg)
        repeat_aspect_embedding = RepeatVector(self.max_len)(aspect_flatten)
        ##将重复后的aspect和content字进行串联
        input_concat = concatenate([content_embedding, repeat_aspect_embedding], axis=-1)
        ##再加个LSTM
        if(self.is_cudnn):
            hidden_vecs, state_h, _ = CuDNNLSTM(self.lstm_units, return_sequences=True, return_state=True)(input_concat)
        else:
            hidden_vecs, state_h, _ = LSTM(self.lstm_units, return_sequences=True, return_state=True)(input_concat)
        concat = concatenate([hidden_vecs, repeat_aspect_embedding], axis=-1)

        # apply attention mechanism
        attend_weight = Attention()(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        attend_hidden_dense = Dense(self.lstm_units)(attend_hidden)
        last_hidden_dense = Dense(self.lstm_units)(state_h)
        final_output = Activation('tanh')(add([attend_hidden_dense, last_hidden_dense]))
        dense_layer = Dense(self.dense_units, activation='relu')(final_output)
        output_layer = Dense(self.n_classes, activation='softmax')(dense_layer)
        return Model([input_content, input_aspect], output_layer)
