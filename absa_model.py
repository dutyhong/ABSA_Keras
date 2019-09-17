from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, Flatten, RepeatVector, concatenate, CuDNNLSTM, LSTM, Lambda, K, \
    multiply, Dense, Activation, add, GRU, TimeDistributed, AveragePooling1D
import tensorflow as tf
from custom_layers import Attention, ContentAttention


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
        self.aspect_max_len = config.aspect_max_len
    #####构造atae——lstm模型
    def atae_lstm(self):
        input_content = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,))
        ###先将每个字进行embed，然后将aspect进行embed，然后根据content中字的个数重复，然后重复后的每个aspect embedding跟每个字的进行串联
        content_embed = Embedding(input_dim=self.max_content_vocab_size, output_dim=self.content_embed_dim)
        aspect_embed = Embedding(input_dim=self.max_aspect_vocab_size, output_dim=self.aspect_embed_dim)
        content_embedding = content_embed(input_content)
        content_embedding = SpatialDropout1D(0.2)(content_embedding)
        aspect_embedding = aspect_embed(input_aspect)
        aspect_flatten = Flatten()(aspect_embedding)
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
    def atae_lstm_new(self):
        input_content = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(self.aspect_max_len,))
        ###先将每个字进行embed，然后将aspect进行embed，然后根据content中字的个数重复，然后重复后的每个aspect embedding跟每个字的进行串联
        content_embed = Embedding(input_dim=self.max_content_vocab_size, output_dim=self.content_embed_dim)
        aspect_embed = Embedding(input_dim=self.max_content_vocab_size, output_dim=self.aspect_embed_dim)
        content_embedding = content_embed(input_content)
        content_embedding = SpatialDropout1D(0.2)(content_embedding)
        aspect_embedding = aspect_embed(input_aspect)
        ##对aspect的字符串向量进行一个pooling 60*128=>1*128
        aspect_embedding = AveragePooling1D(pool_size=self.aspect_max_len)(aspect_embedding)
        aspect_flatten = Flatten()(aspect_embedding)
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

    def cabasc(self):
        def sequence_mask(sequence):
            return K.sign(K.max(K.abs(sequence), 2))

        def sequence_length(sequence):
            return K.cast(K.sum(sequence_mask(sequence), 1), tf.int32)

        input_text = Input(shape=(self.max_len,))
        input_text_l = Input(shape=(self.max_len,))
        input_text_r = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,))
        input_mask = Input(shape=(self.max_len, ))

        word_embedding = Embedding(input_dim=self.max_content_vocab_size, output_dim=self.content_embed_dim)
        text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))
        text_l_embed = SpatialDropout1D(0.2)(word_embedding(input_text_l))
        text_r_embed = SpatialDropout1D(0.2)(word_embedding(input_text_r))

        asp_embedding = Embedding(input_dim=self.max_aspect_vocab_size,
                                  output_dim=self.aspect_embed_dim)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d

        # regarding aspect string as the first unit
        hidden_l = GRU(self.lstm_units, go_backwards=True, return_sequences=True)(text_l_embed)
        hidden_r = GRU(self.lstm_units, return_sequences=True)(text_r_embed)

        # left context attention
        context_attend_l = TimeDistributed(Dense(1, activation='sigmoid'))(hidden_l)
        # Note: I couldn't find `reverse_sequence` in keras
        context_attend_l = Lambda(lambda x: tf.reverse_sequence(x, sequence_length(x), 1, 0))(context_attend_l)
        context_attend_l = Lambda(lambda x: K.squeeze(x, -1))(context_attend_l)

        # right context attention
        context_attend_r = TimeDistributed(Dense(1, activation='sigmoid'))(hidden_r)
        context_attend_r = Lambda(lambda x: K.squeeze(x, -1))(context_attend_r)

        # combine context attention
        # aspect_text_embed = subtract([add([text_l_embed, text_r_embed]), text_embed])
        # aspect_text_mask = Lambda(lambda x: sequence_mask(x))(aspect_text_embed)
        # text_mask = Lambda(lambda x: sequence_mask(x))(text_embed)
        # context_mask = subtract([text_mask, aspect_text_mask])
        # aspect_text_mask_half = Lambda(lambda x: x*0.5)(aspect_text_mask)
        # combine_mask = add([context_mask, aspect_text_mask_half])  # 1 for context, 0.5 for aspect
        context_attend = multiply([add([context_attend_l, context_attend_r]), input_mask])

        # apply context attention
        context_attend_expand = Lambda(lambda x: K.expand_dims(x))(context_attend)
        memory = multiply([text_embed, context_attend_expand])

        # sentence-level content attention
        sentence = Lambda(lambda x: K.mean(x, axis=1))(memory)
        final_output = ContentAttention()([memory, aspect_embed, sentence])
        dense_layer = Dense(self.dense_units, activation='relu')(final_output)
        output_layer = Dense(self.n_classes, activation='softmax')(dense_layer)
        return Model([input_text, input_text_l, input_text_r, input_aspect, input_mask], output_layer)
