# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: models.py

@time: 2019/1/5 16:08

@desc:

"""

import os
import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, SpatialDropout1D, Dropout, Conv1D, MaxPool1D, Flatten, concatenate, Dense, \
    LSTM, Bidirectional, Activation, MaxPooling1D, Add, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, \
    TimeDistributed, Permute, multiply, Lambda, add, Masking, BatchNormalization, Softmax, Reshape, ReLU, \
    ZeroPadding1D, subtract, CuDNNLSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import keras.backend as K
import tensorflow as tf

from custom_layers import Attention, RecurrentAttention, InteractiveAttention, ContentAttention, ELMoEmbedding
from utils import get_score_senti
from data_loader import load_idx2token

# callback for sentiment analysis model
class SentiModelMetrics(Callback):
    def __init__(self):
        super(SentiModelMetrics, self).__init__()

    def on_train_begin(self, logs={}):
        self.val_accs = []
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        if len(self.validation_data[:-3]) == 1:
            x_valid = self.validation_data[0]
        else:
            x_valid = self.validation_data[:-3]
        y_valid = self.validation_data[-3]
        valid_results = self.model.predict(x_valid)
        _val_acc, _val_f1 = get_score_senti(y_valid, valid_results)
        logs['val_acc'] = _val_acc
        logs['val_f1'] = _val_f1
        self.val_accs.append(_val_acc)
        self.val_f1s.append(_val_f1)
        print('val_acc: %f' % _val_acc)
        print('val_f1: %f' % _val_f1)
        return


# model for sentiment analysis
class SentimentModel(object):
    def __init__(self, config):
        self.is_cudnn = False
        self.config = config
        self.level = self.config.level
        self.use_elmo = self.config.use_elmo
        self.max_len = self.config.max_len[self.config.data_name][self.level]
        self.left_max_len = self.config.left_max_len[self.config.data_name][self.level]
        self.right_max_len = self.config.right_max_len[self.config.data_name][self.level]
        self.asp_max_len = self.config.asp_max_len[self.config.data_name][self.level]

        if self.config.use_text_input or self.config.use_text_input_l or self.config.use_text_input_r or self.config.use_text_input_r_with_pad:
            self.text_embeddings = np.load('./data/%s/%s_%s.npy' % (self.config.data_folder, self.level,
                                                                    self.config.word_embed_type))
            self.config.idx2token = load_idx2token(self.config.data_folder, self.level)
        else:
            self.text_embeddings = None
        if self.config.use_aspect_input:
            self.aspect_embeddings = np.load('./data/%s/aspect_%s_%s.npy' % (self.config.data_folder, self.level,
                                                                             self.config.aspect_embed_type))
            if config.aspect_embed_type == 'random':
                self.n_aspect = self.aspect_embeddings.shape[0]
                self.aspect_embeddings = None
        else:
            self.aspect_embeddings = None
        if self.config.use_aspect_text_input:
            self.aspect_text_embeddings = np.load('./data/%s/aspect_text_%s_%s.npy' % (self.config.data_folder,
                                                                                       self.level,
                                                                                       self.config.word_embed_type))
            self.config.idx2aspect_token = load_idx2token(self.config.data_folder, 'aspect_text_{}'.format(self.level))
        else:
            self.aspect_text_embeddings = None

        self.callbacks = []
        self.init_callbacks()

        self.model = None
        self.build_model()

    def init_callbacks(self):
        self.callbacks.append(SentiModelMetrics())

        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir, '%s/%s.hdf5' % (self.config.data_folder,
                                                                              self.config.exp_name)),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))

        # self.callbacks.append(EarlyStopping(
        #     monitor=self.config.early_stopping_monitor,
        #     mode=self.config.early_stopping_mode,
        #     patience=self.config.early_stopping_patience
        # ))

    def load(self):
        print('loading model checkpoint {} ...\n'.format('%s.hdf5') % self.config.exp_name)
        self.model.load_weights(os.path.join(self.config.checkpoint_dir, '%s/%s.hdf5' % (self.config.data_folder,
                                                                                         self.config.exp_name)))
        print('Model loaded')

    def build_base_network(self):
        if self.config.model_name == 'td_lstm':
            base_network = self.td_lstm()
        elif self.config.model_name == 'tc_lstm':
            base_network = self.tc_lstm()
        elif self.config.model_name == 'at_lstm':
            base_network = self.at_lstm()
        elif self.config.model_name == 'ae_lstm':
            base_network = self.ae_lstm()
        elif self.config.model_name == 'atae_lstm':
            base_network = self.atae_lstm()
        elif self.config.model_name == 'memnet':
            base_network = self.memnet()
        elif self.config.model_name == 'ram':
            base_network = self.ram()
        elif self.config.model_name == 'ian':
            base_network = self.ian()
        elif self.config.model_name == 'cabasc':
            base_network = self.cabasc()
        else:
            raise Exception('Model Name `%s` Not Understood' % self.config.model_name)

        return base_network

    def build_model(self):
        network_inputs = list()
        if self.config.use_text_input:
            network_inputs.append(Input(shape=(self.max_len,), name='input_text'))
        if self.config.use_text_input_l:
            if self.config.model_name == 'cabasc':
                network_inputs.append(Input(shape=(self.max_len,), name='input_text_l'))
            else:
                network_inputs.append(Input(shape=(self.left_max_len,), name='input_text_l'))
        if self.config.use_text_input_r:
            if self.config.model_name == 'cabasc':
                network_inputs.append(Input(shape=(self.max_len,), name='input_text_r'))
            else:
                network_inputs.append(Input(shape=(self.right_max_len,), name='input_text_r'))
        if self.config.use_text_input_r_with_pad:
            network_inputs.append(Input(shape=(self.max_len,), name='input_text_r_with_pad'))
        if self.config.use_aspect_input:
            network_inputs.append(Input(shape=(1, ), name='input_aspect'))
        if self.config.use_aspect_text_input:
            network_inputs.append(Input(shape=(self.asp_max_len,), name='input_aspect_text'))
        if self.config.use_loc_input:
            network_inputs.append(Input(shape=(self.max_len,), name='input_loc_info'))
        if self.config.use_offset_input:
            network_inputs.append(Input(shape=(self.max_len,), name='input_offset_info'))
        if self.config.use_mask:
            network_inputs.append(Input(shape=(self.max_len,), name='input_mask'))

        if len(network_inputs) == 1:
            network_inputs = network_inputs[0]
        elif len(network_inputs) == 0:
            raise Exception('No Input!')

        base_network = self.build_base_network()
        sentence_vec = base_network(network_inputs)
        dense_layer = Dense(self.config.dense_units, activation='relu')(sentence_vec)
        output_layer = Dense(self.config.n_classes, activation='softmax')(dense_layer)

        self.model = Model(network_inputs, output_layer)
        self.model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=self.config.optimizer)

    def prepare_input(self, input_data):
        if self.config.model_name == 'td_lstm':
            text_l, text_r = input_data
            input_pad = [pad_sequences(text_l, self.left_max_len), pad_sequences(text_r, self.right_max_len)]
        elif self.config.model_name == 'tc_lstm':
            text_l, text_r, aspect = input_data
            input_pad = [pad_sequences(text_l, self.left_max_len), pad_sequences(text_r, self.right_max_len),
                         np.array(aspect)]
        elif self.config.model_name in ['at_lstm', 'ae_lstm', 'atae_lstm'] or \
                (self.config.model_name in ['memnet', 'ram'] and not self.config.is_aspect_term):
            text, aspect = input_data
            input_pad = [pad_sequences(text, self.max_len), np.array(aspect)]
        elif self.config.model_name == 'memnet' and self.config.is_aspect_term:
            text, aspect, loc = input_data
            input_pad = [pad_sequences(text, self.max_len), np.array(aspect), pad_sequences(loc, self.max_len)]
        elif self.config.model_name == 'ram' and self.config.is_aspect_term:
            text, aspect, loc, offset = input_data
            input_pad = [pad_sequences(text, self.max_len), np.array(aspect), pad_sequences(loc, self.max_len),
                         pad_sequences(offset, self.max_len)]
        elif self.config.model_name == 'ian':
            text, aspect_text = input_data
            input_pad = [pad_sequences(text, self.max_len), pad_sequences(aspect_text, self.asp_max_len)]
        elif self.config.model_name == 'cabasc':
            text, text_l, text_r, aspect, mask = input_data
            input_pad = [pad_sequences(text, self.max_len, padding='post', truncating='post'),
                         pad_sequences(text_l, self.max_len, padding='post', truncating='post'),
                         pad_sequences(text_r, self.max_len, padding='post', truncating='post'),
                         np.array(aspect), pad_sequences(mask, self.max_len, padding='post', truncating='post')]
        else:
            raise ValueError('model name `{}` not understood'.format(self.config.model_name))
        return input_pad

    def prepare_label(self, label_data):
        return to_categorical(label_data, self.config.n_classes)

    def train(self, train_input_data, train_label, valid_input_data, valid_label):
        x_train = self.prepare_input(train_input_data)
        y_train = self.prepare_label(train_label)
        x_valid = self.prepare_input(valid_input_data)
        y_valid = self.prepare_label(valid_label)

        print('start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size, epochs=self.config.n_epochs,
                       validation_data=(x_valid, y_valid), callbacks=self.callbacks)
        print('training end...')

        print('score over valid data:')
        valid_pred = self.model.predict(x_valid)
        get_score_senti(y_valid, valid_pred)

    def score(self, input_data, label):
        input_pad = self.prepare_input(input_data)
        label = self.prepare_label(label)
        prediction = self.model.predict(input_pad)
        get_score_senti(label, prediction)

    def predict(self, input_data):
        input_pad = self.prepare_input(input_data)
        prediction = self.model.predict(input_pad)
        return np.argmax(prediction, axis=-1)

    # target dependent lstm
    def td_lstm(self):
        input_l = Input(shape=(self.left_max_len, ))
        input_r = Input(shape=(self.right_max_len, ))

        if self.use_elmo:
            l_elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                             mask_zero=True, hub_url=self.config.elmo_hub_url,
                                             elmo_trainable=self.config.elmo_trainable)
            r_elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                             mask_zero=True, hub_url=self.config.elmo_hub_url,
                                             elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                input_l_embed = SpatialDropout1D(0.2)(l_elmo_embedding(input_l))
                input_r_embed = SpatialDropout1D(0.2)(r_elmo_embedding(input_r))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                input_l_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_l), l_elmo_embedding(input_l)]))
                input_r_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_r), r_elmo_embedding(input_r)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            input_l_embed = SpatialDropout1D(0.2)(word_embedding(input_l))
            input_r_embed = SpatialDropout1D(0.2)(word_embedding(input_r))

        # regarding aspect string as the last unit
        if(self.is_cudnn):
            hidden_l = CuDNNLSTM(self.config.lstm_units)(input_l_embed)
        else:
            hidden_l = LSTM(self.config.lstm_units)(input_l_embed)
        if(self.is_cudnn):
            hidden_r = CuDNNLSTM(self.config.lstm_units, go_backwards=True)(input_r_embed)
        else:
            hidden_r = LSTM(self.config.lstm_units, go_backwards=True)(input_r_embed)

        hidden_concat = concatenate([hidden_l, hidden_r], axis=-1)

        return Model([input_l, input_r], hidden_concat)

    # target connection lstm
    def tc_lstm(self):
        input_l = Input(shape=(self.left_max_len,))
        input_r = Input(shape=(self.right_max_len,))
        input_aspect = Input(shape=(1,))

        if self.use_elmo:
            l_elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                             mask_zero=True, hub_url=self.config.elmo_hub_url,
                                             elmo_trainable=self.config.elmo_trainable)
            r_elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                             mask_zero=True, hub_url=self.config.elmo_hub_url,
                                             elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                input_l_embed = SpatialDropout1D(0.2)(l_elmo_embedding(input_l))
                input_r_embed = SpatialDropout1D(0.2)(r_elmo_embedding(input_r))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                input_l_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_l), l_elmo_embedding(input_l)]))
                input_r_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_r), r_elmo_embedding(input_r)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            input_l_embed = SpatialDropout1D(0.2)(word_embedding(input_l))
            input_r_embed = SpatialDropout1D(0.2)(word_embedding(input_r))

        if self.config.aspect_embed_type == 'random':
            asp_embedding = Embedding(input_dim=self.n_aspect, output_dim=self.config.aspect_embed_dim)
        else:
            asp_embedding = Embedding(input_dim=self.aspect_embeddings.shape[0],
                                      output_dim=self.config.aspect_embed_dim,
                                      trainable=self.config.aspect_embed_trainable)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)

        aspect_repeat_l = RepeatVector(self.left_max_len)(aspect_embed)
        input_l_concat = concatenate([input_l_embed, aspect_repeat_l], axis=-1)
        aspect_repeat_r = RepeatVector(self.right_max_len)(aspect_embed)
        input_r_concat = concatenate([input_r_embed, aspect_repeat_r], axis=-1)

        # regarding aspect string as the last unit
        if(self.is_cudnn):
            hidden_l = CuDNNLSTM(self.config.lstm_units)(input_l_concat)
        else:
            hidden_l = LSTM(self.config.lstm_units)(input_l_concat)
        if(self.is_cudnn):
            hidden_r = CuDNNLSTM(self.config.lstm_units, go_backwards=True)(input_r_concat)
        else:
            hidden_r = LSTM(self.config.lstm_units, go_backwards=True)(input_r_concat)

        hidden_concat = concatenate([hidden_l, hidden_r], axis=-1)

        return Model([input_l, input_r, input_aspect], hidden_concat)

    # lstm with aspect embedding
    def ae_lstm(self):
        input_text = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,),)

        if self.use_elmo:
            elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                           mask_zero=True, hub_url=self.config.elmo_hub_url,
                                           elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                text_embed = SpatialDropout1D(0.2)(elmo_embedding(input_text))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                text_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text), elmo_embedding(input_text)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

        if self.config.aspect_embed_type == 'random':
            asp_embedding = Embedding(input_dim=self.n_aspect, output_dim=self.config.aspect_embed_dim)
        else:
            asp_embedding = Embedding(input_dim=self.aspect_embeddings.shape[0],
                                      output_dim=self.config.aspect_embed_dim,
                                      trainable=self.config.aspect_embed_trainable)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)  # repeat aspect for every word in sequence

        input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
        if(self.is_cudnn):
            hidden = CuDNNLSTM(self.config.lstm_units)(input_concat)
        else:
            hidden = LSTM(self.config.lstm_units)(input_concat)

        return Model([input_text, input_aspect], hidden)

    # def at_lstm(self):
    #     input_text = Input(shape=(self.max_len,), name='input_text')
    #     input_aspect = Input(shape=(1,), name='input_aspect')
    #
    #     text_embed = Embedding(input_dim=self.weights.shape[0], output_dim=self.config.embedding_dim,
    #                            weights=[self.weights])(input_text)
    #     text_embed = SpatialDropout1D(0.2)(text_embed)
    #
    #     aspect_embed = Embedding(input_dim=self.config.n_aspect, output_dim=self.config.aspect_embed_dim)(
    #         input_aspect)
    #     aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
    #     repeat_aspect = RepeatVector(self.max_len)(aspect_embed)  # repeat aspect for every word in sequence
    #
    #     hidden_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(
    #         text_embed)  # hidden vectors output by bilstm
    #
    #     # compute attention weight for each hidden vector (step), refer to https://aclweb.org/anthology/D16-1058
    #     concat = concatenate([hidden_out, repeat_aspect], axis=-1)
    #     M = TimeDistributed(Dense(self.config.embedding_dim + self.config.aspect_embed_dim, activation='tanh'))(
    #         concat)
    #     attention = TimeDistributed(Dense(1))(M)
    #     attention = Flatten()(attention)
    #     attention = Activation('softmax')(attention)  # [batch_size, max_len]
    #
    #     # apply the attention
    #     repeat_attention = RepeatVector(2 * self.lstm_units)(attention)  # [batch_size, units, max_len)
    #     repeat_attention = Permute((2, 1))(repeat_attention)  # [batch_size, max_len, units]
    #     sent_representation = multiply([hidden_out, repeat_attention])
    #     sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)
    #
    #     return Model([input_text, input_aspect], sent_representation)

    # attention-based lstm (supporting masking)
    def at_lstm(self):
        input_text = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,),)

        if self.use_elmo:
            elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                           mask_zero=True, hub_url=self.config.elmo_hub_url,
                                           elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                text_embed = SpatialDropout1D(0.2)(elmo_embedding(input_text))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                text_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text), elmo_embedding(input_text)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

        if self.config.aspect_embed_type == 'random':
            asp_embedding = Embedding(input_dim=self.n_aspect, output_dim=self.config.aspect_embed_dim)
        else:
            asp_embedding = Embedding(input_dim=self.aspect_embeddings.shape[0],
                                      output_dim=self.config.aspect_embed_dim,
                                      trainable=self.config.aspect_embed_trainable)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)  # repeat aspect for every word in sequence
        if(self.is_cudnn):
            hidden_vecs = CuDNNLSTM(self.config.lstm_units, return_sequences=True)(text_embed)  # hidden vectors output by lstm
        else:
            hidden_vecs = LSTM(self.config.lstm_units, return_sequences=True)(text_embed)  # hidden vectors output by lstm
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)  # mask after concatenate will be same as hidden_out's mask

        # apply attention mechanism
        attend_weight = Attention()(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
        return Model([input_text, input_aspect], attend_hidden)

    # attention-based lstm with aspect embedding
    def atae_lstm(self):
        input_text = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,), )

        if self.use_elmo:
            elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                           mask_zero=True, hub_url=self.config.elmo_hub_url,
                                           elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                text_embed = SpatialDropout1D(0.2)(elmo_embedding(input_text))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                text_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text), elmo_embedding(input_text)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

        if self.config.aspect_embed_type == 'random':
            asp_embedding = Embedding(input_dim=self.n_aspect, output_dim=self.config.aspect_embed_dim)
        else:
            asp_embedding = Embedding(input_dim=self.aspect_embeddings.shape[0],
                                      output_dim=self.config.aspect_embed_dim,
                                      trainable=self.config.aspect_embed_trainable)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(self.max_len)(aspect_embed)  # repeat aspect for every word in sequence

        input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
        if(self.is_cudnn):
            hidden_vecs, state_h, _ = CuDNNLSTM(self.config.lstm_units, return_sequences=True, return_state=True)(input_concat)
        else:
            hidden_vecs, state_h, _ = LSTM(self.config.lstm_units, return_sequences=True, return_state=True)(input_concat)
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = Attention()(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        attend_hidden_dense = Dense(self.config.lstm_units)(attend_hidden)
        last_hidden_dense = Dense(self.config.lstm_units)(state_h)
        final_output = Activation('tanh')(add([attend_hidden_dense, last_hidden_dense]))

        return Model([input_text, input_aspect], final_output)

    # deep memory network
    def memnet(self):
        n_hop = 9
        input_text = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,))
        inputs = [input_text, input_aspect]

        # if self.use_elmo:
        #     elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
        #                                    mask_zero=True, hub_url=self.config.elmo_hub_url,
        #                                    elmo_trainable=self.config.elmo_trainable)
        #     if self.config.use_elmo_alone:
        #         text_embed = SpatialDropout1D(0.2)(elmo_embedding(input_text))
        #     else:
        #         word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
        #                                    output_dim=self.config.word_embed_dim,
        #                                    weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
        #                                    mask_zero=True)
        #         text_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text), elmo_embedding(input_text)]))
        # else:
        word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                   weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                   mask_zero=True)
        text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

        if self.config.use_loc_input:   # location attention
            input_loc = Input(shape=(self.max_len,))
            inputs.append(input_loc)
            input_loc_expand = Lambda(lambda x: K.expand_dims(x))(input_loc)
            text_embed = multiply([text_embed, input_loc_expand])

        if self.config.aspect_embed_type == 'random':
            asp_embedding = Embedding(input_dim=self.n_aspect, output_dim=self.config.aspect_embed_dim)
        else:
            asp_embedding = Embedding(input_dim=self.aspect_embeddings.shape[0],
                                      output_dim=self.config.aspect_embed_dim,
                                      trainable=self.config.aspect_embed_trainable)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d

        # the parameter of attention and linear layers are shared in different hops
        attention_layer = Attention(use_W=False, use_bias=True)
        linear_layer = Dense(self.config.word_embed_dim)
        # output from each computation layer, representing text in different level of abstraction
        computation_layers_out = [aspect_embed]

        for h in range(n_hop):
            # content attention layer
            repeat_out = RepeatVector(self.max_len)(computation_layers_out[-1])
            concat = concatenate([text_embed, repeat_out], axis=-1)
            attend_weight = attention_layer(concat)
            attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
            content_attend = multiply([text_embed, attend_weight_expand])
            content_attend = Lambda(lambda x: K.sum(x, axis=1))(content_attend)

            # linear layer
            out_linear = linear_layer(computation_layers_out[-1])
            computation_layers_out.append(add([content_attend, out_linear]))
        return Model(inputs, computation_layers_out[-1])

    # ram memory network with (location weighted) memory
    def ram(self):
        n_hop = 3

        # input module
        input_text = Input(shape=(self.max_len,))
        input_aspect = Input(shape=(1,))
        inputs = [input_text, input_aspect]

        if self.use_elmo:
            elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                           mask_zero=True, hub_url=self.config.elmo_hub_url,
                                           elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                text_embed = SpatialDropout1D(0.2)(elmo_embedding(input_text))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                text_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text), elmo_embedding(input_text)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

        if self.config.aspect_embed_type == 'random':
            asp_embedding = Embedding(input_dim=self.n_aspect, output_dim=self.config.aspect_embed_dim)
        else:
            asp_embedding = Embedding(input_dim=self.aspect_embeddings.shape[0],
                                      output_dim=self.config.aspect_embed_dim,
                                      trainable=self.config.aspect_embed_trainable)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d

        # memory module
        if(self.is_cudnn):
            hidden_out_1 = Bidirectional(CuDNNLSTM(self.config.lstm_units, return_sequences=True))(text_embed)
            memory = Bidirectional(CuDNNLSTM(self.config.lstm_units, return_sequences=True))(hidden_out_1)
        else:
            hidden_out_1 = Bidirectional(LSTM(self.config.lstm_units, return_sequences=True))(text_embed)
            memory = Bidirectional(LSTM(self.config.lstm_units, return_sequences=True))(hidden_out_1)

        # position-weighted memory module
        if self.config.use_loc_input:
            input_loc = Input(shape=(self.max_len,))
            inputs.append(input_loc)
            input_loc_expand = Lambda(lambda x: K.expand_dims(x))(input_loc)
            memory = multiply([memory, input_loc_expand])
        if self.config.use_offset_input:
            input_offset = Input(shape=(self.max_len,))
            inputs.append(input_offset)
            input_offset_expand = Lambda(lambda x: K.expand_dims(x))(input_offset)
            memory = concatenate([memory, input_offset_expand])

        # recurrent attention module
        final_attend = RecurrentAttention(units=self.config.lstm_units, n_hop=n_hop)([memory, aspect_embed])
        return Model(inputs, final_attend)

    # interactive attention network
    def ian(self):
        input_text = Input(shape=(self.max_len,))
        input_aspect_text = Input(shape=(self.asp_max_len,), )

        if self.use_elmo:
            elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                           mask_zero=True, hub_url=self.config.elmo_hub_url,
                                           elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                text_embed = SpatialDropout1D(0.2)(elmo_embedding(input_text))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                text_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text), elmo_embedding(input_text)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))

        if self.use_elmo:
            asp_elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode,
                                               idx2word=self.config.idx2aspect_token,
                                               mask_zero=True, hub_url=self.config.elmo_hub_url,
                                               elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                asp_text_embed = SpatialDropout1D(0.2)(asp_elmo_embedding(input_aspect_text))
            else:
                asp_text_embedding = Embedding(input_dim=self.aspect_text_embeddings.shape[0],
                                               output_dim=self.config.word_embed_dim,
                                               weights=[self.aspect_text_embeddings],
                                               trainable=self.config.word_embed_trainable,
                                               mask_zero=True)
                asp_text_embed = SpatialDropout1D(0.2)(concatenate([asp_text_embedding(input_aspect_text),
                                                                    asp_elmo_embedding(input_aspect_text)]))
        else:
            asp_text_embedding = Embedding(input_dim=self.aspect_text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.aspect_text_embeddings],
                                           trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
            asp_text_embed = SpatialDropout1D(0.2)(asp_text_embedding(input_aspect_text))
        if(self.is_cudnn):
            hidden_text = CuDNNLSTM(self.config.lstm_units, return_sequences=True)(text_embed)
            hidden_asp_text = CuDNNLSTM(self.config.lstm_units, return_sequences=True)(asp_text_embed)
        else:
            hidden_text = LSTM(self.config.lstm_units, return_sequences=True)(text_embed)
            hidden_asp_text = LSTM(self.config.lstm_units, return_sequences=True)(asp_text_embed)

        attend_concat = InteractiveAttention()([hidden_text, hidden_asp_text])

        return Model([input_text, input_aspect_text], attend_concat)

    # content attention based aspect based sentiment classification model
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

        if self.use_elmo:
            text_elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                                mask_zero=True, hub_url=self.config.elmo_hub_url,
                                                elmo_trainable=self.config.elmo_trainable)
            l_elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                             mask_zero=True, hub_url=self.config.elmo_hub_url,
                                             elmo_trainable=self.config.elmo_trainable)
            r_elmo_embedding = ELMoEmbedding(output_mode=self.config.elmo_output_mode, idx2word=self.config.idx2token,
                                             mask_zero=True, hub_url=self.config.elmo_hub_url,
                                             elmo_trainable=self.config.elmo_trainable)
            if self.config.use_elmo_alone:
                text_embed = SpatialDropout1D(0.2)(text_elmo_embedding(input_text))
                text_l_embed = SpatialDropout1D(0.2)(l_elmo_embedding(input_text_l))
                text_r_embed = SpatialDropout1D(0.2)(r_elmo_embedding(input_text_r))
            else:
                word_embedding = Embedding(input_dim=self.text_embeddings.shape[0],
                                           output_dim=self.config.word_embed_dim,
                                           weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                           mask_zero=True)
                text_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text), text_elmo_embedding(input_text)]))
                text_l_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text_l), l_elmo_embedding(input_text_l)]))
                text_r_embed = SpatialDropout1D(0.2)(concatenate([word_embedding(input_text_r), r_elmo_embedding(input_text_r)]))
        else:
            word_embedding = Embedding(input_dim=self.text_embeddings.shape[0], output_dim=self.config.word_embed_dim,
                                       weights=[self.text_embeddings], trainable=self.config.word_embed_trainable,
                                       mask_zero=True)
            text_embed = SpatialDropout1D(0.2)(word_embedding(input_text))
            text_l_embed = SpatialDropout1D(0.2)(word_embedding(input_text_l))
            text_r_embed = SpatialDropout1D(0.2)(word_embedding(input_text_r))

        if self.config.aspect_embed_type == 'random':
            asp_embedding = Embedding(input_dim=self.n_aspect, output_dim=self.config.aspect_embed_dim)
        else:
            asp_embedding = Embedding(input_dim=self.aspect_embeddings.shape[0],
                                      output_dim=self.config.aspect_embed_dim,
                                      trainable=self.config.aspect_embed_trainable)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d

        # regarding aspect string as the first unit
        hidden_l = GRU(self.config.lstm_units, go_backwards=True, return_sequences=True)(text_l_embed)
        hidden_r = GRU(self.config.lstm_units, return_sequences=True)(text_r_embed)

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

        return Model([input_text, input_text_l, input_text_r, input_aspect, input_mask], final_output)

