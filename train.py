# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: train.py

@time: 2019/1/5 10:02

@desc:

"""

import os
import time
from config import Config
from data_loader import load_input_data, load_label
from models import SentimentModel
from utils import pickle_load

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def train_model(data_folder, data_name, level, model_name, is_aspect_term=True):
    config.data_folder = data_folder
    config.data_name = data_name
    if not os.path.exists(os.path.join(config.checkpoint_dir, data_folder)):
        os.makedirs(os.path.join(config.checkpoint_dir, data_folder))
    config.level = level
    config.model_name = model_name
    config.is_aspect_term = is_aspect_term
    config.init_input()
    config.exp_name = '{}_{}_wv_{}'.format(model_name, level, config.word_embed_type)
    config.exp_name = config.exp_name + '_update' if config.word_embed_trainable else config.exp_name + '_fix'
    if config.use_aspect_input:
        config.exp_name += '_aspv_{}'.format(config.aspect_embed_type)
        config.exp_name = config.exp_name + '_update' if config.aspect_embed_trainable else config.exp_name + '_fix'
    if config.use_emlo:
        config.exp_name += '_elmo_alone_{}_mode_{}_{}'.format(config.use_elmo_alone, config.elmo_output_mode,
                                                              'update' if config.elmo_trainable else 'fix')

    print(config.exp_name)
    model = SentimentModel(config)

    test_input = load_input_data(data_folder, 'test', level, config.use_text_input, config.use_text_input_l,
                                 config.use_text_input_r, config.use_text_input_r_with_pad, config.use_aspect_input,
                                 config.use_aspect_text_input, config.use_loc_input, config.use_offset_input,
                                 config.use_mask)
    test_label = load_label(data_folder, 'test')

    if not os.path.exists(os.path.join(config.checkpoint_dir, '%s/%s.hdf5' % (data_folder, config.exp_name))):
        start_time = time.time()

        train_input = load_input_data(data_folder, 'train', level, config.use_text_input, config.use_text_input_l,
                                      config.use_text_input_r, config.use_text_input_r_with_pad,
                                      config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                      config.use_offset_input, config.use_mask)
        train_label = load_label(data_folder, 'train')
        valid_input = load_input_data(data_folder, 'valid', level, config.use_text_input, config.use_text_input_l,
                                      config.use_text_input_r, config.use_text_input_r_with_pad,
                                      config.use_aspect_input, config.use_aspect_text_input, config.use_loc_input,
                                      config.use_offset_input, config.use_mask)
        valid_label = load_label(data_folder, 'valid')

        train_combine_valid_input = []
        for i in range(len(train_input)):
            train_combine_valid_input.append(train_input[i] + valid_input[i])
        train_combine_valid_label = train_label + valid_label

        model.train(train_combine_valid_input, train_combine_valid_label, test_input, test_label)

        elapsed_time = time.time() - start_time
        print('training time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # load the best model
    model.load()

    # print('score over valid data...')
    # model.score(valid_input, valid_label)
    print('score over test data...')
    model.score(test_input, test_label)


if __name__ == '__main__':
    config = Config()
    config.use_emlo = False
    config.use_elmo_alone = False
    config.elmo_trainable = False

    # config.word_embed_trainable = True
    # config.aspect_embed_trainable = True
    # train_model('laptop/term', 'laptop', 'word', 'td_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'tc_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'ae_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'at_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'atae_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'memnet')
    # train_model('laptop/term', 'laptop', 'word', 'ram')
    # train_model('laptop/term', 'laptop', 'word', 'ian')
    # train_model('laptop/term', 'laptop', 'word', 'cabasc')
    #
    # train_model('restaurant/term', 'restaurant', 'word', 'td_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'tc_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'ae_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'at_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'atae_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'memnet')
    # train_model('restaurant/term', 'restaurant', 'word', 'ram')
    # train_model('restaurant/term', 'restaurant', 'word', 'ian')
    # train_model('restaurant/term', 'restaurant', 'word', 'cabasc')
    #
    # train_model('twitter', 'twitter', 'word', 'td_lstm')
    # train_model('twitter', 'twitter', 'word', 'tc_lstm')
    # train_model('twitter', 'twitter', 'word', 'ae_lstm')
    # train_model('twitter', 'twitter', 'word', 'at_lstm')
    # train_model('twitter', 'twitter', 'word', 'atae_lstm')
    # train_model('twitter', 'twitter', 'word', 'memnet')
    # train_model('twitter', 'twitter', 'word', 'ram')
    # train_model('twitter', 'twitter', 'word', 'ian')
    # train_model('twitter', 'twitter', 'word', 'cabasc')
    #
    # config.word_embed_trainable = False
    # config.aspect_embed_trainable = True
    # train_model('laptop/term', 'laptop', 'word', 'td_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'tc_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'ae_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'at_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'atae_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'memnet')
    # train_model('laptop/term', 'laptop', 'word', 'ram')
    # train_model('laptop/term', 'laptop', 'word', 'ian')
    # train_model('laptop/term', 'laptop', 'word', 'cabasc')
    #
    # train_model('restaurant/term', 'restaurant', 'word', 'td_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'tc_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'ae_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'at_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'atae_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'memnet')
    # train_model('restaurant/term', 'restaurant', 'word', 'ram')
    # train_model('restaurant/term', 'restaurant', 'word', 'ian')
    # train_model('restaurant/term', 'restaurant', 'word', 'cabasc')
    #
    # train_model('twitter', 'twitter', 'word', 'td_lstm')
    # train_model('twitter', 'twitter', 'word', 'tc_lstm')
    # train_model('twitter', 'twitter', 'word', 'ae_lstm')
    # train_model('twitter', 'twitter', 'word', 'at_lstm')
    # train_model('twitter', 'twitter', 'word', 'atae_lstm')
    # train_model('twitter', 'twitter', 'word', 'memnet')
    # train_model('twitter', 'twitter', 'word', 'ram')
    # train_model('twitter', 'twitter', 'word', 'ian')
    # train_model('twitter', 'twitter', 'word', 'cabasc')

    config.word_embed_trainable = True
    config.aspect_embed_trainable = True
    print("开始训练。。。。。。。")
    # train_model('laptop/term', 'laptop', 'word', 'td_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'tc_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'ae_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'at_lstm')
    train_model('laptop/term', 'laptop', 'word', 'atae_lstm')
    # train_model('laptop/term', 'laptop', 'word', 'memnet')
    # train_model('laptop/term', 'laptop', 'word', 'ram')
    # train_model('laptop/term', 'laptop', 'word', 'ian')
    # train_model('laptop/term', 'laptop', 'word', 'cabasc')

    # train_model('restaurant/term', 'restaurant', 'word', 'td_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'tc_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'ae_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'at_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'atae_lstm')
    # train_model('restaurant/term', 'restaurant', 'word', 'memnet')
    # train_model('restaurant/term', 'restaurant', 'word', 'ram')
    # train_model('restaurant/term', 'restaurant', 'word', 'ian')
    # train_model('restaurant/term', 'restaurant', 'word', 'cabasc')

    # train_model('twitter', 'twitter', 'word', 'td_lstm')
    # train_model('twitter', 'twitter', 'word', 'tc_lstm')
    # train_model('twitter', 'twitter', 'word', 'ae_lstm')
    # train_model('twitter', 'twitter', 'word', 'at_lstm')
    # train_model('twitter', 'twitter', 'word', 'atae_lstm')
    # train_model('twitter', 'twitter', 'word', 'memnet')
    # train_model('twitter', 'twitter', 'word', 'ram')
    # train_model('twitter', 'twitter', 'word', 'ian')
    # train_model('twitter', 'twitter', 'word', 'cabasc')

