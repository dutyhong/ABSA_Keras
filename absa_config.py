class Config(object):
    def __init__(self):
        self.max_len = 15
        self.content_embed_dim = 128
        self.max_content_vocab_size = 5000
        self.aspect_embed_dim = 64
        self.max_aspect_vocab_size = 15
        self.lstm_units = 32
        self.is_cudnn = False
        self.dense_units = 64
        self.n_classes = 3