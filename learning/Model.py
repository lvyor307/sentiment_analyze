import numpy as np
from learning.DataProvider import DataProvider
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint


class Model:
    def __init__(self):
        self.dp = None
        self.model = None
        self.words_embeded = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.lable = None
        self.max_len = 200
        self.max_words = 5000
        self.tokenizer = Tokenizer(num_words=5000)

    def add_dp_to_model(self, dp: DataProvider):
        self.dp = dp

    def word_embeding(self):
        X = self.dp.df['text']
        self.tokenizer.fit_on_texts(X)
        sequences = self.tokenizer.texts_to_sequences(X)
        self.words_embeded = pad_sequences(sequences, maxlen=self.max_len)
        self.lable = self.dp.df['sentiment'].map(
            {"neutral": 1, "negative": 0, "positive": 2})

    def split_train_test(self):
        X, y = self.words_embeded, self.lable
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                                                test_size=0.2, random_state=217)

    def train_model(self):
        # Here I train a LSTM model:
        self.model = Sequential()
        # The embedding layer
        self.model.add(layers.Embedding(self.max_words, 20))
        self.model.add(layers.LSTM(15, dropout=0.5))  # Our LSTM layer
        self.model.add(layers.Dense(3, activation='softmax'))
        self.model.compile(
            optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        checkpoint1 = ModelCheckpoint("model.hdf5", monitor='val_accuracy',
                                      verbose=1, save_best_only=True, mode='auto',
                                      period=1, save_weights_only=False)

        self.model.fit(self.X_train, self.y_train, epochs=1,
                       validation_data=(self.X_test, self.y_test),
                       callbacks=[checkpoint1])

    def add_model_to_model(self, model):
        self.model = model
