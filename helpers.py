from learning.DataProvider import DataProvider
from learning.Model import Model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os.path
import tensorflow as tf
import dill
import weakref


def provide_data(path: str) -> DataProvider:
    data: DataProvider = DataProvider(path=path)
    data.load_from_csv()
    data.remove_duplicates()
    data.converting_upper_case_to_lower_case()
    data.converting_upper_case_to_lower_case()
    data.stop_word_removal()
    data.convert_label_to_categorical()
    return(data)


def train_data(dp: DataProvider) -> Model:
    model: Model = Model()
    model.add_dp_to_model(dp=dp)
    model.word_embeding()
    model.split_train_test()
    model.train_model()
    return model


def predict(model: Model, text: str) -> str:
    sentiment = ['Negative', 'Neutral', 'Positive']
    sequence = model.tokenizer.texts_to_sequences(text)
    test = pad_sequences(sequence, maxlen=model.max_len)
    res = sentiment[np.around(model.predict(
        test), decimals=0).argmax(axis=1)[0]]
    return(res)


def load_model():
    filename = 'finalized_model'
    if not os.path.isdir(filename):
        # path = '/home/orlevi/dev/projects/sentiment_analyze/learning/train.csv'
        path = '/home/orlevi/dev/Projects/sentiment_analyze/learning/train.csv'
        data: DataProvider = provide_data(path=path)
        model = train_data(data)
        model.save(filename)

    loaded_model = tf.keras.models.load_model(filename)
    return loaded_model


if __name__ == '__main__':
    load_model()
