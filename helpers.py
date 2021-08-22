from learning.DataProvider import DataProvider
from learning.Model import Model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os.path
import tensorflow as tf
import pickle


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
    model.set_dp(dp=dp)
    model.word_embeding()
    model.split_train_test()
    model.train_model()
    return model


def load_model():
    model_name = 'model'
    if not os.path.isfile(model_name):
        path = '/home/orlevi/dev/projects/sentiment_analyze/learning/train.csv'
        # path = '/home/orlevi/dev/Projects/sentiment_analyze/learning/train.csv'
        data: DataProvider = provide_data(path=path)
        data.save_dp()
        model: Model = train_data(data)
        model.model.save('model')
    loaded_model = tf.keras.models.load_model(model_name)
    return(loaded_model)


def load_dp():
    with open('dp.pickle', 'rb') as handle:
             dp = pickle.load(handle)
    return(dp)

def predict(model: Model, text: str) -> str:
    sentiment = ['Negative', 'Neutral', 'Positive']
    sequence = model.tokenizer.texts_to_sequences([text])
    test = pad_sequences(sequence, maxlen=model.max_len)
    res = sentiment[np.around(model.model.predict(
        test), decimals=0).argmax(axis=1)[0]]
    return(res)


