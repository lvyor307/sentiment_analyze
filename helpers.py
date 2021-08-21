from learning.DataProvider import DataProvider
from learning.Model import Model
import numpy as np
from keras.preprocessing.sequence import pad_sequences
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
    model.add_dp_to_model(dp=dp)
    model.word_embeding()
    model.split_train_test()
    model.train_model()
    return model


def predict(model: Model, text: str) -> str:
    sentiment = ['Negative', 'Neutral', 'Positive']
    sequence = model.tokenizer.texts_to_sequences(text)
    test = pad_sequences(sequence, maxlen=model.max_len)
    res = sentiment[np.around(model.model.predict(
        test), decimals=0).argmax(axis=1)[0]]
    return(res)


def load_model():
    filename = 'finalized_model.pickle'
    if not os.path.isdir(filename):
        # path = '/home/orlevi/dev/projects/sentiment_analyze/learning/train.csv'
        path = '/home/orlevi/dev/Projects/sentiment_analyze/learning/train.csv'
        data: DataProvider = provide_data(path=path)
        model: Model = train_data(data)
        with open(filename, "wb") as file_:
            pickle.dump(model, file_, -1)
        # model.model.save(filename)

    # loaded_model = tf.keras.models.load_model(filename)
    loaded_model = pickle.load(open(filename, "rb", -1))
    return(loaded_model)


# if __name__ == '__main__':
#     m = load_model()
#     text = 'Hi it is a great text'
#     model_to_predict: Model = Model()
#     model_to_predict.add_model_to_model(model=m)
#     str = predict(model=model_to_predict, text=text)
#     print(str)
