import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class DataProvider:    
    def __init__(self, path: str):

        self.path = path
        self.df: pd.DataFrame = None 


    def load_from_csv(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError('File not exists')

        df: pd.DataFrame = pd.read_csv(self.path, encoding='ISO-8859-1')
        if 'text' not in df.columns:
            raise Exception("No text column in Data Frame")
        if 'sentiment' not in df.columns:
            raise Exception("No sentiment column in Data Frame")
        self.df = df[['text', 'sentiment']]

    def remove_duplicates(self):
        '''
        This method drop duplicates from text column
        '''
        self.df.drop_duplicates(subset=['text'], inplace=True)


    def converting_upper_case_to_lower_case(self):
        '''
        This method clean the text from wired symboles
        and converting from upper case to lower case
        '''
        self.df['processed_text'] = self.df['text'].str.lower()\
            .str.replace('(@[a-z0-9]+)\w+',' ')\
            .str.replace('(http\S+)', ' ')\
            .str.replace('([^0-9a-z \t])',' ')\
            .str.replace(' +',' ')

        self.df.drop('text',axis=1, inplace=True)
        self.df.rename(columns={'processed_text': 'text'}, inplace=True)


    def stop_word_removal(self):
        '''
        asfdgsdfhds
        '''
        stop_words = set(stopwords.words('english'))
        word_tokens = []
        self.df.dropna(subset = ['text'], inplace=True)

        for j,i in self.df['text'].iteritems():
            word_tokens.append(word_tokenize(i))
        
        tokens_without_stop_words = []
        for j,i in self.df['text'].iteritems():
                word_token = word_tokenize(i)
                token = []
                for word in word_token:
                    if not word in stop_words:
                        token.append(word)
                token = ' '.join(token)
                tokens_without_stop_words.append(token)

        self.df['text'] = tokens_without_stop_words

    def convert_label_to_categorical(self):
        self.df['sentiment'] = pd.Categorical(self.df.sentiment,
                         categories=['negative', 'neutral', 'positive'])

