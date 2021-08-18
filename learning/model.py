from DataProvider import DataProvider














if __name__ == '__main__':
    path = '/home/orlevi/dev/projects/sentiment_analyze/learning/train.csv'
    data: DataProvider = DataProvider(path=path)
    data.load_from_csv()
    data.remove_duplicates()
    data.converting_upper_case_to_lower_case()
    data.converting_upper_case_to_lower_case()
    data.stop_word_removal()    

