from learning.DataProvider import DataProvider
import pytest

def test_load_from_csv_file_not_exist(data_provider: DataProvider):

    path = 'not_found'
    dp = data_provider(path)
    with pytest.raises (FileNotFoundError):
        dp.load_from_csv()
        # assert ex == "file not found"


def test_load_from_csv_success(data_provider: DataProvider):
    try:
        path='/home/orlevi/dev/projects/sentiment_analyze/learning/train.csv'
        dp = data_provider (path)
        # dp_1 = dp (path)
        dp.load_from_csv()
    except Exception as exc:
        pytest.fail ("file not found")