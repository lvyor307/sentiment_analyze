import os
import sys
import pytest

path_to_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
sys.path.insert(0, path_to_root)

from learning.DataProvider import DataProvider


@pytest.fixture(scope="module")
def data_provider():
    def _data_provider (path :str):
        return DataProvider(path)
    
    return _data_provider

