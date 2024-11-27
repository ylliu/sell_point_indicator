from unittest import TestCase

from Ashare import get_price


class Test(TestCase):
    def test_get_price(self):
        df = get_price('bj832175', frequency='1m', count=241)
        print(df)
