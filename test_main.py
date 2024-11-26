from datetime import datetime
from unittest import TestCase

from main import save_data


class Test(TestCase):
    def test_save_data(self):
        test_code = '300010.XSHE'
        sell_start_time = '2024-11-26 10:02:00'
        sell_end_time = '2024-11-26 10:08:00'
        save_data(test_code, datetime.strptime(sell_start_time, '%Y-%m-%d %H:%M:%S'),
                  datetime.strptime(sell_end_time, '%Y-%m-%d %H:%M:%S'))
