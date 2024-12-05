import datetime
from unittest import TestCase

from SimulatedClock import SimulatedClock
from train_model import TrainModel


class TestTrainModel(TestCase):
    def test_use_model_predictions(self):
        train_model = TrainModel()
        train_model.retrain_with_all_data()
        # train_model.code_sell_point('bj832175')
        train_model.code_sell_point('sz001308')

    def test_save_data2(self):
        train_model = TrainModel()
        train_model.save_data2('sh600171', 241)

    def test_send_message_to_wechat(self):
        train_model = TrainModel()
        train_model.send_message_to_wechat("300001")

    def test_send_message2_wechat(self):
        strain_model = TrainModel()
        strain_model.send_message2_wechat("ss")

    def test_should_train_model_use_new_files(self):
        train_model = TrainModel()
        # file_names = ['test/sell/601360.csv', 'test/sell/301171.csv', 'test/sell/300785.csv']  # 添加你的文件名
        # train_model.load_test_case(file_names)
        # train_model.train_model()

        train_model.save_data('002416.XSHE',
                              '2024-11-28 9:51:00',
                              '2024-11-28 9:54:00', "Sell_Point")
        train_model.retrain_with_all_data()
        train_model.code_sell_point('002469.XSHE')

    def test_get_all_test_csv(self):
        train_model = TrainModel()
        train_model.get_all_test_csv('')

    def test_should_save_buy_point_data(self):
        train_model = TrainModel()
        train_model.save_data('sz300561', '2024-12-02 9:32:00', '2024-12-02 9:34:00', train_model.BUY_POINT)
        train_model.save_data('sz300033', '2024-12-02 10:20:00', '2024-12-02 10:23:00', train_model.BUY_POINT)
        train_model.save_data('sh600206', '2024-12-02 9:38:00', '2024-12-02 9:40:00', train_model.BUY_POINT)
        train_model.train_with_all_buy_data()
        train_model.code_buy_point('sz300561')

    def test_code_sell_point_use_file(self):
        train_model = TrainModel()
        train_model.retrain_with_all_data()
        train_model.code_sell_point('sz300622')
        # train_model.code_sell_point_use_file('sz300622_60')

    # todo 一次获取60分钟的数据
    def test_simulate_market_by_minute(self):
        train_model = TrainModel()
        train_model.retrain_with_all_data()
        code = 'sz002862'
        train_model.save_data2(code,500)
        clock = SimulatedClock()
        time = clock.get_current_time()
        while not clock.is_time_to_end():
            df = train_model.get_time_series_data('%s.csv' % code, time, 241)
            train_model.code_sell_point_use_date(df, code)
            time = clock.next()

    def test_get_time_series_data(self):
        train_model = TrainModel()
        time = datetime.datetime.strptime("2024-11-29 09:30:00", '%Y-%m-%d %H:%M:%S')
        df = train_model.get_time_series_data('sz300622.csv', time)
        self.assertEqual(60, len(df))
        self.assertEqual("2024-11-29 09:30:00", df['time'].iloc[-1])
