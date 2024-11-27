import datetime
from unittest import TestCase

from train_model import TrainModel


class TestTrainModel(TestCase):
    def test_use_model_predictions(self):
        train_model = TrainModel()
        file_names = ['./test/601360.csv', './test/301171.csv', './test/300785.csv', './test/300450.csv']  # 添加你的文件名
        train_model.load_test_case(file_names)
        train_model.train_model()

        # train_model.code_sell_point('bj832175')
        train_model.code_sell_point('300450.XSHE')

    def test_save_data2(self):
        self.fail()

    def test_send_message_to_wechat(self):
        train_model = TrainModel()
        train_model.send_message_to_wechat("300001")

    def test_send_message2_wechat(self):
        strain_model = TrainModel()
        strain_model.send_message2_wechat("ss")

    def test_should_train_model_use_new_files(self):
        train_model = TrainModel()
        file_names = ['test/601360.csv', 'test/301171.csv', 'test/300785.csv']  # 添加你的文件名
        train_model.load_test_case(file_names)
        train_model.train_model()

        train_model.save_data('002156.XSHE',
                              '2024-11-27 10:57:00',
                              '2024-11-27 11:01:00', "Sell_Point")
        train_model.retrain_with_all_data()
        train_model.code_sell_point('300703.XSHE')

    def test_get_all_test_csv(self):
        train_model = TrainModel()
        train_model.get_all_test_csv('')

    def test_should_save_buy_point_data(self):
        train_model = TrainModel()
        train_model.save_data('300822.XSHE', '2024-11-27 9:44:00', '2024-11-27 9:48:00', train_model.BUY_POINT)
        train_model.save_data('301011.XSHE', '2024-11-27 9:54:00', '2024-11-27 10:03:00', train_model.BUY_POINT)
        train_model.save_data('300220.XSHE', '2024-11-27 9:48:00', '2024-11-27 9:57:00', train_model.BUY_POINT)
        train_model.save_data('300459.XSHE', '2024-11-27 9:42:00', '2024-11-27 9:53:00', train_model.BUY_POINT)
        # train_model.save_data('600250.XSHG', '2024-11-27 9:20:00', '2024-11-27 9:20:00', train_model.BUY_POINT)
        # train_model.save_data('002054.XSHE', '2024-11-27 9:20:00', '2024-11-27 9:20:00', train_model.BUY_POINT)
        train_model.train_with_all_buy_data()
        train_model.code_buy_point('832175.BJ')
