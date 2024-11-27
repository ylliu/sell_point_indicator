import datetime
from unittest import TestCase

from train_model import TrainModel


class TestTrainModel(TestCase):
    def test_use_model_predictions(self):
        train_model = TrainModel()
        file_names = ['./test/601360.csv', './test/301171.csv', './test/300785.csv', './test/300450.csv']  # 添加你的文件名
        train_model.load_test_case(file_names)
        train_model.train_model()

        train_model.code_sell_point('601360.XSHG')
        # train_model.code_sell_point('300450.XSHE')

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
                              datetime.datetime.strptime('2024-11-27 10:57:00', '%Y-%m-%d %H:%M:%S'),
                              datetime.datetime.strptime('2024-11-27 11:01:00', '%Y-%m-%d %H:%M:%S'))
        train_model.retrain_with_all_data()
        train_model.code_sell_point('300703.XSHE')

    def test_get_all_test_csv(self):
        train_model = TrainModel()
        train_model.get_all_test_csv()
