from unittest import TestCase

from train_model import TrainModel


class TestTrainModel(TestCase):
    def test_use_model_predictions(self):
        train_model = TrainModel()
        file_names = ['601360.csv', '301171.csv', '300785.csv', '300450.csv']  # 添加你的文件名
        train_model.load_test_case(file_names)
        train_model.train_model()

        train_model.code_sell_point('000548.XSHE')

    def test_save_data2(self):
        self.fail()


    def test_send_message_to_wechat(self):
        train_model = TrainModel()
        train_model.send_message_to_wechat("300001")