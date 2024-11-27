import time
from datetime import datetime

from train_model import TrainModel

if __name__ == "__main__":
    train_model = TrainModel()
    file_names = ['601360.csv', '301171.csv', '300785.csv', '300450.csv']  # 添加你的文件名
    train_model.load_test_case(file_names)
    train_model.train_model()
    send_times = [8, 12, 15, 20]
    while True:
        to_monitor_code = ['601360.XSHG', '000548.XSHE', '301171.XSHE', '300058.XSHE']
        for code in to_monitor_code:
            train_model.code_sell_point(code)
        time.sleep(60)
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        if current_hour in send_times and current_minute == 0:  # 确保是整点
            train_model.send_message2_wechat("卖点助手在线中")
