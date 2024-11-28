import time
from datetime import datetime

from train_model import TrainModel

if __name__ == "__main__":
    train_model = TrainModel()
    train_model.retrain_with_all_data()
    send_times = [8, 12, 15, 20]
    to_monitor_code = ['601360.XSHG', '000548.XSHE', '301171.XSHE', '002131.XSHE']
    while True:
        # 获取当前时间
        current_time = datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        print('monitoring')
        current_time_str = current_time.strftime('%H:%M')
        if '09:30' <= current_time_str < '11:30' or '13:00' <= current_time_str < '15:00':
            for code in to_monitor_code:
                train_model.code_sell_point(code)
            if current_hour in send_times and current_minute == 0:  # 确保是整点
                train_model.send_message2_wechat("卖点助手在线中")
        time.sleep(60)