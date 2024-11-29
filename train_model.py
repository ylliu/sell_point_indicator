import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from Ashare import get_price


class TrainModel:
    def __init__(self):
        self.loaded_model = None
        self.sell_model_file = None
        self.features = ['Price', 'Volume', 'SMA5', 'Price_change', 'Volume_change']
        self.BUY_POINT = "Buy_Point"
        self.SELL_POINT = "Sell_Point"
        self.buy_model_file = None

    def data_convert(self, file_name):
        data = pd.read_csv(file_name)
        # 预处理：如去除缺失值、归一化等
        data.ffill()
        # 归一化价格和成交量数据
        scaler = MinMaxScaler()
        data[['Price', 'Volume']] = scaler.fit_transform(data[['Price', 'Volume']])

        # 添加简单的移动平均线作为特征
        data['SMA5'] = data['Price'].rolling(window=5).mean()
        # 添加价格变化率
        data['Price_change'] = data['Price'].pct_change()
        data['Price_change'] = data['Price_change'].replace([np.inf, -np.inf], 0)
        # 添加成交量变化率
        data['Volume_change'] = data['Volume'].pct_change()
        data['Volume_change'] = data['Volume_change'].replace([np.inf, -np.inf], 0)

        # 去除空值
        data.dropna(inplace=True)
        return data

    def data_convert2(self, data):
        # 预处理：如去除缺失值、归一化等
        data.ffill()
        # 归一化价格和成交量数据
        scaler = MinMaxScaler()
        data[['Price', 'Volume']] = scaler.fit_transform(data[['Price', 'Volume']])

        # 添加简单的移动平均线作为特征
        data['SMA5'] = data['Price'].rolling(window=5).mean()
        # 添加价格变化率
        data['Price_change'] = data['Price'].pct_change()
        data['Price_change'] = data['Price_change'].replace([np.inf, -np.inf], 0)
        # 添加成交量变化率
        data['Volume_change'] = data['Volume'].pct_change()
        data['Volume_change'] = data['Volume_change'].replace([np.inf, -np.inf], 0)

        # 去除空值
        data.dropna(inplace=True)
        return data

    def load_and_merge_data(self, file_names):
        all_data = []
        for file in file_names:
            data = self.data_convert(file)
            all_data.append(data)
        # 合并所有数据
        merged_data = pd.concat(all_data, axis=0, ignore_index=True)
        return merged_data

    def load_test_case(self, file_names):
        self.data = self.load_and_merge_data(file_names)

    def train_model(self):
        # 输入特征和目标变量
        X = self.data[self.features]
        y = self.data['Sell_Point']  # 卖点标签（1 为卖点，0 为非卖点）
        X_train = X
        y_train = y
        # 划分训练集和测试集

        # 创建 XGBoost 模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # model = xgb.XGBClassifier()
        # 训练模型
        model.fit(X_train, y_train)

        # 保存模型到文件
        self.sell_model_file = 'sell_point_model.pkl'
        pickle.dump(model, open(self.sell_model_file, 'wb'))

    def load_model_predict(self, x_input):
        # 使用加载的模型进行预测
        predictions = self.loaded_model.predict(x_input)
        return predictions

    def code_sell_point(self, code):
        test_code = code
        self.save_data2(test_code)
        data_test = self.data_convert(f'{test_code}.csv')
        # 输入特征和目标变量
        X_test = data_test[self.features]

        # 评估模型
        from sklearn.metrics import accuracy_score

        y_pred = self.load_model_predict(X_test)
        if y_pred[-1] == 1:
            self.send_message_to_wechat(code)
        # print(y_test)
        # 输出预测结果与对应的时间
        data_test['Predicted_Sell_Point'] = y_pred  # 将预测的卖点添加到数据中

        # 只显示预测为卖点（Sell_Point = 1）的记录
        sell_points = data_test[data_test['Predicted_Sell_Point'] == 1]
        # 打印时间、卖点预测值和实际标签
        print('code:', code)
        print(sell_points[['time', 'Predicted_Sell_Point']].reset_index())

    def code_sell_point_use_date(self, data_input, code):
        data_test = self.data_convert2(data_input)
        # 输入特征和目标变量
        X_test = data_test[self.features]
        y_pred = self.load_model_predict(X_test)
        if y_pred[-1] == 1:
            self.send_message_to_wechat(code)
            print('code:', code)
            data_test['Predicted_Sell_Point'] = y_pred  # 将预测的卖点添加到数据中
            # 只显示预测为卖点（Sell_Point = 1）的记录
            sell_points = data_test[data_test['Predicted_Sell_Point'] == 1]
            # 打印时间、卖点预测值和实际标签
            print(sell_points[['time', 'Predicted_Sell_Point']].reset_index())

    def code_sell_point_use_file(self, csv_file):

        data_test = self.data_convert(f'{csv_file}.csv')
        # 输入特征和目标变量
        X_test = data_test[self.features]

        # 评估模型
        from sklearn.metrics import accuracy_score

        y_pred = self.load_model_predict(X_test)
        if y_pred[-1] == 1:
            self.send_message_to_wechat(csv_file)
        # print(y_test)
        # 输出预测结果与对应的时间
        data_test['Predicted_Sell_Point'] = y_pred  # 将预测的卖点添加到数据中

        # 只显示预测为卖点（Sell_Point = 1）的记录
        sell_points = data_test[data_test['Predicted_Sell_Point'] == 1]
        # 打印时间、卖点预测值和实际标签
        print(sell_points[['time', 'Predicted_Sell_Point']].reset_index())

    def save_data(self, code, sell_start, sell_end, action_type):
        sell_start = datetime.strptime(sell_start, '%Y-%m-%d %H:%M:%S')
        sell_end = datetime.strptime(sell_end, '%Y-%m-%d %H:%M:%S')
        df = get_price(code, frequency='1m', count=241)  # 支持'1m','5m','15m','30m','60m'
        last_index = df.index[-1]
        df = df.drop(last_index)
        df.index.name = 'time'
        # 重命名列
        df.rename(columns={'close': 'Price', 'volume': 'Volume'}, inplace=True)
        sell_mask = (df.index >= sell_start) & (df.index <= sell_end)
        # 保存DataFrame为CSV文件
        df[action_type] = sell_mask.astype(int)
        if action_type == self.BUY_POINT:
            csv_file_path = f'./test/buy/{code}.csv'
        else:
            csv_file_path = f'./test/sell/{code}.csv'
        df.to_csv(csv_file_path, index=True)  # index=False表示不保存DataFrame的索引
        print(f'数据已保存至 {csv_file_path}')
        return csv_file_path

    def save_data2(self, code):
        df = get_price(code, frequency='1m', count=500)  # 支持'1m','5m','15m','30m','60m'
        print(df)
        last_index = df.index[-1]
        df = df.drop(last_index)
        df.index.name = 'time'
        # 重命名列
        df.rename(columns={'close': 'Price', 'volume': 'Volume'}, inplace=True)

        csv_file_path = f'{code}.csv'
        df.to_csv(csv_file_path, index=True)  # index=False表示不保存DataFrame的索引
        print(f'数据已保存至 {csv_file_path}')

    def send_message_to_wechat(self, code):
        # 你的Server酱API密钥
        SCKEY = 'SCT205498TVznAyJOnylNd4bE42tWSz3mp'

        # 发送消息到钉钉的URL
        url = f'https://sctapi.ftqq.com/{SCKEY}.send?channel=9'

        # 要发送的消息内容，你可以根据Server酱的文档来格式化这个JSON
        # 这里只是一个简单的示例
        data = {
            "text": f"code:{code}提示卖点"
        }

        # 发送POST请求
        response = requests.post(url, data=data)

        # 打印响应结果，检查是否发送成功
        print(response.text)

    def send_message2_wechat(self, title):
        SCKEY = 'SCT205498TVznAyJOnylNd4bE42tWSz3mp'

        # 发送消息到钉钉的URL
        url = f'https://sctapi.ftqq.com/{SCKEY}.send?channel=9'

        # 要发送的消息内容，你可以根据Server酱的文档来格式化这个JSON
        # 这里只是一个简单的示例
        data = {
            "text": f"{title}",
        }

        # 发送POST请求
        response = requests.post(url, data=data)

    def train_use_new_file(self, new_file):

        pass

    def get_all_test_csv(self, type):
        # 指定目录路径
        directory = 'test'
        if type == self.BUY_POINT:
            directory = 'test/buy'
        if type == self.SELL_POINT:
            directory = 'test/sell'
        # 初始化一个空列表来存储CSV文件的路径
        csv_files = []

        # 遍历目录中的所有文件和子目录
        for root, dirs, files in os.walk(directory):
            for file in files:
                # 检查文件扩展名是否为.csv
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))

        # 打印找到的CSV文件路径
        print(csv_files)
        return csv_files

    def retrain_with_all_data(self):
        file_names = self.get_all_test_csv(self.SELL_POINT)
        self.load_test_case(file_names)
        self.train_model()
        self.load_model()

    def train_with_all_buy_data(self):
        file_names = self.get_all_test_csv(self.BUY_POINT)
        self.load_test_case(file_names)
        self.train_buy_model()

    def train_buy_model(self):
        X = self.data[self.features]
        y = self.data[self.BUY_POINT]  # 卖点标签（1 为卖点，0 为非卖点）
        X_train = X
        y_train = y
        # 划分训练集和测试集

        # 创建 XGBoost 模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # model = xgb.XGBClassifier()
        # 训练模型
        model.fit(X_train, y_train)

        # 保存模型到文件
        self.buy_model_file = 'buy_point_model.pkl'
        pickle.dump(model, open(self.buy_model_file, 'wb'))

    def load_buy_model_predict(self, x_input):
        loaded_model = pickle.load(open(self.buy_model_file, 'rb'))

        # 使用加载的模型进行预测
        predictions = loaded_model.predict(x_input)
        return predictions

    def code_buy_point(self, code):
        test_code = code
        self.save_data2(test_code)
        data_test = self.data_convert(f'{test_code}.csv')
        # 输入特征和目标变量
        X_test = data_test[self.features]

        # 评估模型
        from sklearn.metrics import accuracy_score

        y_pred = self.load_buy_model_predict(X_test)
        if y_pred[-1] == 1:
            self.send_message_to_wechat(code)
        # print(y_test)
        # 输出预测结果与对应的时间
        point = 'Predicted_Buy_Point'
        data_test[('%s' % point)] = y_pred  # 将预测的卖点添加到数据中

        # 只显示预测为卖点（Sell_Point = 1）的记录
        sell_points = data_test[data_test[point] == 1]
        # 打印时间、卖点预测值和实际标签
        print('code:', code)
        print(sell_points[['time', point]].reset_index())

    def get_time_series_data(self, file_name, target_time, time_window_minutes=60):
        # 读取 CSV 文件
        df = pd.read_csv(file_name)

        # 将 'time' 列转换为 datetime 格式
        target_time = datetime.strftime(target_time, '%Y-%m-%d %H:%M:%S')
        # 确保目标时间存在于数据中
        if target_time not in df['time'].values:
            raise ValueError(f"Target time {target_time} not found in the data.")

        # 获取目标时间的索引位置
        target_index = df[df['time'] == target_time].index[0]

        # 计算60条数据的起始索引
        start_index = max(target_index - time_window_minutes + 1, 0)

        # 获取从目标时间前60条数据
        result_df = df.iloc[start_index:target_index + 1]

        return result_df

    def load_model(self):
        self.loaded_model = pickle.load(open(self.sell_model_file, 'rb'))
