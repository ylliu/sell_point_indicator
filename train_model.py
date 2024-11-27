import pickle

import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from Ashare import get_price


class TrainModel:
    def __init__(self):
        self.features = ['Price', 'Volume', 'SMA5', 'Price_change', 'Volume_change']

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

        # 训练模型
        model.fit(X_train, y_train)

        # 保存模型到文件
        self.model_file = 'xgboost_model.pkl'
        pickle.dump(model, open(self.model_file, 'wb'))

    def load_model_predict(self, x_input):
        loaded_model = pickle.load(open(self.model_file, 'rb'))

        # 使用加载的模型进行预测
        predictions = loaded_model.predict(x_input)
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

    def save_data(self, code, sell_start, sell_end):
        df = get_price(code, frequency='1m', count=241)  # 支持'1m','5m','15m','30m','60m'
        last_index = df.index[-1]
        df = df.drop(last_index)
        df.index.name = 'time'
        # 重命名列
        df.rename(columns={'close': 'Price', 'volume': 'Volume'}, inplace=True)
        sell_mask = (df.index >= sell_start) & (df.index <= sell_end)
        # 保存DataFrame为CSV文件
        df['Sell_Point'] = sell_mask.astype(int)
        csv_file_path = f'{code}.csv'
        df.to_csv(csv_file_path, index=True)  # index=False表示不保存DataFrame的索引
        print(f'数据已保存至 {csv_file_path}')

    def save_data2(self, code):
        df = get_price(code, frequency='1m', count=241)  # 支持'1m','5m','15m','30m','60m'
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

    def send_message2_wechat(self, title, content):
        SCKEY = 'SCT205498TVznAyJOnylNd4bE42tWSz3mp'

        # 发送消息到钉钉的URL
        url = f'https://sctapi.ftqq.com/{SCKEY}.send?channel=9'

        # 要发送的消息内容，你可以根据Server酱的文档来格式化这个JSON
        # 这里只是一个简单的示例
        data = {
            "text": f"{title}",
            "desp": f"{content}"
        }

        # 发送POST请求
        response = requests.post(url, data=data)