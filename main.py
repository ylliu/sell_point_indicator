import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Ashare import get_price


def data_convert(file_name):
    data = pd.read_csv(file_name)
    # 预处理：如去除缺失值、归一化等
    data.fillna(method='ffill', inplace=True)
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


# 读取多个 CSV 文件并合并
def load_and_merge_data(file_names):
    all_data = []
    for file in file_names:
        data = data_convert(file)
        all_data.append(data)
    # 合并所有数据
    merged_data = pd.concat(all_data, axis=0, ignore_index=True)
    return merged_data


def save_data(code, sell_start, sell_end):
    df = get_price(code, frequency='1m', count=241)  # 支持'1m','5m','15m','30m','60m'
    print(df)
    last_index = df.index[-1]
    df = df.drop(last_index)
    df.index.name = 'time'
    # 重命名列
    df.rename(columns={'close': 'Price', 'volume': 'Volume'}, inplace=True)
    print(df.index)
    sell_mask = (df.index >= sell_start) & (df.index <= sell_end)
    print('上证指数分钟线\n', df)
    # 保存DataFrame为CSV文件
    df['Sell_Point'] = sell_mask.astype(int)
    csv_file_path = f'{code}.csv'
    df.to_csv(csv_file_path, index=True)  # index=False表示不保存DataFrame的索引
    print(f'数据已保存至 {csv_file_path}')


def save_data2(code):
    df = get_price(code, frequency='1m', count=541)  # 支持'1m','5m','15m','30m','60m'
    # print(df)
    last_index = df.index[-1]
    df = df.drop(last_index)
    df.index.name = 'time'
    # 重命名列
    df.rename(columns={'close': 'Price', 'volume': 'Volume'}, inplace=True)
    # print(df.index)

    # 保存DataFrame为CSV文件

    csv_file_path = f'{code}.csv'
    df.to_csv(csv_file_path, index=True)  # index=False表示不保存DataFrame的索引
    print(f'数据已保存至 {csv_file_path}')


def code_sell_point(code):
    test_code = code
    sell_start_time = ''
    sell_end_time = ''
    # save_data2(test_code)
    data_test = data_convert(f'{test_code}.csv')
    # 输入特征和目标变量
    X_test = data_test[features]

    # 评估模型
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # 输出预测结果与对应的时间
    data_test['Predicted_Sell_Point'] = y_pred  # 将预测的卖点添加到数据中

    # 只显示预测为卖点（Sell_Point = 1）的记录
    sell_points = data_test[data_test['Predicted_Sell_Point'] == 1]

    # 打印时间、卖点预测值和实际标签
    print('code:', code)
    print(sell_points[['time', 'Predicted_Sell_Point']].reset_index())


def test_and_verify(test_code, sell_start_time, sell_end_time):
    # test
    test_code = '300010.XSHE'
    sell_start_time = '2024-11-26 10:02:00'
    sell_end_time = '2024-11-26 10:08:00'
    save_data(test_code, datetime.datetime.strptime(sell_start_time, '%Y-%m-%d %H:%M:%S'),
              datetime.datetime.strptime(sell_end_time, '%Y-%m-%d %H:%M:%S'))
    data_test = data_convert(f'{test_code}.csv')
    # 输入特征和目标变量
    X_test = data_test[features]
    y_test = data_test['Sell_Point']  # 卖点标签（1 为卖点，0 为非卖点）

    # 评估模型
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(X_test)
    # print(y_pred)
    # print(y_test)
    # 输出预测结果与对应的时间
    data_test['Predicted_Sell_Point'] = y_pred  # 将预测的卖点添加到数据中

    # 只显示预测为卖点（Sell_Point = 1）的记录
    sell_points = data_test[data_test['Sell_Point'] == 1]

    # 打印时间、卖点预测值和实际标签
    print(sell_points[['time', 'Predicted_Sell_Point', 'Sell_Point']].reset_index())

    print(f"模型准确率: {accuracy_score(y_test, y_pred):.4f}")


# 文件列表
file_names = ['601360.csv', '301171.csv', '300785.csv', '300450.csv']  # 添加你的文件名
data = load_and_merge_data(file_names)

# 假设你已经有了分时图数据，格式如下
# 数据列：时间, 价格, 成交量, 卖点标签(1为卖点，0为非卖点)

import xgboost as xgb

# 输入特征和目标变量
features = ['Price', 'Volume', 'SMA5', 'Price_change', 'Volume_change']
X = data[features]
y = data['Sell_Point']  # 卖点标签（1 为卖点，0 为非卖点）
X_train = X
y_train = y
# 划分训练集和测试集


# 创建 XGBoost 模型
model = xgb.XGBClassifier()

# 训练模型
model.fit(X_train, y_train)
