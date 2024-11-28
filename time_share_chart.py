import pandas as pd
import numpy as np

# 模拟市场开盘和收盘时间
start_time = pd.Timestamp("2024-01-01 09:30")
end_time = pd.Timestamp("2024-01-01 15:00")

# 创建每分钟的时间序列
time_index = pd.date_range(start=start_time, end=end_time, freq="T")

# 模拟股价数据
np.random.seed(42)  # 固定随机种子以便结果可复现
prices = np.cumsum(np.random.randn(len(time_index)) * 0.2 + 0.01) + 100

# 模拟成交量数据
volumes = np.random.randint(100, 1000, size=len(time_index))

# 构建 DataFrame
data = pd.DataFrame({
    "time": time_index,
    "price": prices,
    "volume": volumes
})

# 保存数据为 JSON 格式
data.to_json("timeshare_data.json", orient="records")
print("模拟数据已生成。")
