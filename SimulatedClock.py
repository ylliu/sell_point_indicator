import datetime


class SimulatedClock:
    def __init__(self, start_time=None):
        # 默认时间从2024-11-29 09:30:00开始
        self.current_time = start_time or datetime.datetime(2024, 11, 29, 9, 30, 0)
        self.end_time = datetime.datetime(2024, 11, 29, 15, 0, 0)  # 模拟结束时间为15:00

    def next(self):
        """
        将时间推进1分钟，如果到达11:30则跳过到13:01
        """
        if self.current_time < self.end_time:
            # 如果当前时间到达11:30，跳到13:01
            if self.current_time.hour == 11 and self.current_time.minute == 30:
                self.current_time = datetime.datetime(2024, 11, 29, 13, 1, 0)  # 跳到13:01
            else:
                self.current_time += datetime.timedelta(minutes=1)
        else:
            print("Reached the end time.")

        return self.current_time

    def get_current_time(self):
        return self.current_time

    def is_time_to_end(self):
        """
        判断是否已经到达结束时间
        """
        return self.current_time >= self.end_time


# 示例使用
if __name__ == "__main__":
    clock = SimulatedClock()

    # 模拟每分钟推进一次
    while not clock.is_time_to_end():
        print("Current Time:", clock.get_current_time().strftime('%Y-%m-%d %H:%M:%S'))
        clock.next()  # 推进时间
