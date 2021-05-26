import time


class TimeUtils:
    def __init__(self):
        self.format = "%Y-%m-%d-%H-%M-%S"

    def get_format_time(self):
        return time.strftime(self.format)