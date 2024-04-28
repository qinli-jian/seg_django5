# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
Project: seg_django_5
Name: snowflake_idgen
Description:
Author: qinlinjian
Datetime: 4/21/2024 10:00 AM
Product: PyCharm
-----------------------------------------------------------
"""
__author__ = "qinlinjian"
__version__ = "1.0.0"

import time
import logging



# 64 位 id 的划分,通常机器位和数据位各为 5 位
WORKER_ID_BITS = 5  # 机器位
DATACENTER_ID_BITS = 5  # 数据位
SEQUENCE_BITS = 12  # 循环位

# 最大取值计算,计算机中负数表示为他的补码
MAX_WORKER_ID = -1 ^ (-1 << WORKER_ID_BITS)  # 2**5 -1 =31
MAX_DATACENTER_ID = -1 ^ (-1 << DATACENTER_ID_BITS)

# 移位偏移计算
WORKER_ID_SHIFT = SEQUENCE_BITS
DATACENTER_ID_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS
TIMESTAMP_LEFT_SHIFT = SEQUENCE_BITS + WORKER_ID_BITS + DATACENTER_ID_BITS

# X序号循环掩码
SEQUENCE_MASK = -1 ^ (-1 << SEQUENCE_BITS)

# Twitter 元年时间戳
TWEPOCH = 1288834974657

logger = logging.getLogger('雪花算法')


class IdWorker(object):
    '''
    用于生成IDS.
    '''

    def __init__(self, datacenter_id, worker_id, sequence=0):
        '''
        初始化方法
        :param datacenter_id:数据id
        :param worker_id:机器id
        :param sequence:序列码
        '''
        if worker_id > MAX_WORKER_ID or worker_id < 0:
            raise ValueError('worker_id 值越界')
        if datacenter_id > MAX_DATACENTER_ID or datacenter_id < 0:
            raise ValueError('datacenter_id 值越界')

        self.worker_id = worker_id
        self.datacenter_id = datacenter_id
        self.sequence = sequence

        self.last_timestamp = -1  # 上次计算的时间戳

    def _gen_timestamp(self):
        '''
        生成整数时间戳。
        :return:
        '''
        return int(time.time() * 1000)

    def get_id(self):
        '''
        获取新的ID.
        :return:
        '''
        # 获取当前时间戳
        timestamp = self._gen_timestamp()

        # 时钟回拨的情况
        if timestamp < self.last_timestamp:
            logging.error('clock is moving backwards. Rejecting requests util {}'.format(self.last_timestamp))
            return -1

        if timestamp == self.last_timestamp:
            # 同一毫秒的处理。
            self.sequence = (self.sequence + 1) & SEQUENCE_MASK
            if self.sequence == 0:
                timestamp = self._til_next_millis(self.last_timestamp)
        else:
            self.sequence = 0

        self.last_timestamp = timestamp

        new_id = (((timestamp - TWEPOCH) << TIMESTAMP_LEFT_SHIFT) | (self.datacenter_id << DATACENTER_ID_SHIFT) | (
                    self.worker_id << WORKER_ID_SHIFT)) | self.sequence
        return str(new_id)

    def _til_next_millis(self, last_timestamp):
        '''
        等到下一毫秒。
        :param last_timestamp:
        :return:
        '''
        timestamp = self._gen_timestamp()
        while timestamp <= last_timestamp:
            timestamp = self._gen_timestamp()
        return timestamp


if __name__ == '__main__':
    worker = IdWorker(1, 2, 0)
    print(worker.get_id())
