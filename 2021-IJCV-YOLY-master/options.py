"""
Created on 2020/1/14

@author: Boyun Li
"""

import argparse
import os
parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--name', type=str, default="school_output")
parser.add_argument('--datasets', type=str, default="school")

parser.add_argument('--clip', type=bool, default=True)
parser.add_argument('--num_iter', type=int, default=10) #迭代次数
parser.add_argument('--learning_rate', type=float, default=0.001) #学习率，控制优化器的学习步长

options = parser.parse_args()

# 创建输出目录路径：使用相对路径和os.path.join()实现跨平台兼容
output_path = os.path.join("output", options.datasets, options.name)

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

# 创建日志目录路径
log_dir = "log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# 添加路径配置函数，供其他模块使用
def get_output_path(datasets, name):
    """获取输出路径"""
    return os.path.join("output", datasets, name)

def get_log_path(datasets, name):
    """获取日志文件路径"""
    return os.path.join("log", f"{datasets}_{name}.txt")

def get_process_log_path():
    """获取处理过程日志路径"""
    return os.path.join("log", "process.txt")


