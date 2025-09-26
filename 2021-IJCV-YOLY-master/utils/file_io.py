#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:29:27 2020

@author: lester
"""

def write_log(file_name, title, psnr, ssim):
    fp = open(file_name, "a+")
    fp.write(title+ ':\n')
    fp.write('PSNR:%0.6f\n'%psnr)
    fp.write('SSIM:%0.6f\n'%ssim)
    fp.close()

def write_niqelog(file_name, title, niqe):
    fp = open(file_name, "a+")
    fp.write(title+ ':\n')
    fp.write('niqe:%0.6f\n'%niqe)
    fp.close()

def write_process(file_name, title):
    fp = open(file_name, "a+")

    fp.write(title)

    fp.close()


def get_image_content(image_name, file_path='./data/HSTS/haze.txt'):
    # 读取文件内容
    try:
        with open(file_path, 'r') as file:
            # 逐行读取文件
            for line in file:
                # 去除行末的换行符
                line = line.strip()
                # 分割行内容
                if ':' in line:
                    name, content = line.split(':', 1)
                    # 查找匹配的图像名称
                    if name.strip() == image_name:
                        return content.strip()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"An error occurred: {e}"

    return "Image name not found."





