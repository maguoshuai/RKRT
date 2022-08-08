# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 下午12:01
# @Author  : nevermore.huachi
# @File    : tools.py
# @Software: PyCharm

def dict2seqlist(ll):
    list_values = [i for i in ll.values()]
    list_keys = [i for i in ll.keys()]
    return sorted(zip(list_keys, list_values), key=lambda x: x[0], reverse=True)