#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import time
from typing import Any

nesting_level = 0


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(str(space) + " " + str(entry))


def timeit(method, start_log=None):
    def wrapper(*args, **kw):
        global nesting_level

        log("Start [" + str(method.__name__) + "]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log("End   [" + str(method.__name__) + "]. Time elapsed: "+ "%.2f" % ((end_time - start_time)) + " sec.")
        return result

    return wrapper
