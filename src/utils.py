# -*- coding: utf-8 -*-

import pandas as pd
import time


def timer(fn):
    def wrapper(*args):
        st = time.time()
        result = fn(*args)
        et = time.time()
        total = et - st
        print('[{0:.1f}s]'.format(total))
        return result
    return wrapper


@timer
def concat_same_key(cols, _key, *dfs):
    cdf = pd.DataFrame()
    for df, col in zip(dfs, cols):
        _df = df.reset_index()
        _df.columns = [_key, col]
        _df.sort_values(by=_key, inplace=True)
        if cdf.shape[0] == 0:
            cdf = _df
        else:
            _df = _df.drop(_key, axis=1)
            cdf = pd.concat([cdf, _df], axis=1)
    return cdf
