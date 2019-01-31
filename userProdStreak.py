# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import os
from instakartDataLoad import *
from utils import *

'''
Calculates (user, product) order_streak for the last n orders.

- abs(order_streak) is length of streak
- sgn(order_streak) encodes type of streak (non-ordered vs ordered)
'''

data_dir = '../data/instakart'


def _apply_parallel(df_groups, _func):
    nthreads = multiprocessing.cpu_count() >> 1
    print('nthreads: {}'.format(nthreads))
    res = Parallel(n_jobs=nthreads)(delayed(_func)(grp.copy())
                                    for _, grp in df_groups)
    return pd.concat(res)


def _add_order_streak(df):
    tmp = df.copy()
    tmp.user_id = 1

    UP = tmp.pivot(index='product_id', columns='order_number').fillna(-1)
    UP.columns = UP.columns.droplevel(0)

    x = np.abs(UP.diff(axis=1).fillna(2)).values[:, ::-1]
    df.set_index('product_id', inplace=True)
    df['order_streak'] = np.multiply(np.argmax(x, axis=1) + 1, UP.iloc[:, -1])
    df.reset_index(drop=False, inplace=True)
    return df


if __name__ == '__main__':
    print('Getting priors...')
    priors = get_priors(data_dir)
    print('Priors loaded {}'.format(priors.shape))
    print('Getting orders...')
    orders = get_orders(data_dir)

    print('Orders: {}'.format(orders.shape))
    print('Take only recent 5 orders per user!')
    orders = orders.groupby(['user_id']).tail(5 + 1)
    print('orders: {}'.format(orders.shape))

    priors = orders.merge(priors, how='inner', on='order_id')
    priors = priors[['user_id', 'product_id', 'order_number']]
    print('priors: {}'.format(priors.shape))

    user_groups = priors.groupby('user_id')
    df = _apply_parallel(user_groups, _add_order_streak)

    df = df.drop('order_number', axis=1).drop_duplicates()\
           .reset_index(drop=True)

    print('\nSample //')
    df = df[['user_id', 'product_id', 'order_streak']]
    print(df.head(n=10))
    print(df.shape)
    df.to_pickle(os.path.join(data_dir, 'feat/orderStreak.pkl'))
    print('\nWritten')
