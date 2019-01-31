# -*- coding: utf-8 -*-

from __future__ import print_function, division
from tabulate import tabulate
from instakartDataLoad import *
from utils import *

'''
user features -
# 'user_total_orders', 'user_total_items', 'total_distinct_items',
# 'user_average_days_between_orders', 'user_average_basket'
'''

data_dir = '../data/instakart'


@timer
def _up_total_n(df, col):
    return df.groupby(col).size()


@timer
def _user_distinct_n(df):
    return df.groupby('user_id')['product_id'].apply(lambda x: len(set(x)))


@timer
def _user_avg_days_btw(df):
    df = df[~df.days_since_prior_order.isnull()]
    avg_days = df.groupby('user_id')['days_since_prior_order'].mean()
    return avg_days


@timer
def _prod_re(df):
    return df.groupby('product_id')['reordered'].sum()


@timer
def _up_total_n2(df):
    return df.groupby(['user_id', 'product_id']).size()


@timer
def _up_avg_cart(df):
    return df.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()


if __name__ == '__main__':
    print('Getting priors...')
    priors = get_priors(data_dir)
    print('Priors loaded {}'.format(priors.shape))
    print('Getting orders...')
    orders = get_orders(data_dir)
    print('Orders loaded {}'.format(orders.shape))
    priors = pd.merge(priors, orders[['order_id', 'user_id']], how='left',
                      on='order_id')

    # user features
    print('\nAdding user - total orders...')
    usr_n_orders = _up_total_n(orders[['user_id', 'order_id']], 'user_id')
    print('\nAdding user - total items...')
    usr_n_items = _up_total_n(priors, 'user_id')
    print('\nAdding user - n_unique items...')
    usr_n_unique_items = _user_distinct_n(priors)
    print('\nAdding user - mean avg days...')
    usr_avg_days = \
        _user_avg_days_btw(orders)
    print('\nAdding user - avg basket...')
    usr_avg_basket = usr_n_items / usr_n_orders
    print('\nFeature description //')
    desc = pd.concat([usr_n_orders.describe(), usr_n_items.describe(),
                      usr_n_unique_items.describe(),
                      usr_avg_days.describe(),
                      usr_avg_basket.describe()], axis=1)
    desc = desc.astype('int')
    desc.columns = ['usr_n_orders', 'usr_n_items', 'usr_n_unique_items',
                    'usr_avg_days', 'usr_avg_basket']
    print(tabulate(desc.T, headers='keys', tablefmt='psql'))
    user_feat = concat_same_key(desc.columns, 'user_id', usr_n_orders,
                                usr_n_items, usr_n_unique_items,
                                usr_avg_days, usr_avg_basket)

    # Sample
    print('\nSample out //')
    print(tabulate(user_feat.head(), headers='keys', tablefmt='psql'))

    # Output
    user_feat.to_pickle(os.path.join(data_dir, 'feat/userFeat.pkl'))
    print('\nWritten!')
