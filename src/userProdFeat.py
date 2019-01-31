# -*- coding: utf-8 -*-

from __future__ import print_function, division
from tabulate import tabulate
from instakartDataLoad import *
from utils import *

'''
user X product features -
# 'up_orders', 'up_orders_ratio', 'up_average_pos_in_cart',
# 'up_reorder_rate', 'up_orders_since_last'
'''

data_dir = '../data/instakart'


@timer
def _up_total_n(df, col):
    return df.groupby(col).size()


@timer
def _up_total_n2(df):
    return df.groupby(['user_id', 'product_id']).size()


@timer
def _up_avg_cart(df):
    return df.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean()


@timer
def _up_n_ord_since(df):
    return df.groupby(['user_id', 'product_id'])['order_number'].max()


if __name__ == '__main__':
    print('Getting priors...')
    priors = get_priors(data_dir)
    print('Priors loaded {}'.format(priors.shape))
    print('Getting orders...')
    orders = get_orders(data_dir)
    print('Orders loaded {}'.format(orders.shape))
    priors = pd.merge(priors, orders[['order_id', 'user_id', 'order_number']],
                      how='left', on='order_id')

    # user features
    print('\nAdding user - total orders...')
    usr_n_orders = _up_total_n(orders[['user_id', 'order_id']], 'user_id')

    # UP features
    print('\nAdding userXproduct total orders...')
    up_n_orders = pd.DataFrame(_up_total_n2(priors)).reset_index()
    up_n_orders.columns = ['user_id', 'product_id', 'up_n_orders']
    print('\nAdding userXproduct re ratio...')
    df_user_n_orders = pd.DataFrame(usr_n_orders).reset_index()
    df_user_n_orders.columns = ['user_id', 'user_n_orders']
    up_n_orders = pd.merge(up_n_orders, df_user_n_orders, how='inner',
                           on='user_id')
    up_n_orders['up_re_ratio'] = \
        up_n_orders.up_n_orders / up_n_orders.user_n_orders
    up_n_orders = up_n_orders.drop('user_n_orders', axis=1)
    print('\nAdding userXproduct avg cart pos...')
    up_avg_a2co = _up_avg_cart(priors).reset_index()
    up_avg_a2co.columns = ['user_id', 'product_id', 'up_avg_a2co']
    print('\nAdding userXproduct n orders since last...')
    up_n_ord_since = _up_n_ord_since(priors).reset_index()
    up_n_ord_since.columns = ['user_id', 'product_id', 'up_max_ordno']
    up_n_ord_since = pd.merge(up_n_ord_since, df_user_n_orders, how='inner',
                              on='user_id')
    up_n_ord_since['up_n_ord_since'] = \
        up_n_ord_since.user_n_orders - up_n_ord_since.up_max_ordno

    print('\nFeature description //')
    desc = pd.concat([up_n_orders.up_n_orders.describe(),
                      up_n_orders.up_re_ratio.describe(),
                      up_avg_a2co.up_avg_a2co.describe(),
                      up_n_ord_since.up_n_ord_since.describe()], axis=1)
    desc.columns = ['up_n_orders', 'up_re_ratio', 'up_avg_a2co',
                    'up_n_ord_since']
    print(tabulate(desc, headers='keys', tablefmt='psql'))

    up_feat = pd.merge(up_n_orders, up_avg_a2co, how='inner',
                       on=['user_id', 'product_id'])
    up_feat = \
        pd.merge(up_feat, up_n_ord_since[['user_id', 'product_id',
                                         'up_n_ord_since']],
                 how='left', on=['user_id', 'product_id'])

    # Sample
    print('\nSample out //')
    print(tabulate(up_feat.head(), headers='keys', tablefmt='psql'))

    # Output
    up_feat.to_pickle(os.path.join(data_dir, 'feat/upFeat.pkl'))
    print('\nWritten!')
