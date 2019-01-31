# -*- coding: utf-8 -*-

from __future__ import print_function, division
from tabulate import tabulate
from instakartDataLoad import *
from utils import *

'''
user X aisle features -
# 'ua_orders', 'ua_orders_ratio', 'ua_average_pos_in_cart',
# 'ua_reorder_rate', 'ua_orders_since_last'
'''

data_dir = '../data/instakart'


@timer
def _ua_total_n(df, col):
    return df.groupby(col).size()


@timer
def _ua_total_n2(df):
    return df.groupby(['user_id', 'aisle_id'])['order_id'].nunique()


@timer
def _ua_avg_cart(df):
    return df.groupby(['user_id', 'aisle_id'])['add_to_cart_order'].mean()


@timer
def _ua_n_ord_since(df):
    return df.groupby(['user_id', 'aisle_id'])['order_number'].max()


if __name__ == '__main__':
    print('Getting priors...')
    priors = get_priors(data_dir)
    print('Priors loaded {}'.format(priors.shape))
    print('Getting orders...')
    orders = get_orders(data_dir)
    print('Orders loaded {}'.format(orders.shape))
    prod_map, _, _ = get_mappings(data_dir)
    print('Prod-Aisle map loaded {}'.format(prod_map.shape))
    priors = pd.merge(priors, orders[['order_id', 'user_id', 'order_number']],
                      how='left', on='order_id')
    priors = pd.merge(priors, prod_map[['product_id', 'aisle_id']],
                      how='left', on='product_id')

    # user features
    print('\nAdding user - total orders...')
    usr_n_orders = _ua_total_n(orders[['user_id', 'order_id']], 'user_id')

    # UP features
    print('\nAdding userXaisle total orders...')
    ua_n_orders = pd.DataFrame(_ua_total_n2(priors)).reset_index()
    ua_n_orders.columns = ['user_id', 'aisle_id', 'ua_n_orders']
    print('\nAdding userXaisle re ratio...')
    df_user_n_orders = pd.DataFrame(usr_n_orders).reset_index()
    df_user_n_orders.columns = ['user_id', 'user_n_orders']
    ua_n_orders = pd.merge(ua_n_orders, df_user_n_orders, how='inner',
                           on='user_id')
    ua_n_orders['ua_re_ratio'] = \
        ua_n_orders.ua_n_orders / ua_n_orders.user_n_orders
    ua_n_orders = ua_n_orders.drop('user_n_orders', axis=1)
    print('\nAdding userXaisle avg cart pos...')
    ua_avg_a2co = _ua_avg_cart(priors).reset_index()
    ua_avg_a2co.columns = ['user_id', 'aisle_id', 'ua_avg_a2co']
    print('\nAdding userXaisle n orders since last...')
    ua_n_ord_since = _ua_n_ord_since(priors).reset_index()
    ua_n_ord_since.columns = ['user_id', 'aisle_id', 'ua_max_ordno']
    ua_n_ord_since = pd.merge(ua_n_ord_since, df_user_n_orders, how='inner',
                              on='user_id')
    ua_n_ord_since['ua_n_ord_since'] = \
        ua_n_ord_since.user_n_orders - ua_n_ord_since.ua_max_ordno

    print('\nFeature description //')
    desc = pd.concat([ua_n_orders.ua_n_orders.describe(),
                      ua_n_orders.ua_re_ratio.describe(),
                      ua_avg_a2co.ua_avg_a2co.describe(),
                      ua_n_ord_since.ua_n_ord_since.describe()], axis=1)
    desc.columns = ['ua_n_orders', 'ua_re_ratio', 'ua_avg_a2co',
                    'ua_n_ord_since']
    print(tabulate(desc, headers='keys', tablefmt='psql'))

    ua_feat = pd.merge(ua_n_orders, ua_avg_a2co, how='inner',
                       on=['user_id', 'aisle_id'])
    ua_feat = \
        pd.merge(ua_feat, ua_n_ord_since[['user_id', 'aisle_id',
                                         'ua_n_ord_since']],
                 how='left', on=['user_id', 'aisle_id'])

    # Sample
    print('\nSample out //')
    print(tabulate(ua_feat.head(), headers='keys', tablefmt='psql'))

    # Output
    ua_feat.to_pickle(os.path.join(data_dir, 'feat/uaFeat.pkl'))
    print('\nWritten!')
