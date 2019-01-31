# -*- coding: utf-8 -*-

from __future__ import print_function, division
from tabulate import tabulate
from instakartDataLoad import *
from utils import *

'''
user X department features -
# 'ud_orders', 'ud_orders_ratio', 'ud_average_pos_in_cart',
# 'ud_reorder_rate', 'ud_orders_since_last'
'''

data_dir = '../data/instakart'


@timer
def _ud_total_n(df, col):
    return df.groupby(col).size()


@timer
def _ud_total_n2(df):
    return df.groupby(['user_id', 'department_id'])['order_id'].nunique()


@timer
def _ud_avg_cart(df):
    return df.groupby(['user_id', 'department_id'])['add_to_cart_order'].mean()


@timer
def _ud_n_ord_since(df):
    return df.groupby(['user_id', 'department_id'])['order_number'].max()


if __name__ == '__main__':
    print('Getting priors...')
    priors = get_priors(data_dir)
    print('Priors loaded {}'.format(priors.shape))
    print('Getting orders...')
    orders = get_orders(data_dir)
    print('Orders loaded {}'.format(orders.shape))
    prod_map, _, _ = get_mappings(data_dir)
    print('Prod-Department map loaded {}'.format(prod_map.shape))
    priors = pd.merge(priors, orders[['order_id', 'user_id', 'order_number']],
                      how='left', on='order_id')
    priors = pd.merge(priors, prod_map[['product_id', 'department_id']],
                      how='left', on='product_id')

    # user features
    print('\nAdding user - total orders...')
    usr_n_orders = _ud_total_n(orders[['user_id', 'order_id']], 'user_id')

    # UP features
    print('\nAdding userXdepartment total orders...')
    ud_n_orders = pd.DataFrame(_ud_total_n2(priors)).reset_index()
    ud_n_orders.columns = ['user_id', 'department_id', 'ud_n_orders']
    print('\nAdding userXdepartment re ratio...')
    df_user_n_orders = pd.DataFrame(usr_n_orders).reset_index()
    df_user_n_orders.columns = ['user_id', 'user_n_orders']
    ud_n_orders = pd.merge(ud_n_orders, df_user_n_orders, how='inner',
                           on='user_id')
    ud_n_orders['ud_re_ratio'] = \
        ud_n_orders.ud_n_orders / ud_n_orders.user_n_orders
    ud_n_orders = ud_n_orders.drop('user_n_orders', axis=1)
    print('\nAdding userXdepartment avg cart pos...')
    ud_avg_a2co = _ud_avg_cart(priors).reset_index()
    ud_avg_a2co.columns = ['user_id', 'department_id', 'ud_avg_a2co']
    print('\nAdding userXdepartment n orders since last...')
    ud_n_ord_since = _ud_n_ord_since(priors).reset_index()
    ud_n_ord_since.columns = ['user_id', 'department_id', 'ud_max_ordno']
    ud_n_ord_since = pd.merge(ud_n_ord_since, df_user_n_orders, how='inner',
                              on='user_id')
    ud_n_ord_since['ud_n_ord_since'] = \
        ud_n_ord_since.user_n_orders - ud_n_ord_since.ud_max_ordno

    print('\nFeature description //')
    desc = pd.concat([ud_n_orders.ud_n_orders.describe(),
                      ud_n_orders.ud_re_ratio.describe(),
                      ud_avg_a2co.ud_avg_a2co.describe(),
                      ud_n_ord_since.ud_n_ord_since.describe()], axis=1)
    desc.columns = ['ud_n_orders', 'ud_re_ratio', 'ud_avg_a2co',
                    'ud_n_ord_since']
    print(tabulate(desc, headers='keys', tablefmt='psql'))

    ud_feat = pd.merge(ud_n_orders, ud_avg_a2co, how='inner',
                       on=['user_id', 'department_id'])
    ud_feat = \
        pd.merge(ud_feat, ud_n_ord_since[['user_id', 'department_id',
                                         'ud_n_ord_since']],
                 how='left', on=['user_id', 'department_id'])

    # Sample
    print('\nSample out //')
    print(tabulate(ud_feat.head(), headers='keys', tablefmt='psql'))

    # Output
    ud_feat.to_pickle(os.path.join(data_dir, 'feat/udFeat.pkl'))
    print('\nWritten!')
