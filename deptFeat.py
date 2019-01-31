# -*- coding: utf-8 -*-

from __future__ import print_function, division
from tabulate import tabulate
from instakartDataLoad import *
from utils import *

'''
department features -
# 'department_n_orders', 'department_re', 'department_re_ratio'
'''

data_dir = '../data/instakart'


@timer
def _up_total_n(df, col):
    return df.groupby(col).size()


@timer
def _department_re(df):
    return df.groupby('department_id')['reordered'].sum()


if __name__ == '__main__':
    print('Getting priors...')
    priors = get_priors(data_dir)
    print('Priors loaded {}'.format(priors.shape))
    print('Getting orders...')
    orders = get_orders(data_dir)
    print('Orders loaded {}'.format(orders.shape))
    prod_map, _, _ = get_mappings(data_dir)
    print('Prod-department map loaded {}'.format(prod_map.shape))
    priors = pd.merge(priors, orders[['order_id', 'user_id']], how='left',
                      on='order_id')
    priors = pd.merge(priors, prod_map[['product_id', 'department_id']],
                      how='left', on='product_id')

    # departmentuct_features
    print('\nAdding department - total orders...')
    department_n_orders = _up_total_n(priors, 'department_id')
    print('\nAdding department - total reorders...')
    department_re = _department_re(priors)
    print('\nAdding department - reordered ratio...')
    department_re_ratio = department_re / department_n_orders
    print('\nFeature description //')
    desc = pd.concat([department_n_orders.describe(), department_re.describe(),
                      department_re_ratio.describe()], axis=1)
    desc.columns = ['department_n_orders', 'department_re',
                    'department_re_ratio']
    print(tabulate(desc, headers='keys', tablefmt='psql'))
    department_feat = concat_same_key(desc.columns, 'department_id',
                                      department_n_orders, department_re,
                                      department_re_ratio)

    # Sample
    print('\nSample out //')
    print(tabulate(department_feat.head(), headers='keys', tablefmt='psql'))

    # Output
    department_feat.to_pickle(os.path.join(data_dir,
                                           'feat/departmentFeat.pkl'))
    print('\nWritten!')
