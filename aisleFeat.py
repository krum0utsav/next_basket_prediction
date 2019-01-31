# -*- coding: utf-8 -*-

from __future__ import print_function, division
from tabulate import tabulate
from instakartDataLoad import *
from utils import *

'''
aisle features -
# 'aisle_n_orders', 'aisle_re', 'aisle_re_ratio'
'''

data_dir = '../data/instakart'


@timer
def _up_total_n(df, col):
    return df.groupby(col).size()


@timer
def _aisle_re(df):
    return df.groupby('aisle_id')['reordered'].sum()


if __name__ == '__main__':
    print('Getting priors...')
    priors = get_priors(data_dir)
    print('Priors loaded {}'.format(priors.shape))
    print('Getting orders...')
    orders = get_orders(data_dir)
    print('Orders loaded {}'.format(orders.shape))
    prod_map, _, _ = get_mappings(data_dir)
    print('Prod-Aisle map loaded {}'.format(prod_map.shape))
    priors = pd.merge(priors, orders[['order_id', 'user_id']], how='left',
                      on='order_id')
    priors = pd.merge(priors, prod_map[['product_id', 'aisle_id']],
                      how='left', on='product_id')

    # aisleuct_features
    print('\nAdding aisle - total orders...')
    aisle_n_orders = _up_total_n(priors, 'aisle_id')
    print('\nAdding aisle - total reorders...')
    aisle_re = _aisle_re(priors)
    print('\nAdding aisle - reordered ratio...')
    aisle_re_ratio = aisle_re / aisle_n_orders
    print('\nFeature description //')
    desc = pd.concat([aisle_n_orders.describe(), aisle_re.describe(),
                      aisle_re_ratio.describe()], axis=1)
    desc.columns = ['aisle_n_orders', 'aisle_re', 'aisle_re_ratio']
    print(tabulate(desc, headers='keys', tablefmt='psql'))
    aisle_feat = concat_same_key(desc.columns, 'aisle_id',
                                 aisle_n_orders, aisle_re, aisle_re_ratio)

    # Sample
    print('\nSample out //')
    print(tabulate(aisle_feat.head(), headers='keys', tablefmt='psql'))

    # Output
    aisle_feat.to_pickle(os.path.join(data_dir, 'feat/aisleFeat.pkl'))
    print('\nWritten!')
