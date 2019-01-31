# -*- coding: utf-8 -*-

from __future__ import print_function, division
from tabulate import tabulate
from instakartDataLoad import *
from utils import *

'''
product features -
# 'prod_n_orders', 'prod_re', 'prod_re_ratio'
'''

data_dir = '../data/instakart'


@timer
def _up_total_n(df, col):
    return df.groupby(col).size()


@timer
def _prod_re(df):
    return df.groupby('product_id')['reordered'].sum()


if __name__ == '__main__':
    print('Getting priors...')
    priors = get_priors(data_dir)
    print('Priors loaded {}'.format(priors.shape))
    print('Getting orders...')
    orders = get_orders(data_dir)
    print('Orders loaded {}'.format(orders.shape))
    priors = pd.merge(priors, orders[['order_id', 'user_id']], how='left',
                      on='order_id')

    # product_features
    print('\nAdding prod - total orders...')
    prod_n_orders = _up_total_n(priors, 'product_id')
    print('\nAdding prod - total reorders...')
    prod_re = _prod_re(priors)
    print('\nAdding prod - reordered ratio...')
    prod_re_ratio = prod_re / prod_n_orders
    print('\nFeature description //')
    desc = pd.concat([prod_n_orders.describe(), prod_re.describe(),
                      prod_re_ratio.describe()], axis=1)
    desc.columns = ['prod_n_orders', 'prod_re', 'prod_re_ratio']
    print(tabulate(desc, headers='keys', tablefmt='psql'))
    prod_feat = concat_same_key(desc.columns, 'product_id',
                                prod_n_orders, prod_re, prod_re_ratio)

    # Sample
    print('\nSample out //')
    print(tabulate(prod_feat.head(), headers='keys', tablefmt='psql'))

    # Output
    prod_feat.to_pickle(os.path.join(data_dir, 'feat/prodFeat.pkl'))
    print('\nWritten!')
