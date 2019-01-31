# -*- coding: utf-8 -*-

from __future__ import print_function, division
import lightgbm as lgb
import pandas as pd
import gc
import os
from instakartDataLoad import *
from utils import *
import warnings
warnings.filterwarnings('ignore')

'''
Local evaluation
'''

data_dir = '../data/instakart'
F1_THRESH = 0.22
SEED = 0


@timer
def _split(df_tr, r=0.1):
    print('\nCreating Train & Validation data...')
    n_users, n_orders = df_tr.user_id.nunique(), df_tr.order_id.nunique()
    print('N users: {}, N orders: {}'.format(n_users, n_orders))
    users = df_tr.user_id.unique()
    np.random.seed(SEED)
    val_users = np.random.choice(users, int(r * len(users)))
    train = df_tr[~df_tr.user_id.isin(val_users)]
    val = df_tr[df_tr.user_id.isin(val_users)]
    train_order_id = train[['order_id', 'product_id', 'reordered']]
    train_X = train.drop(['user_id', 'product_id', 'aisle_id', 'department_id',
                         'order_id', 'reordered'], axis=1)
    train_y = train.reordered
    val_order_id = val[['order_id', 'product_id', 'reordered']]
    val_X = val.drop(['user_id', 'product_id', 'aisle_id', 'department_id',
                      'order_id', 'reordered'], axis=1)
    val_y = val.reordered
    val_n_ord, tr_n_ord = val.order_id.nunique(), train.order_id.nunique()
    print('N train orders: {}, N val orders: {}'.format(tr_n_ord, val_n_ord))
    print('train_X{}, train_y{}, val_X{}, val_y{}'
          .format(train_X.shape, train_y.shape, val_X.shape, val_y.shape))
    return train_X, train_y, val_X, val_y, train_order_id, val_order_id


def _lgb(train_X, train_y, val_X, val_y):
    d_train = lgb.Dataset(train_X, label=train_y)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 80,
        'max_depth': 8,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'verbose': 1
    }
    d_val = lgb.Dataset(val_X, label=val_y)
    bst = lgb.train(params, d_train, early_stopping_rounds=10,
                    num_boost_round=1000, valid_sets=[d_train, d_val])
    val_preds = bst.predict(val_X)
    tr_preds = bst.predict(train_X)
    val_preds_thresh = [1 if p > F1_THRESH else 0 for p in val_preds]
    tr_preds_thresh = [1 if p > F1_THRESH else 0 for p in tr_preds]
    fi = pd.DataFrame({'Feature': bst.feature_name(),
                       'Importance': bst.feature_importance()})\
           .sort_values(by='Importance', ascending=False)
    print('\nFeature Importance //')
    print(fi.head(20))
    return val_preds_thresh, tr_preds_thresh


def _make_baskets(df, col):
    df.product_id = df.product_id.astype('str')
    df_baskets = \
        df.groupby('order_id')\
          .apply(lambda x:
                 ' '.join([p for p, r in zip(x.product_id, x[col])
                           if r == 1]))
    df_baskets = df_baskets.reset_index()
    df_baskets.columns = ['order_id', 'basket']
    df_baskets.order_id = df_baskets.order_id.astype('int32')
    return df_baskets


def _baskets(val_preds, tr_preds, val_oid, tr_oid):
    # Validation
    val_oid['preds'] = val_preds
    df_val_orig = _make_baskets(val_oid, 'reordered')
    df_val_pred = _make_baskets(val_oid, 'preds')
    df_val_pred.columns = ['order_id', 'preds_basket']
    df_val = pd.merge(df_val_orig, df_val_pred, how='inner', on='order_id')
    print('\nValidation //')
    print(df_val.head(10))
    print(df_val.shape)
    # Train
    tr_oid['preds'] = tr_preds
    df_tr_orig = _make_baskets(tr_oid, 'reordered')
    df_tr_pred = _make_baskets(tr_oid, 'preds')
    df_tr_pred.columns = ['order_id', 'preds_basket']
    df_tr = pd.merge(df_tr_orig, df_tr_pred, how='inner', on='order_id')
    print('\nTrain //')
    print(df_tr.head(10))
    print(df_tr.shape)
    return df_tr, df_val


def _F1(df):
    f1 = []
    n = df.shape[0]
    for i, (a, b) in enumerate(zip(df.basket, df.preds_basket)):
        lgt = a.split(' ')
        lpred = b.split(' ')
        rr = (np.intersect1d(lgt, lpred))
        precision = np.float(len(rr)) / len(lpred)
        recall = np.float(len(rr)) / len(lgt)
        denom = precision + recall
        f11 = ((2 * precision * recall) / denom) if denom > 0 else 0
        f1.append(f11)
        if i % 1000 == 0:
            print('{}/{}[{}]'.format(i, n, np.mean(f1)))
    print('Mean F1 score: {}'.format(np.mean(f1)))


if __name__ == '__main__':
    print('Gettint Train data...')
    train = get_train(data_dir)
    print('Getting orders...')
    orders = get_orders(data_dir)
    print('Orders loaded {}'.format(orders.shape))
    prod_map, _, _ = get_mappings(data_dir)
    print('Prod-Aisle-Dept map loaded {}'.format(prod_map.shape))
    train = pd.merge(train, orders[['order_id', 'user_id']], how='left',
                     on='order_id')

    # feat
    user_feat = pd.read_pickle(os.path.join(data_dir, 'feat/userFeat.pkl'))
    prod_feat = pd.read_pickle(os.path.join(data_dir, 'feat/prodFeat.pkl'))
    aisle_feat = pd.read_pickle(os.path.join(data_dir, 'feat/aisleFeat.pkl'))
    dept_feat = \
        pd.read_pickle(os.path.join(data_dir, 'feat/departmentFeat.pkl'))
    up_feat = pd.read_pickle(os.path.join(data_dir, 'feat/upFeat.pkl'))
    ua_feat = pd.read_pickle(os.path.join(data_dir, 'feat/uaFeat.pkl'))
    ud_feat = pd.read_pickle(os.path.join(data_dir, 'feat/udFeat.pkl'))
    ord_streak = pd.read_pickle(os.path.join(data_dir, 'feat/orderStreak.pkl'))

    # all feat
    print('Merging all features...')
    all_feat = pd.merge(up_feat, user_feat, how='left', on='user_id')
    all_feat = \
        pd.merge(all_feat,
                 prod_map[['product_id', 'aisle_id', 'department_id']],
                 how='left', on='product_id')
    all_feat = pd.merge(all_feat, prod_feat, how='left', on='product_id')
    all_feat = pd.merge(all_feat, aisle_feat, how='left', on='aisle_id')
    all_feat = pd.merge(all_feat, dept_feat, how='left', on='department_id')
    all_feat = pd.merge(all_feat, ua_feat, how='left',
                        on=['user_id', 'aisle_id'])
    all_feat = pd.merge(all_feat, ud_feat, how='left',
                        on=['user_id', 'department_id'])
    all_feat = pd.merge(all_feat, ord_streak, how='left',
                        on=['user_id', 'product_id'])
    all_feat.order_streak.fillna(-4, inplace=True)
    print('\nFeatures //')
    print(sorted(all_feat.columns))

    del user_feat, prod_feat, aisle_feat, dept_feat, up_feat, ua_feat, ud_feat
    del ord_streak
    gc.collect()

    # prep data
    print('\nGetting target...')
    _tr = pd.merge(all_feat,
                   train[['user_id', 'product_id', 'reordered']],
                   how='left', on=['user_id', 'product_id'])
    _tr = _tr[_tr.user_id.isin(train.user_id.unique())]
    _tr.reordered.fillna(0, inplace=True)
    user_order = train[['user_id', 'order_id']].drop_duplicates()
    user_order_map = dict(zip(train.user_id, train.order_id))
    _tr['order_id'] = [user_order_map[u] for u in _tr.user_id]
    print('\nSample //')
    print(_tr.head())
    print(_tr.shape)

    # eval
    train_X, train_y, val_X, val_y, tr_oid, val_oid = _split(_tr, 0.1)
    del _tr
    gc.collect()

    print('\n')
    val_preds, tr_preds = _lgb(train_X, train_y, val_X, val_y)
    tr_baskets, val_baskets = _baskets(val_preds, tr_preds, val_oid, tr_oid)
    print('\nValidation F1 //')
    _F1(val_baskets)
    print('\nTraining F1 //')
    _F1(tr_baskets)
