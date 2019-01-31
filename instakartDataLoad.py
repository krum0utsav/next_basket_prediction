# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import pandas as pd
import numpy as np


# Priors
def get_priors(data_dir):
    """
    Historical user orders
    """
    order_dtypes = {'order_id': np.int32,
                    'product_id': np.int32,
                    'add_to_cart_order': np.uint8,
                    'reordered': np.int8}
    order_prior = \
        pd.read_csv(os.path.join(data_dir, 'order_products__prior.csv'),
                    dtype=order_dtypes)
    return order_prior


# Train
def get_train(data_dir):
    """
    Last order that is used for Training models
    """
    order_dtypes = {'order_id': np.int32,
                    'product_id': np.int32,
                    'add_to_cart_order': np.uint8,
                    'reordered': np.int8}
    order_train = \
        pd.read_csv(os.path.join(data_dir, 'order_products__train.csv'),
                    dtype=order_dtypes)
    return order_train


# Mappings
def get_mappings(data_dir):
    """
    Product mappings up the hierarchy
    """
    prod_map = pd.read_csv(os.path.join(data_dir, 'products.csv'),
                           dtype={'product_id': np.int32,
                                  'product_name': np.str,
                                  'aisle_id': np.uint8,
                                  'department_id': np.uint8})
    dep_map = pd.read_csv(os.path.join(data_dir, 'departments.csv'))
    aisle_map = pd.read_csv(os.path.join(data_dir, 'aisles.csv'))
    return prod_map, dep_map, aisle_map


# Orders
def get_orders(data_dir):
    """
    Orders split and other info
    """
    orders_dtypes = {'order_id': np.int32, 'user_id': np.int32,
                     'eval_set': 'category', 'order_number': np.int8,
                     'order_dow': np.int8, 'order_hour_of_day': np.int8,
                     'days_since_prior_order': np.float32}
    orders = pd.read_csv(os.path.join(data_dir, 'orders.csv'),
                         dtype=orders_dtypes)
    return orders
