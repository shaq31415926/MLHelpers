# -*- coding: utf-8 -*-
import pandas as pd


def dummy_var(data, var):
    """specify dataset and variable name to hot encode"""
    one_hot = pd.get_dummies(data[var], prefix=var)
    data.drop(var, axis=1, inplace=True)
    data = data.join(one_hot)

    return data
