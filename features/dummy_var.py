import pandas as pd


def dummy_var(data, var):
    """specify dataset and list of variables to hot encode"""
    one_hot = pd.get_dummies(data[var], prefix=var)
    data_new = data.drop(var, axis=1)
    data_new = data_new.join(one_hot)

    return data_new