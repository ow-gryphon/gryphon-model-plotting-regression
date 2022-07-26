import numpy as np
import pandas as pd

def add_cumulative_sum_by_subject(frame, var_names, subject_id, prefix = "cumsum_", na_fill = None):
    """
    Calculates cumulative sum of variables as new columns in a Pandas DataFrame

    :param frame: a single sorted Pandas DataFrame containing the subject identifier,
    and relevant variables to calculate function sum. This needs to be sorted by subject and date
    :param subject_id: a single string with the name of the subject identifier variable
    :param var_names: a list of strings containing the names of the variables for which to calculate the cumulative sum
    :param prefix: Prefix (default: cumsum_) for the new variables to be created: {prefix}{variable}
    :return: Pandas DataFrame with new columns containing the cumulative sum with updated names
    """

    # Check existence
    for var_name in var_names:
        assert var_name in frame.columns, "{feat} not in data frame".format(feat=var_name)

    if na_fill is None:
        cumsums = frame.groupby(subject_id)[var_names].cumsum()
    else:
        cumsums = frame.groupby(subject_id)[var_names].fillna(na_fill).cumsum()

    new_names = ["{pf}{var}".format(pf=prefix, var=var_name) for var_name in var_names]
    cumsums.rename(columns=dict(zip(var_names, new_names)), inplace=True)

    return pd.concat([frame, cumsums], axis=1)


def add_cumulative_prod_by_subject(frame, var_names, subject_id, prefix = "cumprod_", na_fill = None):
    """
    Calculates cumulative sum of variables as new columns in a Pandas DataFrame

    :param frame: a single sorted Pandas DataFrame containing the subject identifier,
    and relevant variables to calculate function sum. This needs to be sorted by subject and date
    :param subject_id: a single string with the name of the subject identifier variable
    :param var_names: a list of strings containing the names of the variables for which to calculate the cumulative sum
    :param prefix: Prefix (default: cumprod_) for the new variables to be created: {prefix}{variable}
    :return: Pandas DataFrame with new columns containing the cumulative prod with updated names
    """

    # Check existence
    for var_name in var_names:
        assert var_name in frame.columns, "{feat} not in data frame".format(feat=var_name)

    if na_fill is None:
        cumprods = frame.groupby(subject_id)[var_names].cumprod()
    else:
        cumprods = frame.groupby(subject_id)[var_names].fillna(na_fill).cumprod()

    new_names = ["{pf}{var}".format(pf=prefix, var=var_name) for var_name in var_names]
    cumprods.rename(columns=dict(zip(var_names, new_names)), inplace=True)

    return pd.concat([frame, cumprods], axis=1)
