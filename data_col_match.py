import pandas as pd
import numpy as np

import data_cat
import mushroom_class_fix

def find_different_named_columns_list(data1, data2):
    """
    Parameters
    ----------
    *datas: tuple of two pandas.DataFrame with nominal columns

    Returns
    -------
    var name: different_categories, list of lists of strs with two inner lists
        different_categories[0] includes the column names of datas[0] missing from datas[1]
        different_categories[1] includes the column names of datas[1] missing from datas[0]

    Example
    -------
    datas[0].columns.values = ['ex_a', 'ex_b', 'ex_c']
    datas[1].columns.values = ['ex_a', 'ex_d']

    return:
    [['ex_b', 'ex_c'], ['ex_d']]
    """

    datas = [data1, data2]
    data_categories_list_of_dicts = [{}, {}]
    different_categories = [[], []]
    # find columns that do not match (-> not matched attributed values)
    for data_count in range(0, len(datas)):
        for i in range(1, len(datas[data_count].columns)):
            data_categories_list_of_dicts[data_count][datas[data_count].columns.values[i]] = i
    for category in data_categories_list_of_dicts[0].keys():
        if category not in data_categories_list_of_dicts[1].keys():
            different_categories[0].append(category)
    for category in data_categories_list_of_dicts[1].keys():
        if category not in data_categories_list_of_dicts[0].keys():
            different_categories[1].append(category)
    return different_categories


def add_columns_to_match_encoded_datasets(data1, data2, **kwargs):
    """
    Parameters
    ----------
    data1: pandas.DataFrame with encoded nominal columns
    data2: pandas.DataFrame with encoded nominal columns

    **kwargs:
    sort_columns: bool, default=True
    if True reindexes the columns for data1 and data2 by sorting them lexicographically


    Returns
    -------
    list of pandas.DataFrame corresponding containing a variant of data1 and data2 with
    identical column names (the column names missing from each other in data1 and data2 are
    added as only zeros)

    Example
    -------
    data1:              data2:
    'ex_a' 'ex_b'       'ex_a' 'ex_c'
    1       0           1       1
    0       1           1       0

    return (sort_columns=True):
    [data1, data2]

    data1:                  data2:
    'ex_a' 'ex_b' 'ex_c'    'ex_a' 'ex_b' 'ex_c'
    1       0     0         1       0      1
    0       1     0         1       0      0
    """

    if 'sort_columns' not in kwargs:
        kwargs['sort_columns'] = True
    add_categories = find_different_named_columns_list(data1, data2)
    for category in add_categories[1]:
        if category not in data1.columns.values:
            data1[category] = [0] * len(data1)
    for category in add_categories[0]:
        if not category in data2.columns.values:
            data2[category] = [0] * len(data2)
    if kwargs['sort_columns']:
        data1 = data1.reindex(sorted(data1.columns), axis=1)
        data2 = data2.reindex(data1.columns, axis=1)
    return [data1, data2]


def get_datasets_with_identical_columns(data1, data2, **kwargs):
    """
    Parameters
    ----------
    data1: pandas.DataFrame with encoded nominal columns
    data2: pandas.DataFrame with encoded nominal columns

    **kwargs:
    encode: bool, default=False
    if True encodes the columns using encode_data_numerical in mushroom_class_fix.py and
    keeps all columns from both datasets, else the columns are unencoded and simply dropped
    sort_columns: bool, default=True
    if True reindexes the columns for data1 and data2 by sorting them lexicographically


    Returns
    -------
    list of pandas.DataFrame corresponding containing a variant of data1 and data2 with
    identical column names

    Example
    -------
    data1:      data2:
    'a' 'b'     'a' 'c'
     1   0       1   1
     0   1       1   1

    return (sort_columns=True, encode=False):
    [data1, data2]

    data1:      data2:
    'a'         'a'
     1           1
     0           1

     for a return(encode=True) example look at add_columns_to_match_encoded_datasets
    """

    if 'encode' not in kwargs:
        kwargs['encode'] = False
    if 'sort_columns' not in kwargs:
        kwargs['sort_columns'] = True
    datas = []
    drop_categories = find_different_named_columns_list(data1, data2)
    # drop columns from datasets that could not be matched
    if kwargs['encode']:
        datas.append(mushroom_class_fix.encode_data_numerical(data1.drop(drop_categories[0], 1)))
        datas.append(mushroom_class_fix.encode_data_numerical(data2.drop(drop_categories[1], 1)))
        add_columns_to_match_encoded_datasets(datas[0], datas[1], sort_columns=kwargs['sort_columns'])
    else:
        datas.append(data1.drop(drop_categories[0], 1))
        datas.append(data2.drop(drop_categories[1], 1))
    return datas


def get_encoded_column_matchings_dataframe(data1, data2, **kwargs):
    """
    Parameters
    ----------
    data1: pandas.DataFrame with encoded nominal columns
    data2: pandas.DataFrame with encoded nominal columns (some of them should match data1)

    **kwargs:
    columns: list, default=['data1', 'data2']
    name of the columns in the returned pandas.DataFrame

    Returns
    -------
    pandas.DataFrame with one column for data1 and data2, having equal columns in the same row
    and NaN for columns without a match

    Example
    -------
    data1.columns.values = ['ex_a', 'ex_b']
    data2.columns.values = ['ex_a', 'ex_c']

    return:
    data1   data2
    'ex_a'  'ex_a'
    'ex_b'  NaN
    NaN     'ex_c'
    """

    if 'columns' not in kwargs:
        kwargs['columns'] = ['data1', 'data2']
    df = pd.DataFrame(columns=kwargs['columns'], dtype=str)
    datas_matched = get_datasets_with_identical_columns(data1, data2, encode=False)
    assert (all(datas_matched[0].columns.values == datas_matched[1].columns.values)), "All columns must match"
    columns_matched = datas_matched[0].columns.values
    datas_matched_encoded = [mushroom_class_fix.encode_data_numerical(datas_matched[0]),
                             mushroom_class_fix.encode_data_numerical(datas_matched[1])]
    for col_matched in columns_matched:
        for col in datas_matched_encoded[0].columns.values:
            if col_matched in col:
                if col in datas_matched_encoded[1].columns.values:
                    df = df.append({kwargs['columns'][0]: col, kwargs['columns'][1]: col}, ignore_index=True)
                else:
                    df = df.append({kwargs['columns'][0]: col}, ignore_index=True)
        for col in datas_matched_encoded[1].columns.values:
            if col_matched in col:
                if col not in datas_matched_encoded[0].columns.values:
                    df = df.append({kwargs['columns'][1]: col}, ignore_index=True)
    return df


def rename_and_merge_columns_on_dict(data_encoded, rename_encoded_columns_dict, **kwargs):
    """
    Parameters
    ----------
    data_encoded: pandas.DataFrame with numerical columns
    rename_encoded_columns_dict: dict of columns to rename in data_encoded

    **kwargs
    inplace:bool, default=False
    decides if data_encoded is edited inplace or if a copy is returned

    Returns
    -------
    pandas.DataFrame with columns renamed according to rename_encoded_columns_dict, columns that
    share the same name after renaming are merged by adding the columns up

    Example
    -------
    data_encoded:
    x y z
    0 0 1
    1 0 1
    0 1 0

    rename_encoded_columns_dict:
    {'y': 'x'}

    return:
    x z
    0 1
    1 1
    1 0
    """

    if 'inplace' not in kwargs:
        kwargs['inplace'] = False
    if kwargs['inplace']:
        data_copy = data_encoded
    else:
        data_copy = data_encoded.copy()
    data_copy.rename(columns=rename_encoded_columns_dict, inplace=True)
    for col in data_copy.columns:
        df_col = data_copy[col]
        # if column name col appears more than once in data_encoded.columns -> df_col is DataFrame (else it is a Series)
        if isinstance(df_col, pd.DataFrame):
            # add index to identical column names: [cap-shape_x0, cap-shape_x1, ...]
            df_col.columns = [col + str(i) for i in range(0, len(df_col.columns))]
            # drop identical columns col from DataFrame
            data_copy.drop(columns=col, inplace=True)
            # create column of zeros and add the numerical columns up
            col_merged = pd.Series(np.zeros(len(data_copy)), dtype=int)
            for col_indexed in df_col.columns:
                col_merged += df_col[col_indexed]
            data_copy[col] = col_merged
    if kwargs['inplace']:
        data_encoded = data_encoded.reindex(sorted(data_encoded.columns), axis=1)
        return
    else:
        data_copy = data_copy.reindex(sorted(data_copy.columns), axis=1)
    return data_copy


def get_rename_encoded_columns_dict(data_rename_columns):
    """
    Parameters
    ----------
    data_rename_columns: pandas.DataFrame with two columns,
        first the original column names and the renamed column names

    Returns
    -------
    dict of original column names: renamed column names (NaN dropped)

    Example
    -------
    data_rename_columns:
    original    renamed
    'ex_a'      NaN
    'ex_b'      'ex_a'
    'ex_c'      'ex_d'

    return:
    {'ex_b': 'ex_a', 'ex_c': 'ex_d'}
    """

    data_encoded_columns_dict = dict(zip(data_rename_columns.iloc[:, 0], data_rename_columns.iloc[:, 1]))
    pop_keys = []
    for key in data_encoded_columns_dict:
        if pd.isna(data_encoded_columns_dict[key]):
            pop_keys.append(key)
    for key in pop_keys:
        data_encoded_columns_dict.pop(key)
    return data_encoded_columns_dict


def get_identical_values_in_columns(column1, column2):
    """
    Parameters
    ----------
    column1: pandas.Series of str values
    column2: pandas.Series of str values of same length as column1

    Function
    --------
    prints the difference ration between the values of column1 and column2 with each value
    being compared to one with the same index (column1[0] == column2[0]).
    Used to decide if to keep the above or below ring features for the 1987 stem attributes.
    """

    column_difference_series = column1 == column2
    false_count = len(column_difference_series[column_difference_series == False])
    print('column-value-difference-ratio:', false_count / len(column1))


if __name__ == "__main__":
    """
    WARNING: 
    Running this module overwrites the following files in data:
        - secondary_data_encoded_matched.csv
        - 1987_data_encoded_matched.csv
    """

    # import new and original dataset without missing values
    data_new = pd.read_csv(data_cat.FILE_PATH_SECONDARY_NO_MISS, sep=';', header=0, na_values='?', low_memory=False)
    data_og = pd.read_csv(data_cat.FILE_PATH_1987_NO_MISS, sep=';', header=0, dtype=object, na_values='?')

    ## mapping of secondary dataset into original dataset ##
    data_new = data_new.drop(['cap-diameter', 'stem-height', 'stem-width', 'season'], 1)
    # get difference ratio of stalk-surface and stalk-color above/below ring
    for stalk_feat in ['surface', 'color']:
        get_identical_values_in_columns(data_og['stalk-' + stalk_feat + '-above-ring'],
                                        data_og['stalk-' + stalk_feat + '-below-ring'])
    data_og = data_og.drop(['odor', 'gill-size', 'stalk-shape', 'stalk-surface-below-ring',
                            'stalk-color-below-ring', 'population'], 1)
    # renaming columns in original dataset to match the secondary dataset
    data_og = data_og.rename(columns={'bruises': 'does-bruise-or-bleed',
                                      'stalk-root': 'stem-root', 'stalk-surface-above-ring': 'stem-surface',
                                      'stalk-color-above-ring': 'stem-color', 'ring-number': 'has-ring',
                                      'spore-print-color': 'spore-color'})
    # drop columns that are not in both datasets
    data_new, data_og = get_datasets_with_identical_columns(data_new, data_og, encode=False)

    # get csv of encoded column matchings of secondary and original dataset
    matching_dataframe = get_encoded_column_matchings_dataframe(data_new, data_og, columns=['secondary', 'original'])
    matching_dataframe.to_csv(data_cat.FILE_PATH_COLUMN_MATCHING, sep=';', index=False)

    # encode datasets
    data_new = mushroom_class_fix.encode_data_numerical(data_new)
    data_og = mushroom_class_fix.encode_data_numerical(data_og)

    # rename encoded columns according to dataset_variable_encoding_mapping_edited.csv
    data_columns_encoded_mapping = pd.read_csv(data_cat.FILE_PATH_COLUMN_MATCHING_EDITED, sep=';', header=0)
    data_new_rename_encoded_columns_dict = get_rename_encoded_columns_dict(
        data_columns_encoded_mapping[['secondary', 'rename_secondary']])
    data_og_rename_encoded_columns_dict = get_rename_encoded_columns_dict(
        data_columns_encoded_mapping[['original', 'rename_original']])
    data_new = rename_and_merge_columns_on_dict(data_new, data_new_rename_encoded_columns_dict, inplace=False)
    data_og = rename_and_merge_columns_on_dict(data_og, data_og_rename_encoded_columns_dict, inplace=False)
    data_new, data_og = add_columns_to_match_encoded_datasets(data_new, data_og)
    data_og.to_csv(data_cat.FILE_PATH_1987_MATCHED, sep=';', index=False)
    data_new.to_csv(data_cat.FILE_PATH_SECONDARY_MATCHED, sep=';', index=False)
