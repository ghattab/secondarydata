import pandas as pd
import numpy as np

import data_cat


mode_dict = {'nb': 'Gaussian Naive Bayes', 'log_reg': 'Logistic regression',
             'lda': 'Linear Discriminant Analysis'}
              # 'qda': 'Quadratic Discriminant Analysis' removed for sketchy results probably caused by collinear variables



def get_variables_missing_dict(data, **kwargs):
    """
    Parameters
    ----------
    data: pandas.DataFrame,
    Input data for evaluating missing values in attributes

    kwargs:
    type: str, default='ratio',
    'ratio': return dict values as ratio, 'abs': return dict values as total numbers
    min: float, default=0.0,
    Only returns attributes with a missing ratio > kwargs['min']
    max: float, default=1.0,
    Only returns attributes with a missing ratio <= kwargs['max']
    print: bool, default=False,
    if True prints the return dict, does print nothing else
    round:int, default=3,
    Decides the number of rounded decimal places for the return dict values

    Returns
    -------
    var name=attributes_missing_dict: dict {str: float or int),
    Dict containing the attributes as keys and the missing values for each attribute
    as values (depending on kwargs['type'])
    """

    if 'type' not in kwargs:
        kwargs['type'] = 'ratio'
    if 'min' not in kwargs:
        kwargs['min'] = 0.0
    if 'max' not in kwargs:
        kwargs['max'] = 1.0
    if 'print' not in kwargs:
        kwargs['print'] = False
    if 'round' not in kwargs:
        kwargs['round'] = 3
    attributes_missing_dict = {}
    missing_categories_count = 0
    for column in data.columns:
        attributes_missing_dict[column] = 0
        attributes_missing_dict[column] += data[column].isnull().sum()
        missing_ratio = attributes_missing_dict[column] / len(data)
        if missing_ratio > kwargs['min'] and missing_ratio <= kwargs['max']:
            if kwargs['type'] == 'ratio':
                attributes_missing_dict[column] = missing_ratio
            if attributes_missing_dict[column] > 0:
                missing_categories_count += 1
        else:
            attributes_missing_dict.pop(column)
    if kwargs['print']:
        print("numbers of categories with missing values:", missing_categories_count)
        for e in attributes_missing_dict:
            if attributes_missing_dict[e] > 0:
                print(e + ' : ' + str(round(attributes_missing_dict[e], kwargs['round'])))
    if kwargs['type'] == 'abs' or kwargs['type'] == 'ratio':
        return attributes_missing_dict
    else:
        raise TypeError("invalid argument for type")


from sklearn.impute import SimpleImputer
def impute_missing_values_nominal(data):
    """
    Parameters
    ----------
    data: pandas.DataFrame,
    Input data for imputing missing values in nominal attributes

    Returns
    -------
    pandas.DataFrame,
    data that is in-place most frequent imputed, thus without missing values in nominal attributes
    """

    for col in data.select_dtypes(include=['object']).columns:
        imputer_freq = SimpleImputer(strategy='most_frequent')
        data[col] = imputer_freq.fit_transform(data[col].values.reshape(-1, 1))
    return data


def handle_missing_values(data, **kwargs):
    """
    Parameters
    ----------
    data: pandas.DataFrame,
    Input data for evaluating missing values in attributes

    kwargs:
    Explained in get_variables_missing_dict

    Returns
    -------
    pandas.DataFrame,
    data, that in-place drops all attributes with more than kwargs['min'] missing values
    and imputes the remaining nominal attributes with impute_missing_values_nominal
    """

    if 'type' not in kwargs:
        kwargs['type'] = 'ratio'
    if 'min' not in kwargs:
        kwargs['min'] = 0.0
    if 'max' not in kwargs:
        kwargs['max'] = 1.0
    if 'print' not in kwargs:
        kwargs['print'] = False
    if 'round' not in kwargs:
        kwargs['round'] = 3
    # print all absolute values and ratios of missing values:
    if kwargs['print']:
        get_variables_missing_dict(data, type=kwargs['type'], print=True)
    # find attributes with missing value ratios >= threshold and remove them
    missing_attributes_dict = get_variables_missing_dict(data, type=kwargs['type'], print=False, min=0.5)
    drop_list = []
    for missing_attribute in missing_attributes_dict.keys():
        drop_list.append(missing_attribute)
        data = data.drop(missing_attribute, 1)
    if kwargs['print']:
        print("Variables with missing val ratio >=", kwargs['min'], drop_list)
    # impute remaining nominal attributes
    data = impute_missing_values_nominal(data)
    return data


from sklearn.preprocessing import LabelEncoder
def encode_data_numerical(data):
    """
    Parameters
    ----------
    data: pandas.DataFrame,
    Input data for numerical encoding

    Returns
    -------
    pandas.DataFrame,
    new data with label encoded binary class and one-hot encoded nominal attributes
    """

    encoded_data = data.copy()
    le = LabelEncoder()
    encoded_data['class'] = le.fit_transform(data['class'])
    encoded_data = pd.get_dummies(encoded_data)
    return encoded_data


from sklearn.model_selection import train_test_split
def get_train_test(*datas, **kwargs):
    """
    Parameters
    ----------
    datas: list of pandas.DataFrame,
    len(datas) == 1: train-test-split on datas[0] is performed
    len(datas) == 2: train-test-split on both datas is performed, use training set from datas[0] and test set form datas[1]

    kwargs:
    test_size: float, default=0.2,
    test set size in percent (training set size = 1 - test set size)

    Returns
    -------
    var names=[X_train, X_test, y_train, y_test]: list of objects,
    X_train: pandas.DataFrame, attributes without class of the training set
    X_test: pandas.DataFrame, attributes without class of the test set
    y_train: numpy.ndarray, class of the training set
    y_test: numpy.ndarray, class of the test set
    """

    if 'test_size' not in kwargs:
        kwargs['test_size'] = 0.2
    # One dataset -> use sklearn.model_selection.train_test_split
    if len(datas) == 1:
        X = datas[0].drop(columns='class')
        y = datas[0]['class'].values.reshape(-1, 1)
        return train_test_split(X, y, test_size=kwargs['test_size'], random_state=1)
    # Two datasets -> use the first as training set, the second as test set
    elif len(datas) == 2:
        # assign datasets as train and test set and divide into X and y
        X1 = datas[0].drop(columns='class')
        y1 = datas[0]['class'].values.reshape(-1, 1)
        X_train, _, y_train, _ = train_test_split(X1, y1, test_size=kwargs['test_size'], random_state=1)
        X2 = datas[1].drop(columns='class')
        y2 = datas[1]['class'].values.reshape(-1, 1)
        _, X_test, _, y_test = train_test_split(X2, y2, test_size=kwargs['test_size'], random_state=1)
        return [X_train, X_test, y_train, y_test]
    else:
        raise TypeError("Invalid parameter for *datas")


# classifier with logistic regression, LDA and QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
def get_model(mode):
    """
    Parameters
    ----------
    mode: str,
    used classifier, 'nb': Naive Bayes, 'log_reg': Logistic regression, 'lda': Linear discriminant analysis,
    'qda':  Quadratic discriminant analysis

    Returns
    -------
    sklearn classification model depending on chosen mode,
    all classifiers share a fit(X, y) method to train on a training set
    """

    if mode == 'nb':
        model = GaussianNB()
    if mode == 'log_reg':
        model = LogisticRegression(max_iter=10000)
    if mode == 'lda':
        model = LinearDiscriminantAnalysis()
    if mode == 'qda':
        model = QuadraticDiscriminantAnalysis()
    return model


def train_model(X_train, y_train, mode):
    """
    Parameters
    ----------
    X_train: pandas.DataFrame,
    attributes excluding class of a training set
    y_train: numpy.ndarry,
    class of training set
    mode: str,
    used classifier, 'nb': Naive Bayes, 'log_reg': Logistic regression, 'lda': Linear discriminant analysis,
    'qda':  Quadratic discriminant analysis

    Returns
    -------
    sklearn classification model depending on chosen mode,
    all classifiers share a fit(X, y) method to train on a training set, the trained model is returned
    """

    model = get_model(mode)
    model.fit(X_train, y_train.ravel())
    return model


from sklearn.model_selection import cross_val_score
def cross_fold_validation(data, **kwargs):
    """
    Parameters
    ----------
    datas: list of pandas.DataFrame,
    data to perform k-fold cross-validation on

    kwargs:
    k: int, default=5,
    number of folds for the validation, corresponds to cv in sklearn.model_selection.cross_val_score
    scoring: str, default='accuracy',
    score method, corresponds to scoring in sklearn.model_selection.cross_val_score
    mode: str, default='log_reg',
    used classifier, 'nb': Naive Bayes, 'log_reg': Logistic regression, 'lda': Linear discriminant analysis,
    'qda':  Quadratic discriminant analysis

    Returns
    -------
    numpy.ndarray,
    array of kwargs[k] scores, each representing an iteration of cross-validation
    """

    if 'k' not in kwargs:
        kwargs['k'] = 5
    if 'scoring' not in kwargs:
        kwargs['scoring'] = 'accuracy'
    if 'mode' not in kwargs:
        kwargs['mode'] = 'log_reg'
    X = data.drop(columns=['class'])
    y = data['class'].values.reshape(-1, 1).ravel()
    model = get_model(kwargs['mode'])
    scores = cross_val_score(model, X, y, cv=kwargs['k'], scoring=kwargs['scoring'])
    return scores


# getting probability and confusion matrices
def get_y_prob_pred(X_test, model, **kwargs):
    """
    Parameters
    ----------
    X_test: pandas.DataFrame,
    attributes excluding class of a training set
    model: returned by get_model,
    sklearn classification model like GaussianNB

    kwargs:
    threshold: float, default=0.5,
    class dividing threshold, probabilities > kwargs['threshold'] are predicted as 1 others as 0

    Returns
    -------
    var names=[y_prob, y_pred], list of numpy.ndarray,
    y_prob, probability of belonging to class 1 for each test value,
    y_pred, prediction of 1 or 0 for each test value depending on kwargs['threshold']
    """

    if 'threshold' not in kwargs:
        kwargs['threshold'] = 0.5
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = np.where(y_prob > kwargs['threshold'], 1, 0)
    return [y_prob, y_pred]


from sklearn.metrics import confusion_matrix
def get_confusion_matrix(y_test, y_pred, **kwargs):
    """
    Parameters
    ----------
    y_test, pandas.Series: actual class values from the test set
    y_pred, pandas.Series: predicted class values

    kwargs:
    print, bool: prints return confusion matrix to console
    reformat, bool: changes the sklearn format for the confusion matrix to a common format:
     [[TN  FP]  ->  [[TP  FN]
      [FN  TP]]      [FP  TN]]


    Returns
    -------
    numpy.ndarray,
    confusion matrix, format depends on kwargs['reformat']
    """

    if 'print' not in kwargs:
        kwargs['print'] = True
    if 'reformat' not in kwargs:
        kwargs['reformat'] = True
    conf_mat = confusion_matrix(y_test, y_pred)
    if kwargs['reformat']:
        conf_mat_temp = np.zeros(shape=(2, 2))
        conf_mat_temp[0, 0] = conf_mat[1, 1]
        conf_mat_temp[0, 1] = conf_mat[1, 0]
        conf_mat_temp[1, 0] = conf_mat[0, 1]
        conf_mat_temp[1, 1] = conf_mat[0, 0]
        conf_mat = conf_mat_temp
    if kwargs['print']:
        print("Confusion Matrix:", conf_mat_temp, sep="\n")
    return conf_mat


from sklearn.metrics import roc_curve
def get_roc(y_test, y_prob):
    """
    Parameters
    ----------
    y_test, pandas.Series: actual class values from the test set
    y_pred, pandas.Series: predicted class values


    Returns
    -------
    var names=[false_positive_rate, true_positive_rate, thresholds]: list of pandas.Series,
    basically a pandas.DataFrame of three columns, including the FPR and TPR for each given threshold
    """

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    return [false_positive_rate, true_positive_rate, thresholds]


def classify_data(data, **kwargs):
    """
    Wrapper method for the entire classification process utilizing most functions of this module

    Parameters
    ----------
    data: pandas.DataFrame,
    input data for binary classification

    kwargs:
    mode: str, default='log_reg',
    used classifier, 'nb': Naive Bayes, 'log_reg': Logistic regression, 'lda': Linear discriminant analysis,
    'qda':  Quadratic discriminant analysis
    threshold: float, default=0.5,
    class dividing threshold, probabilities > kwargs['threshold'] are predicted as 1 others as 0
    encode: bool, default=True,
    if True numerically encodes the data, if False expects already encoded data
    impute: bool, default=True,
    if True imputes missing values of the data, if False expects data without missing values

    Returns
    -------
    var names=[X_train, X_test, y_train, y_test, model, y_prob, y_pred]: list of objects,
    X_train: pandas.DataFrame, attributes without class of the training set
    X_test: pandas.DataFrame, attributes without class of the test set
    y_train: numpy.ndarray, class of the training set
    y_test: numpy.ndarray, class of the test set
    model: returned by get_model, sklearn classifier like GaussianNB
    y_prob: numpy.ndarray, probability of belonging to class 1 for each test value
    y_pred: numpy.ndarray, prediction of 1 or 0 for each test value depending on kwargs['threshold']

    """

    if 'mode' not in kwargs:
        kwargs['mode'] = 'log_reg'
    if 'threshold' not in kwargs:
        kwargs['threshold'] = 0.5
    if 'encode' not in kwargs:
        kwargs['encode'] = True
    if 'impute' not in kwargs:
        kwargs['impute'] = True
    data_copy = data.copy()
    if kwargs['impute']:
        data_copy = impute_missing_values_nominal(data)
    if kwargs['encode']:
        data_copy = encode_data_numerical(data)
    X_train, X_test, y_train, y_test = get_train_test(data_copy)
    model = train_model(X_train, y_train, kwargs['mode'])
    y_prob, y_pred = get_y_prob_pred(X_test, model, threshold=kwargs['threshold'])
    return X_train, X_test, y_train, y_test, model, y_prob, y_pred


from sklearn import metrics
def get_evaluation_scores_dict(y_test, y_pred, **kwargs):
    """
    Parameters
    ----------
    y_test: numpy.ndarray,
    actual classes of the test set
    y_pred: numpy.ndarray,
    predicted classes for the test set

    kwargs:
    beta: int, default=2,
    corresponds to beta in sklearn.metrics.fbeta_score
    round: int, default=3,
    Decides the number of rounded decimal places for the return dict values
    print: bool, default=True,
    if True prints the formatted return dict or else, does nothing

    Returns
    -------
    var name=evaluation_scores_dict, dict {str: float},
    dict with score types as keys and the calculated score results as values
    """

    if 'beta' not in kwargs:
        kwargs['beta'] = 2
    if 'round' not in kwargs:
        kwargs['round'] = 3
    if 'print' not in kwargs:
        kwargs['print'] = True
    accuracy = round(metrics.accuracy_score(y_test, y_pred), kwargs['round'])
    precision = round(metrics.precision_score(y_test, y_pred), kwargs['round'])
    recall = round(metrics.recall_score(y_test, y_pred), kwargs['round'])
    f_beta = round(metrics.fbeta_score(y_test, y_pred, beta=kwargs['beta']), kwargs['round'])
    evaluation_scores_dict = {'Accuracy': accuracy, 'Precision': precision,
                              'Recall': recall, 'F' + str(kwargs['beta']): f_beta}
    if kwargs['print']:
        for score_key in evaluation_scores_dict:
            print(score_key + ": " + str(evaluation_scores_dict[score_key]))
    return evaluation_scores_dict


if __name__ == "__main__":
    """
    WARNING: 
    Running this module overwrites the following files in data:
        - secondary_data_no_miss.csv
        - 1987_data_no_miss.csv
    """

    # import datasets
    data_primary = pd.read_csv(data_cat.FILE_PATH_PRIMARY_EDITED, sep=';', header=0)
    data_secondary = pd.read_csv(data_cat.FILE_PATH_SECONDARY_NO_MISS, sep=';', header=0, low_memory=False)
    data_original = pd.read_csv(data_cat.FILE_PATH_1987, sep=',', header=0, dtype=object, na_values='?')
    data_dict = {'Secondary dataset': data_secondary, 'Original dataset': data_original}

    ## exploratory data analysis ##
    # missing values #
    # print absolute values and ratios of missing values:
    data_secondary = handle_missing_values(data_secondary, min=0.5, print=True)
    data_original = handle_missing_values(data_original, min=0.5, print=True)
    # write missing imputed versions of the data into csvs
    data_secondary.to_csv(data_cat.FILE_PATH_SECONDARY_NO_MISS, sep=';', index=False)
    data_original.to_csv(data_cat.FILE_PATH_1987_NO_MISS, sep=';', index=False)


    ## cross validation ##
    """accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)
    f2_scorer = metrics.make_scorer(metrics.fbeta_score, beta=2)
    scorers = [accuracy_scorer, f2_scorer]
    for data_key in data_dict:
        for score in scorers:
            print(score)
            for mode_key in mode_dict:
                data = data_dict[data_key].copy()
                data_encoded = encode_data_numerical(data)
                cross_val_scores = cross_fold_validation(data_encoded, k=5, scoring=score, mode=mode_key)
                print(data_key, mode_key, [round(s, 2) for s in cross_val_scores])
                print('mean:', round(np.mean(cross_val_scores), 2), 'var:', round(np.var(cross_val_scores), 4) * 100)
                print()"""


    ## classification task ##
    for data_key in data_dict:
        print("\n***" + data_key + "***")
        for mode_key in mode_dict:
            print("\n" + mode_dict[mode_key] + ":")
            X_train, X_test, y_train, y_test, model, y_prob, y_pred = \
                classify_data(data_dict[data_key], mode=mode_key, threshold=0.5)
            get_confusion_matrix(y_test, y_pred)
            scores_dict = get_evaluation_scores_dict(y_test, y_pred)


    ## direct test between datasets ##
    print("\n*** direct tests between datasets ***\n")
    # get datasets with encoded and matched columns created by data_col_match.py
    data_new_matched = pd.read_csv(data_cat.FILE_PATH_SECONDARY_MATCHED, header=0, sep=';')
    data_1987_matched = pd.read_csv(data_cat.FILE_PATH_1987_MATCHED, header=0, sep=';')

    # test reduced instances
    print('\n* Test reduced datasets on themselves *')
    print('\n Secondary dataset')
    X_train, X_test, y_train, y_test, model, y_prob, y_pred = \
        classify_data(data_new_matched, mode='lda', encode=False)
    get_confusion_matrix(y_test, y_pred)
    get_evaluation_scores_dict(y_test, y_pred)
    print('\n 1987 dataset')
    X_train, X_test, y_train, y_test, model, y_prob, y_pred = \
        classify_data(data_1987_matched, mode='lda', encode=False)
    get_confusion_matrix(y_test, y_pred)
    get_evaluation_scores_dict(y_test, y_pred)

    # use one dataset as the training set and the other dataset as the test set
    for mode_key in mode_dict:
        print("\n* training set = secondary -> test set = original", "model: ", mode_key + " *")
        X_train, X_test, y_train, y_test = get_train_test(data_new_matched, data_1987_matched)
        model = train_model(X_train, y_train, mode_key)
        y_prob, y_pred = get_y_prob_pred(X_test, model)
        print("Conf.-Mat.: " + str(get_confusion_matrix(y_test, y_pred)))
        get_evaluation_scores_dict(y_test, y_pred)

        print("\ntraining set = original -> test set = secondary", "model:", mode_key)
        X_train, X_test, y_train, y_test = get_train_test(data_1987_matched, data_new_matched)
        model = train_model(X_train, y_train, mode_key)
        y_prob, y_pred = get_y_prob_pred(X_test, model)
        print("Conf.-Mat.: " + str(get_confusion_matrix(y_test, y_pred)))
        get_evaluation_scores_dict(y_test, y_pred)
