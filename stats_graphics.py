import altair as alt
import altair_viewer as view
import pandas as pd
import numpy as np
import sklearn
import scipy.stats.stats

import data_cat
import mushroom_class_fix
import util_func

from altair import pipe, limit_rows, to_values
t = lambda data: pipe(data, limit_rows(max_rows=100000), to_values)
alt.data_transformers.register('custom', t)
alt.data_transformers.enable('custom')

def get_balance_chart(data, **kwargs):
    """
    Parameters
    ----------
    data: pandas.DataFrame
    DataFrame with nominal or metrical columns

    kwargs:
    title: str, default="Balance plot",
    altair.Chart title
    count: bool, default=True,
    adds the percentage values of the class values to the title
    reindex: list of strs or False, default=False,
    nothing if False, else reindexes the class values according to the given list

    Returns
    -------
    var name=chart: altair.Chart,
    bar plot of the class value occurences
    """

    if 'title' not in kwargs:
        kwargs['title'] = "Balance plot"
    if 'count' not in kwargs:
        kwargs['count'] = True
    if 'reindex' not in kwargs:
        kwargs['reindex'] = False
    if kwargs['count']:
        size = len(data)
        val_counts = data['class'].value_counts()
        if kwargs['reindex']:
            val_counts = val_counts.reindex(kwargs['reindex'])
        kwargs['title'] += " ("
        for val in val_counts.index:
            ratio = val_counts[val] / size
            kwargs['title'] = "".join([kwargs['title'], val, ": %0.2f" % ratio, ", "])
        kwargs['title'] = "".join([kwargs['title'][:-2], ")"])
    chart = alt.Chart(data, title=kwargs['title']).mark_bar(size=150).encode(
        alt.X('class:N', sort='descending'),
        alt.Y('count():Q'),
        color=alt.value('grey')
    ).properties(width=400)
    return chart


from dython import nominal
def get_correlation_dataframe(data, **kwargs):
    """
    Parameters
    ----------
    data: pandas.DataFrame
    DataFrame with nominal or metrical columns

    kwargs:
    show_progress: bool, default=False
    Prints each row if True

    Returns
    -------
    var name=data_corr: pandas.DataFrame,
    with two column names and their correlation
    """

    if 'show_progress' not in kwargs:
        kwargs['show_progress'] = False
    data_corr = pd.DataFrame(columns=['variable1', 'variable2', 'correlation', 'correlation_rounded'])
    for variable1 in data:
        for variable2 in data:
            # nominal-nominal -> Theils U
            if type(data[variable1][0]) == str and type(data[variable2][0]) == str:
                corr = nominal.theils_u(data[variable1], data[variable2], nan_replace_value='f')
            # metircal-metrical -> Pearsons R
            elif util_func.is_number(data[variable1][0]) and util_func.is_number(data[variable2][0]):
                corr = scipy.stats.stats.pearsonr(data[variable1], data[variable2])[0]
                # change range from [-1, 1] to [0, 1] as the other metrics
                corr = (corr + 1) / 2
            # metrical-nominal -> correlation ratio
            elif type(data[variable1][0]) == str and util_func.is_number(data[variable2][0]):
                corr = nominal.correlation_ratio(data[variable1], data[variable2], nan_replace_value='f')
            elif type(data[variable2][0]) == str and util_func.is_number(data[variable1][0]):
                corr = nominal.correlation_ratio(data[variable2], data[variable1], nan_replace_value='f')
            else:
                print('var1-type: ' + str(type(data[variable1][0])) + ', var2-type: ' + str(type(data[variable2][0])))
                print('var1: ' + str(data[variable1][0]) + ', var2: ' + str(data[variable2][0]))
            new_row = {'variable1': variable1, 'variable2': variable2,
                'correlation': corr, 'correlation_rounded': round(corr, 2)}
            data_corr = data_corr.append(new_row, ignore_index=True)
            if kwargs['show_progress']:
                print(new_row)
    return data_corr


def get_correlation_chart(data, **kwargs):
    """
    Parameters
    ----------
    data: pandas.DataFrame
    data with nominal or metrical columns

    kwargs:
    show_progress: bool, default=False,
    prints each row if True

    Returns
    -------
    altair.Chart,
    correlation heatmap of the data columns based on get_correlation_dataframe
    """

    if 'show_progress' not in kwargs:
        kwargs['show_progress'] = False

    data_corr = get_correlation_dataframe(data, show_progress=kwargs['show_progress'])

    base_chart = alt.Chart(data_corr).encode(
        alt.X('variable1:N', sort=data.columns.values),
        alt.Y('variable2:N', sort=data.columns.values)
    )

    corr_chart = base_chart.mark_rect().encode(
        alt.Color('correlation:Q', scale=alt.Scale(scheme='greys')),
    )

    text_chart = base_chart.mark_text().encode(
        alt.Text('correlation_rounded:Q'),
        color = (alt.condition(
            alt.datum.correlation > 0.5,
            alt.value('white'),
            alt.value('black')
        ))
    )

    return corr_chart + text_chart


def get_score_threshold_dataframe(X_train, X_test, y_train, y_test, mode, score):
    """
    Parameters
    ----------
    X_train: pandas.DataFrame, attributes without class of the training set
    X_test: pandas.DataFrame, attributes without class of the test set
    y_train: numpy.ndarray, class of the training set
    y_test: numpy.ndarray, class of the test set
    mode: str, used classifier, look at mushroom_class_fix.train_model for details
    score: str, used scoring method, look at mushroom_class_fix.get_evaluation_scores_dict for details


    Returns
    -------
    var name=data: pandas.DataFrame,
    with a threshold column from [0; 1] in 0.1 steps
    and a score column with the calculated score for each threshold using mushroom_class_fix.get_y_prob_pred
    """

    data = pd.DataFrame(columns=['scores', 'thresholds'], dtype=np.float64)
    data.thresholds = [t / 1000 for t in range(0, 1001, 10)]
    model = mushroom_class_fix.train_model(X_train, y_train, mode)
    scores = []
    for threshold in data.thresholds:
        y_prob, y_pred = mushroom_class_fix.get_y_prob_pred(X_test, model, threshold=threshold)
        scores.append(mushroom_class_fix.get_evaluation_scores_dict(y_test, y_pred, print=False)[score])
    data.scores = scores
    return data



def get_score_threshold_chart(X_train, X_test, y_train, y_test, mode, score):
    """
    Parameters
    ----------
    explained in get_score_threshold_dataframe

    Returns
    -------
    altair.Chart,
    threshold scoring plot (to choose the threshold for the best scoring) using get_score_threshold_dataframe
    """

    data = get_score_threshold_dataframe(X_train, X_test, y_train, y_test, mode, score)
    title = ''.join(['Score-threshold-plot ', mode, ' ', score])
    chart = alt.Chart(data, title=title).mark_line().encode(
        alt.X('thresholds:Q'),
        alt.Y('scores:Q'),
        color=alt.value('black')
    )
    return chart


def get_roc_dataframe(X_train, X_test, y_train, y_test, mode):
    """
    Parameters
    ----------
    X_train: pandas.DataFrame, attributes without class of the training set
    X_test: pandas.DataFrame, attributes without class of the test set
    y_train: numpy.ndarray, class of the training set
    y_test: numpy.ndarray, class of the test set
    mode: str, used classifier, look at mushroom_class_fix.train_model for details

    Returns
    -------
    var name=data_roc: pandas.DataFrame,
    contains the necessary columns for a ROC plot TPR, FPR and threshold
    """

    data_roc = pd.DataFrame(columns=['tpr', 'fpr', 'threshold'], dtype=np.float64)
    model = mushroom_class_fix.train_model(X_train, y_train, mode)
    y_prob, y_pred = mushroom_class_fix.get_y_prob_pred(X_test, model)
    false_positive_rate, true_positive_rate, thresholds = mushroom_class_fix.get_roc(y_test, y_prob)
    for i in range(0, len(false_positive_rate)):
        new_row = {'true positive rate': true_positive_rate[i],
               'false positive rate': false_positive_rate[i],
               'threshold': thresholds[i]}
        data_roc = data_roc.append(new_row, ignore_index=True)
    return data_roc


from sklearn.metrics import auc
def get_roc_chart(X_train, X_test, y_train, y_test, mode, **kwargs):
    """
    Parameters
    ----------
    explained in get_roc_dataframe

    kwargs:
    title: str, default is constructed containing 'ROC curve', the mode and the AUC,
    title of the return altair.Chart

    Returns
    -------
    altair.Chart,
    ROC curve plot with colored AUC
    """

    data_roc = get_roc_dataframe(X_train, X_test, y_train, y_test, mode)
    if 'title' not in kwargs:
        kwargs['title'] = 'ROC curve for ' + mushroom_class_fix.mode_dict[mode]\
            + ', AUC = %0.2f' % auc(data_roc['false positive rate'], data_roc['true positive rate'])
    line_chart = alt.Chart(data_roc, title=kwargs['title']).mark_line().encode(
        alt.X('false positive rate:Q'),
        alt.Y('true positive rate:Q'),
        color=alt.value('black')
    )
    area_chart = alt.Chart(data_roc).mark_area().encode(
        alt.X('false positive rate:Q'),
        alt.Y('true positive rate:Q'),
        color=alt.value('grey')
    )
    return area_chart + line_chart


if __name__ == "__main__":
    data_new = pd.read_csv(data_cat.FILE_PATH_SECONDARY_NO_MISS,
        sep=';', header=0, low_memory=False)
    data_original = pd.read_csv(data_cat.FILE_PATH_1987_NO_MISS,
        sep=';', header=0, dtype=object)
    # matched sets of data
    #data_new = pd.read_csv(data_cat.FILE_PATH_SECONDARY_MATCHED, sep=',')
    #data_original = pd.read_csv(data_cat.FILE_PATH_ORIGINAL_MATCHED, sep=',')

    categories_secondary_list = data_cat.categories_secondary_list
    categories_original_list = data_cat.categories_original_list

    # encode data
    data_new_encoded = mushroom_class_fix.encode_data_numerical(data_new)
    data_original_encoded = mushroom_class_fix.encode_data_numerical(data_original)

    # classification model 0: nb, 1: log_reg, 2: lda, 3: qda
    # mode = list(mushroom_class_fix.mode_dict.keys())[0]
    # training and test set based on one set of data
    # X_train, X_test, y_train, y_test = mushroom_class_fix.get_train_test(data_new_encoded)
    # training and test set based on two sets of data
    # X_train, X_test, y_train, y_test = mushroom_class_fix.get_train_test(data_new_encoded, data_original_encoded)


    """classification for ROC curve"""
    data = data_new.copy()
    mode = 'rf'
    # assign to variables
    X = data.drop(columns=['class'])
    y = data['class']
    # encoding: Label encoding for binary class, one-hot encoding for the nominal variables
    y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
    X = pd.get_dummies(X)
    log_reg = sklearn.linear_model.LogisticRegression(max_iter=10000)
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    gnb = sklearn.naive_bayes.GaussianNB()
    models = [log_reg, lda, gnb]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1)


    # threshold charts
    threshold_chart = get_score_threshold_chart(X_train, X_test, y_train, y_test, 'nb', 'F2')

    # roc charts
    # chart_secondary = get_roc_chart(X_train, X_test, y_train, y_test, mode)
    # chart_original = get_roc_chart(X_train, X_test, y_train, y_test, mode)

    """ comment/uncomment depending on the chart that shall be displayed"""
    chart = get_roc_chart(X_train, X_test, y_train, y_test, mode)
    chart.save('roc.html')
    view.display(chart)
    #view.display(get_balance_chart(data_new, title='Balance plot for secondary data', reindex=['p', 'e']))
    #view.display(get_balance_chart(data_original, title='Balance plot for 1987 data'))#, count=False, reindex=['p', 'e']))
    #view.display(get_correlation_chart(data_new, show_progress=True))
    #view.display(get_correlation_chart(data_original, show_progress=True))
    #view.display(threshold_chart)