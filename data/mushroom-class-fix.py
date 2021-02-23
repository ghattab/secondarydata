import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics
import pandas as pd
import numpy as np
from "data-cat" import *
from "mushroom-class" import *

"""
This class is used for reliable classification results, running the main method or
using the high level functions of mushroom_classifier.py for classification produces
flawed results. The bugs will be fixed in a later version.
"""

if __name__ == "__main__":
    # import and data and asign  to variables
    data_primary = pd.read_csv(dataset_categories.FILE_PATH_PRIMARY_EDITED, header=0, sep=';')
    data_secondary = pd.read_csv(dataset_categories.FILE_PATH_SECONDARY_NO_MISS, header=0, sep=';')
    data_1987 = pd.read_csv(dataset_categories.FILE_PATH_1987_NO_MISS, header=0, sep=';')
    data_dict = {'Secondary data': data_secondary, '1987 data': data_1987}
    data1 = pd.read_csv(dataset_categories.FILE_PATH_SECONDARY_MATCHED, header=0, sep=';')
    data2 = pd.read_csv(dataset_categories.FILE_PATH_1987_MATCHED, header=0, sep=';')

    print('Different families in primary data:')
    print(data_primary.family.unique())
    print('Number of families:', data_primary.family.nunique())


    # cross validation
    print('\n*** Cross validation ***\n')
    accuracy_scorer = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)
    f2_scorer = sklearn.metrics.make_scorer(sklearn.metrics.fbeta_score, beta=2)
    scorers = [accuracy_scorer, f2_scorer]
    for data_key in data_dict:
        for score in scorers:
            print(score)
            for mode_key in mushroom_classifier.mode_dict:
                data = data_dict[data_key].copy()
                # assign to variables
                X = data.drop(columns=['class'])
                y = data['class']
                # encoding: Label encoding for binary class, one-hot encoding for the nominal variables
                y = sklearn.preprocessing.LabelEncoder().fit_transform(y)
                X = pd.get_dummies(X)
                cross_val_scores = sklearn.model_selection.cross_val_score(mushroom_classifier.get_model(mode_key), X, y, cv=5, scoring=score)
                print(data_key, mode_key, [round(s, 2) for s in cross_val_scores])
                print('mean:', round(np.mean(cross_val_scores), 2), 'var:', round(np.var(cross_val_scores), 4) * 100)
                print()

    # classify the secondary data and the 1987 data separately with Naive Bayes, logistic regression and LDA
    print('\n*** Separate classification ***\n')
    for data_key in data_dict:
        print('\n**', data_key, '**\n')
        data = data_dict[data_key].copy()
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

        for model in models:
            print(model)
            model.fit(X_train, y_train.ravel())
            y_pred = model.predict(X_test)
            mushroom_classifier.get_confusion_matrix(y_test, y_pred)
            mushroom_classifier.get_evaluation_scores_dict(y_test, y_pred)



    # classify variable matched versions of the secondary data and the 1987 data
    # classify each of them once on their own test set and once the other test set
    print('\n*** Matched classification ***\n')

    # assign to variables
    X1 = data1.drop(columns=['class'])
    X2 = data2.drop(columns=['class'])
    y1 = data1['class']
    y2 = data2['class']


    # get training and test set
    X_train1, X_test1, y_train1, y_test1 = sklearn.model_selection.train_test_split(X1, y1, test_size=0.2, random_state=1)
    X_train2, X_test2, y_train2, y_test2 = sklearn.model_selection.train_test_split(X2, y2, test_size=0.2, random_state=1)
    X1_train, X2_test, y1_train, y2_test = X1.copy(), X2.copy(), y1.copy(), y2.copy()
    log_reg1 = sklearn.linear_model.LogisticRegression(max_iter=10000)
    lda1 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    gnb1 = sklearn.naive_bayes.GaussianNB()
    log_reg2 = sklearn.linear_model.LogisticRegression(max_iter=10000)
    lda2 = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    gnb2 = sklearn.naive_bayes.GaussianNB()

    models1 = [gnb1, log_reg1, lda1]
    models2 = [gnb2, log_reg2, lda2]

    print('\nTraining on secondary')
    for model in models1:
        model.fit(X_train1, y_train1.ravel())
        y_pred_self1 = model.predict(X_test1)
        y_pred_2_1 = model.predict(X_test2)
        print('\n', model)
        print('test set=self', 'conf-mat:\n', mushroom_classifier.get_confusion_matrix(y_test1, y_pred_self1, print=False))
        print('test set=other', 'conf-mat:\n', mushroom_classifier.get_confusion_matrix(y_test2, y_pred_2_1, print=False))
        print('test set=self', 'accuracy:', round(sklearn.metrics.accuracy_score(y_test1, y_pred_self1), 2))
        print('test set=other', 'accuracy:', round(sklearn.metrics.accuracy_score(y_test2, y_pred_2_1), 2))
        print('test set=self', 'F2',  round(sklearn.metrics.fbeta_score(y_test1, y_pred_self1, beta=2), 2))
        print('test set=other', 'F2', round(sklearn.metrics.fbeta_score(y_test2, y_pred_2_1, beta=2), 2))

    print('\nTraining on 1987 data')
    for model in models2:
        model.fit(X_train2, y_train2.ravel())
        y_pred_self2 = model.predict(X_test2)
        y_pred_1_2 = model.predict(X_test1)
        print('\n', model)
        print('test set=self', 'conf-mat:\n', mushroom_classifier.get_confusion_matrix(y_test2, y_pred_self2, print=False))
        print('test set=other', 'conf-mat:\n', mushroom_classifier.get_confusion_matrix(y_test1, y_pred_1_2, print=False))
        print('test set=self', 'accuracy:', round(sklearn.metrics.accuracy_score(y_test2, y_pred_self2), 2))
        print('test set=other', 'accuracy:', round(sklearn.metrics.accuracy_score(y_test1, y_pred_1_2), 2))
        print('test set=self', 'F2',  round(sklearn.metrics.fbeta_score(y_test2, y_pred_self2, beta=2), 2))
        print('test set=other', 'F2', round(sklearn.metrics.fbeta_score(y_test1, y_pred_1_2, beta=2), 2))
