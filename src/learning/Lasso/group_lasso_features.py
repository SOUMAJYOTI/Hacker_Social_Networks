# Sparse group lasso for inducing sparsity within the groups
# The idea is to induce sparsity for entire group of features for all time points
# The second is to induce regularization like ridge within the selected groups
# It can be either jointly modeled or modeled in a 2 stage process***.

# This is to only test all the features combined ----> this would result in t * |f| features (time X # fearures)
# So sparisty is necessary for training with small datasets

import numpy as np
import subgradients, subgradients_semisparse, blockwise_descent, blockwise_descent_semisparse
import pickle
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import sklearn.metrics
from sklearn import linear_model, ensemble
from blockwise_descent_semisparse import *


def prepareOutput(eventsDf, start_date, end_date):
    eventsDf['date'] = pd.to_datetime(eventsDf['date'])

    # For now, just consider the boolean case of attacks, later can extend to count
    currDay = start_date
    datesList = []
    attackFlag = []
    while(currDay < end_date):
        try:
            events = eventsDf[eventsDf['date'] == currDay]
            total_count = pd.DataFrame(events.groupby(['date']).sum())
            count_attacks = (total_count['count'].values)[0]

            if count_attacks == 0:
                attackFlag.append(0)
            else:
                attackFlag.append(1)
        except:
            attackFlag.append(0)

        datesList.append(currDay)
        currDay = currDay + datetime.timedelta(days=1)

    outputDf = pd.DataFrame()
    outputDf['date'] = datesList
    outputDf['attackFlag'] = attackFlag

    return outputDf


def prepareData(inputDf, outputDf, delta_gap_time, delta_prev_time_start):
    train_start_date = pd.to_datetime(outputDf.iloc[0, 0]) # first row
    train_end_date = pd.to_datetime(outputDf.iloc[-1, 0]) # last row

    inputDf['date'] = pd.to_datetime(inputDf['date'])
    inputDf['date'] = inputDf['date'].dt.date

    outputDf['date'] = pd.to_datetime(outputDf['date'])
    outputDf['date'] = outputDf['date'].dt.date

    y_true = outputDf['attackFlag']

    currDate = train_start_date
    countDayIndx = 0

    num_features = len(inputDf.columns) - 1
    X = np.zeros((y_true.shape[0], delta_gap_time*num_features)) #input
    Y = -np.ones((y_true.shape[0], )) #output

    while (currDate <= train_end_date):
        ''' This loop checks values on either of delta days prior '''
        for idx in range(delta_gap_time):
            historical_day = pd.to_datetime(currDate - datetime.timedelta(days=delta_prev_time_start - idx)) # one week before
            try:
                feat_count = 0
                for feat in inputDf.columns.values:
                    if feat == 'date':
                        continue
                    # Arrange the column values grouped in time intervals of individual features
                    X[countDayIndx, (feat_count*delta_gap_time) + idx] = (inputDf[inputDf['date'] == historical_day.date()][feat]).values[0]
                    feat_count += 1
            except:
                continue

        Y[countDayIndx] = (outputDf[outputDf['date'] == currDate.date()].ix[:, 1])

        countDayIndx += 1
        currDate += datetime.timedelta(days=1)

    # Fill the missing cells with median of column
    # for col in range(X.shape[1]):
    #     X[np.where(X[:, col] == 0)[0], col] = np.median(X[~np.where(X[:, col] == -1)[0], col])

    return X, Y


def assign_group_ids(inputDf, time_gap):
    '''
    The task of this function is to assign each feature with temporal points a separate group
    :param inputDf:
    :return:
    '''

    num_features = len(inputDf.columns) - 1 # exclude date
    groups = []

    for idx in range(num_features):
        groups += [idx] * time_gap

    return np.array(groups)


def feature_scaling(inputData):
    for idx_col in range(inputData.shape[1]):
        inputData[:, idx_col] = (inputData[:, idx_col] - min(inputData[:, idx_col])) / (max(inputData[:, idx_col]) - min(inputData[:, idx_col]))

    return inputData

def main():
    # Just setting the features list among all the groups
    featList = ['date', 'numUsers', 'numVulnerabilities', 'numThreads', 'expert_NonInteractions',
                'communityCount', 'CondExperts', 'expertsThreads']


    # model.fit(X, y)
    # beta_hat = model.coef_
    #
    # exit()
    #
    # print(np.linalg.norm(secret_beta - beta_hat))
    # for i, (betai_hat, betai) in enumerate(zip(beta_hat, secret_beta)):
    #     print("Component %02d: %.4f | %.4f" % (i, betai_hat, betai))
    # print("Correct classification rate: %.3f" % (np.sum(model.predict(X) == y) / n))


    ''' SET THE TRAINING AND TEST TIME PERIODS - THIS IS MANUAL '''
    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')

    testStart_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    amEvents = pd.read_csv('../../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

    feat_df = pickle.load(
        open('../../../data/DW_data/features/feat_combine/features_Delta_T0_Mar16-Aug17.pickle', 'rb'))

    feat_df = feat_df[featList]

    # the previous month is needed for features - since we measure lag correlation for the prediction
    instance_TrainStartDate = trainStart_date - relativedelta(months=1)  # the previous month is needed for features

    # Make the date common for all
    feat_df['date'] = pd.to_datetime(feat_df['date'])
    trainDf = feat_df[feat_df['date'] >= instance_TrainStartDate]
    trainDf = trainDf[trainDf['date'] < trainEnd_date]
    trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)

    instance_TestStartDate = testStart_date - relativedelta(months=1)
    testDf = feat_df[feat_df['date'] >= instance_TestStartDate]
    testDf = testDf[testDf['date'] < testEnd_date]
    testOutput = prepareOutput(amEvents_malware, testStart_date, testEnd_date)


    ''' PARAMETER INITIALIZATION '''
    alpha = .5
    epsilon = .001
    lbda = 0.01

    delta_gap_time = [7, ]
    delta_prev_time_start = [8, 14, 21, 28, 35]

    for dgt in delta_gap_time:
        outputDf = pd.DataFrame()
        scoreDict = {}

        # Assign the group IDs in order of the features
        group_ids = assign_group_ids(trainDf, dgt)

        d = group_ids.shape[0]  # number of dimensions
        # Other assumption: ind_sparse is of dimension X.shape[1] and has 0 if the dimension should not be pushed
        # towards sparsity and 1 otherwise
        ind_sparse = np.zeros((d,))

        secret_beta = np.random.randn(d)

        precList = []
        recList = []
        f1List = []
        for dprev in delta_prev_time_start:
            ''

            X_train, y_train = prepareData(trainDf, trainOutput, dgt, dprev)
            X_test, y_test = prepareData(testDf, testOutput, dgt, dprev)

            X_train = feature_scaling(X_train)
            X_test = feature_scaling(X_test)

            lambda_max = blockwise_descent_semisparse.SGL.lambda_max(X_train, y_train, groups=group_ids, alpha=alpha, ind_sparse=ind_sparse)
            # for l in [lambda_max - epsilon, lambda_max + epsilon]:
            # print('Starting....')
            #     model = blockwise_descent_semisparse.SGL(groups=group_ids, alpha=alpha, lbda=l, ind_sparse=ind_sparse)
            model = blockwise_descent_semisparse.SGL_LogisticRegression(groups=group_ids, alpha=alpha, lbda=lbda,
                                                                ind_sparse=ind_sparse, max_iter_outer=500)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            prec, rec, f1_score = sklearn.metrics.precision_score(y_test, y_pred), \
                                  sklearn.metrics.recall_score(y_test, y_pred), \
                                  sklearn.metrics.f1_score(y_test, y_pred),

            print(prec, rec, f1_score)

            exit()

if __name__ == "__main__":
    main()