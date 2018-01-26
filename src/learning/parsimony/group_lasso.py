import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.functions.nesterov.tv as tv
import numpy as np
import pickle
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import sklearn.metrics
from sklearn import linear_model, ensemble
import parsimony.functions.nesterov.gl as gl


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
    Y = -np.ones((y_true.shape[0], 1)) #output

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
    group_list = []

    for idx in range((num_features+1) * time_gap):
        if idx % time_gap == 0 and idx != 0:
            group_list.append(groups)
            groups = []
        groups.append(idx)

    return group_list


def feature_scaling(inputData):
    for idx_col in range(inputData.shape[1]):
        inputData[:, idx_col] = (inputData[:, idx_col] - min(inputData[:, idx_col])) / (max(inputData[:, idx_col]) - min(inputData[:, idx_col]))

    return inputData

def main():
    # Just setting the features list among all the groups
    featList = ['date', 'numUsers', 'numVulnerabilities', 'numThreads', 'expert_NonInteractions', 'shortestPaths',
                'communityCount', 'CondExperts', 'expertsThreads']


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
    k = 0.0  # l2 ridge regression coefficient
    l = 0.3  # l1 lasso coefficient
    g = 0.5  # group lasso coefficient

    delta_gap_time = [7, 14]
    delta_prev_time_start = [8, 15, 21, 28, 35]

    for dgt in delta_gap_time:
        print(dgt)
        scoreDict = {}
        # Assign the group IDs in order of the features
        group_ids = list(assign_group_ids(trainDf, dgt))

        d = len(list(np.array(group_ids).flatten()) ) # number of dimensions

        # print(d, group_ids)
        A = gl.linear_operator_from_groups(d, group_ids)

        precList = []
        recList = []
        f1List = []
        for dprev in delta_prev_time_start:
            ''

            if dgt >= dprev:
                continue
            X_train, y_train = prepareData(trainDf, trainOutput, dgt, dprev)
            X_test, y_test = prepareData(testDf, testOutput, dgt, dprev)

            X_train = feature_scaling(X_train)
            X_test = feature_scaling(X_test)

            # print(X_train.shape, y_train.shape)
            estimator = estimators.LogisticRegressionL1L2GL(
                                      k, l, g, A=A,
                                      algorithm=algorithms.proximal.FISTA(),
                                      algorithm_params=dict(max_iter=1000))
            model = estimator.fit(X_train, y_train)

            print(model.parameters())

            y_pred = model.predict(X_test)

            prec, rec, f1_score = sklearn.metrics.precision_score(y_test, y_pred), \
                                  sklearn.metrics.recall_score(y_test, y_pred), \
                                  sklearn.metrics.f1_score(y_test, y_pred),

            # print(y_pred)

            precList.append(prec)
            recList.append(rec)
            f1List.append(f1_score)

            print(f1List)

        scoreDict['precision'] = precList
        scoreDict['recall'] = recList
        scoreDict['f1'] = f1List

        pickle.dump(scoreDict,
                    open('../../../data/results/01_09/regular/group/' + 'tgap_' + str(dgt) + '_gparam_' + str(g) + '.pickle', 'wb'))


if __name__ == "__main__":
    main()