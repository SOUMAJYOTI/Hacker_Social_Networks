import sys
sys.path.append('../test')
sys.path.append('../lib')
import glmnet_python
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
from glmnet import glmnet
# from glmnet import LogitNet
# from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict
import pandas as pd
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

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


def prepareData(inputDf, outputDf):
    train_start_date = outputDf.iloc[0, 0] # first row
    train_end_date = outputDf.iloc[-1, 0] # last row

    inputDf['date'] = pd.to_datetime(inputDf['date'])
    inputDf['date'] = inputDf['date'].dt.date

    outputDf['date'] = pd.to_datetime(outputDf['date'])
    outputDf['date'] = outputDf['date'].dt.date

    y_true = outputDf['attackFlag']
    delta_prev_time = 10  # no of days to check before the week of current day

    currDate = train_start_date
    countDayIndx = 0

    num_features = len(inputDf.columns) - 1
    X = np.zeros((y_true.shape[0], delta_prev_time*num_features)) #input
    Y = -np.ones((y_true.shape[0], 1)) #output

    while (currDate <= train_end_date):
        ''' This loop checks values on either of delta days prior '''
        for idx in range(delta_prev_time):
            historical_day = pd.to_datetime(currDate - datetime.timedelta(days=(14 - idx + 1))) # one week before
            try:
                X[countDayIndx, idx*num_features:(idx+1)*num_features] = (inputDf[inputDf['date'] == historical_day.date()].ix[:,1:]).values[0] # exclude date
            except:
                continue

        # print(outputDf[outputDf['date'] == currDate.date()])
        Y[countDayIndx] = [outputDf[outputDf['date'] == currDate.date()].ix[:, 1]]

        countDayIndx += 1
        currDate += datetime.timedelta(days=1)

    # Fill the missing cells with median of column
    # for col in range(X.shape[1]):
    #     X[np.where(X[:, col] == 0)[0], col] = np.median(X[~np.where(X[:, col] == -1)[0], col])

    return X, Y



def main():
    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-03-01', '%Y-%m-%d')

    feat_df = pickle.load(open('../../data/DW_data/features/feat_combine/user_graph_Delta_T0_Sept16-Apr17.pickle', 'rb'))
    # feat_df = pickle.load(open('../../data/DW_data/features/subspace__v12_22.pickle', 'rb'))

    instance_TrainStartDate = trainStart_date - relativedelta(months=1) # the previous month is needed for features
    trainDf = feat_df[feat_df['date'] >= instance_TrainStartDate]
    trainDf = trainDf[trainDf['date'] < trainEnd_date]

    # plot_feat_stats(trainDf)

    # print(trainDf.shape)
    trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)
    # y_actual_train = trainOutput['attackFlag']

    testStart_date = datetime.datetime.strptime('2017-03-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-05-01', '%Y-%m-%d')

    instance_TestStartDate = testStart_date - relativedelta(months=1)
    testDf =feat_df[feat_df['date'] >= instance_TestStartDate]
    testDf = testDf[testDf['date'] < testEnd_date]

    testOutput = prepareOutput(amEvents_malware, testStart_date, testEnd_date)

    y_actual_test = list(testOutput['attackFlag'])

    # y_random = np.random.randint(2, size=len(y_actual_test))
    # random_prec = sklearn.metrics.precision_score(y_actual_test, y_random)
    # random_recall = sklearn.metrics.recall_score(y_actual_test, y_random)
    # random_f1 = sklearn.metrics.f1_score(y_actual_test, y_random)
    # print('Random: ', random_prec, random_recall, random_f1)

    X_train, y_train = prepareData(trainDf, trainOutput)
    X_test, y_test = prepareData(testDf, testOutput)

    fit = glmnet(x=X_train.copy(), y=y_train.copy(), family='binomial')
    # X_train = preprocessing.normalize(X_train, norm='l2')
    # X_test = preprocessing.normalize(X_test, norm='l2')

    # glmnetPlot(fit, xvar='dev', label=True)
    # plt.show()

    y_pred = glmnetPredict(fit, newx = X_test, ptype='class', s = scipy.array([0.05, 0.01]))
    print(y_pred)

    # print(X_train)
    # clf = linear_model.LogisticRegression(penalty='l1', class_weight='balanced')
    # clf = tree.DecisionTreeClassifier()

    # X_res, y_res = reSample(X_train, y_train)
    # print(X_train.shape, X_res.shape)

    # clf = ensemble.RandomForestClassifier()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)


    # Attack prediction evaluation
    # prec, rec, f1_score = sklearn.metrics.precision_score(y_actual_test, y_pred),\
                          # sklearn.metrics.recall_score(y_actual_test, y_pred), \
                          # sklearn.metrics.f1_score(y_actual_test, y_pred),

    # print(prec, rec, f1_score)

if __name__ == "__main__":
    main()
