import sys
sys.path.append('../test')
sys.path.append('../lib')
from pyglmnet import GLM
import pandas as pd
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import sklearn.metrics


class ArgsStruct:
    LOAD_DATA = True

''' The following functions are for formatting the output in a numpy design
    matrix so that normal matrix computations are easier
'''


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
    train_start_date = pd.to_datetime(outputDf.iloc[0, 0]) # first row
    train_end_date = pd.to_datetime(outputDf.iloc[-1, 0]) # last row

    inputDf['date'] = pd.to_datetime(inputDf['date'])
    inputDf['date'] = inputDf['date'].dt.date

    outputDf['date'] = pd.to_datetime(outputDf['date'])
    outputDf['date'] = outputDf['date'].dt.date

    y_true = outputDf['attackFlag']

    # This can be validated through
    delta_prev_time = 7  # no of days to check before the week of current day

    currDate = train_start_date
    countDayIndx = 0

    num_features = len(inputDf.columns) - 1
    X = np.zeros((y_true.shape[0], delta_prev_time*num_features)) #input
    Y = -np.ones((y_true.shape[0], 1)) #output

    while (currDate < train_end_date):
        ''' This loop checks values on either of delta days prior '''
        for idx in range(delta_prev_time):
            historical_day = pd.to_datetime(currDate - datetime.timedelta(days=delta_prev_time)) # one week before
            try:
                X[countDayIndx, idx*num_features:(idx+1)*num_features] = (inputDf[inputDf['date'] == historical_day.date()].ix[:,1:]).values[0] # exclude date
            except:
                continue

        Y[countDayIndx] = outputDf[outputDf['date'] == currDate.date()].ix[:, 1]

        countDayIndx += 1
        currDate += datetime.timedelta(days=1)

    # Fill the missing cells with median of column
    # for col in range(X.shape[1]):
    #     X[np.where(X[:, col] == 0)[0], col] = np.median(X[~np.where(X[:, col] == -1)[0], col])

    return X, Y



def main():
    args = ArgsStruct()
    args.LOAD_DATA = True
    args.forumsSplit = False
    args.FEAT_TYPE = "REGULAR" # REGULAR: graph features, ANOM: anomaly features
    args.TYPE_PLOT = 'LINE'

    ''' SET THE TRAINING AND TEST TIME PERIODS - THIS IS MANUAL '''
    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')

    testStart_date = datetime.datetime.strptime('2017-06-01', '%Y-%m-%d')
    testEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    if args.LOAD_DATA:
        amEvents = pd.read_csv('../../../data/Armstrong_data/amEvents_11_17.csv')
        amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

        if args.FEAT_TYPE == "REGULAR":
            feat_df = pickle.load(open('../../../data/DW_data/features/feat_combine/features_Delta_T0_Mar16-Aug17.pickle', 'rb'))
        else:
            feat_df = pickle.load(
                open('../../data/DW_data/features/feat_combine/user_interStats_Delta_T0_Sep16-Aug17.pickle', 'rb'))

    # the previous month is needed for features - since we measure lag correlation for the prediction
    instance_TrainStartDate = trainStart_date - relativedelta(months=1) # the previous month is needed for features

    # Make the date common for all
    feat_df['date'] = pd.to_datetime(feat_df['date'])
    trainDf = feat_df[feat_df['date'] >= instance_TrainStartDate]
    trainDf = trainDf[trainDf['date'] < trainEnd_date]


    trainOutput = prepareOutput(amEvents_malware, trainStart_date, trainEnd_date)
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

    ''' For each feature perform the following:
        1. Prepare the time lagged longitudinal features
        2. Fit the GLM model
        3. Measure the accuracy


    '''

    for feat in trainDf.columns:
        if feat == 'date' or feat == 'forums':
            continue

        print('Computing for feature: ', feat)

        trainDf_curr = trainDf[['date', feat]]
        testDf_curr = testDf[['date', feat]]
        X_train, y_train = prepareData(trainDf_curr, trainOutput)
        X_test, y_test = prepareData(testDf_curr, testOutput)

        ''' Fit a GLM model with logit link on a binomial distribution data '''
        # TODO: Add sparsity for the single predictor models
        # TODO: FOr combined predictors models, add group lasso sparsity - how to define the groups


        print(X_train.shape, y_train.shape)
        y_train = y_train.astype(int)

        glm_LR = GLM(distr='multinomial', alpha=0.01, verbose=True)
        glm_LR.threshold = 1e-3
        glm_fit = (glm_LR.fit(X_train, y_train.flatten()))
        y_pred = glm_fit[-1].predict(X_train)
        print(y_train)
        print(y_pred)
        exit()
        y_pred = (np.array(y_pred)).transpose()

        prec, rec, f1_score = sklearn.metrics.precision_score(y_train, y_pred),\
        sklearn.metrics.recall_score(y_train, y_pred), \
        sklearn.metrics.f1_score(y_train, y_pred),

        print(prec, rec, f1_score)

        exit()

        ''' Second step - Fit a boosting model/Decision tree stumps for the residual subspace features'''



        ''' Final step: Should be a bagging/ensemble technique for combining the above two '''
        # TODO: explanatory analysis - which attacks can be detected by anomalies
        # TODO: anomaly correlation with count

        # X_train = preprocessing.normalize(X_train, norm='l2')
        # X_test = preprocessing.normalize(X_test, norm='l2')

        # glmnetPlot(fit, xvar='dev', label=True)
        # plt.show()

        # y_pred = glmnetPredict(fit, newx = X_test, ptype='class', s = scipy.array([0.05, 0.01]))
        # print(y_pred)

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
