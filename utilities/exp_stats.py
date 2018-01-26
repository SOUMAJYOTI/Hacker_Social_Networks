import sys
import pandas as pd
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np


def main():
    featList = ['numUsers', 'numVulnerabilities', 'numThreads', 'expertsThreads',
            ]

    # featList = [ 'expert_NonInteractions',
    #             'communityCount', 'shortestPaths', 'CondExperts']

    delta_gap_time = [7,  ]
    delta_prev_time_start = [ 15, 21,]

    thresh_anom_count = 1
    for dgt in delta_gap_time:

        for dprev in delta_prev_time_start:
            data = pickle.load(open('../data/results/01_09/anomaly/thresh_anom_' + str(thresh_anom_count) + '/' +
                                  str('res_') + 'tgap_' + str(dgt) + '_tstart_' + str(dprev) + '.pickle', 'rb'))
            print('Computing for dprev: ', dprev)

            for feat in featList:
                print(feat, data[feat + '_res_vec']['tprList'])
                print(feat, data[feat + '_res_vec']['fprList'])


if __name__ == "__main__":
    main()