from matplotlib.dates import DateFormatter
import pandas as pd
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv
import sys
import matplotlib.dates as mdates

class ArgsStruct:
    name = ''
    plot_data = False

maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True


def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum




# def weeklyCVE_event_corr():


def loadVulnInfo(df):
    vuln_groups= df.groupby(['vulnerabilityid'])


def analyse_events(df_data):
    args = ArgsStruct()
    args.plot_data = True
    print(df_data[:10])

    if args.plot_data == True:
        df_data['start_dates'] = df_data['start_dates'].dt.date

        title = 'Weekly distribution of Armstrong attack counts: Malicious email'
        fig, ax = plt.subplots()
        df_data.plot.bar(ax=ax, x='start_dates', y='number_attacks', color='black', linewidth=2, )
        plt.grid(True)
        plt.xticks(size=20)
        plt.yticks(size=20)
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.title(title, size=20)
        plt.xlabel('Start dates (Week)', size=20)
        plt.ylabel('# attacks', size=20)
        plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
        # file_save = plot_dir + feat + title + '.png'
        # plt.savefig(file_save)
        plt.show()
        plt.close()

    vuln_counts_list = []
    for idx, row in df_data.iterrows():
        vuln_counts = np.sum(np.array(row['vuln_counts']))
        vuln_counts_list.append(vuln_counts)

    df_data['vuln_counts_sum'] = vuln_counts_list

    if args.plot_data == True:
        df_data['start_dates'] = df_data['start_dates']
        title = 'Weekly distribution of Vulnerability mentions: DW'
        fig, ax = plt.subplots()
        df_data.plot.bar(ax=ax, x='start_dates', y='vuln_counts_sum', color='black', linewidth=2, )
        plt.grid(True)
        plt.xticks(size=20)
        plt.yticks(size=20)
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.title(title, size=20)
        plt.xlabel('Start dates (Week)', size=20)
        plt.ylabel('# vulnerabilties mentioned', size=20)
        plt.subplots_adjust(left=0.13, bottom=0.25, top=0.9)
        # file_save = plot_dir + feat + title + '.png'
        # plt.savefig(file_save)
        plt.show()
        plt.close()

    return df_data


def main():
    trainStart_date = datetime.datetime.strptime('2016-04-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')


    cve_eventsDf = pd.read_pickle('../data/DW_data/CPE_events_corr_me.pickle')
    print(cve_eventsDf)



if __name__ == "__main__":
    main()

