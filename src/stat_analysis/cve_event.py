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


def weeklyCVE_event_corr(eventsDf, vulnInfo, cve_cpe_map, start_date, end_date):
    '''

    :param eventsDf:
    :param attackdf:
    :param vulnInfo:
    :param cve_cpe_map:
    :param start_date:
    :param end_date:
    :return:
    '''
    eventsDf['date'] = pd.to_datetime(eventsDf['date'])
    vulnInfo['posteddate'] = pd.to_datetime(vulnInfo['posteddate'])

    startWeek = start_date
    endWeek = startWeek + datetime.timedelta(days=61)

    # DS Structures
    startDatesList = []
    endDatesList = []
    vulnIdsList = []
    vulnCounstList = []
    CPEsList = []
    CPECountsList = []
    numAttacks = []

    while(endWeek < end_date):

        ''' For the darkweb data '''
        vulnsCurr = {}
        cpesCurr = {}
        # try:
        vulWeek = vulnInfo[vulnInfo['posteddate'] >= startWeek]
        vulWeek = vulWeek[vulWeek['posteddate'] < endWeek]

        for idx, row in vulWeek.iterrows():
            cve = row['vulnerabilityid']
            if cve not in vulnsCurr:
                vulnsCurr[cve] = 0
            vulnsCurr[cve] += 1

            if cve in cve_cpe_map:
                cpes = cve_cpe_map[cve]
                for cp in cpes:
                    if cp not in cpesCurr:
                        cpesCurr[cp] = 0
                    cpesCurr[cp] += 1

        # except:
        #     pass

        ''' For the armstrong data '''
        eventsCurr = eventsDf[eventsDf['date'] >= startWeek]
        eventsCurr = eventsCurr[eventsCurr['date'] < endWeek]
        total_count = pd.DataFrame(eventsCurr.groupby(['date']).sum())
        count_attacks = np.sum(total_count['count'].values)

        if count_attacks == 0:
            attacksCurr = 0
        else:
            attacksCurr = count_attacks


        startDatesList.append(startWeek)
        endDatesList.append(endWeek)
        vulnIdsList.append(list(vulnsCurr.keys()))
        vulnCounstList.append(list(vulnsCurr.values()))
        CPEsList.append(list(cpesCurr.keys()))
        CPECountsList.append(list(cpesCurr.values()))
        numAttacks.append(attacksCurr)

        startWeek = endWeek
        endWeek = startWeek + datetime.timedelta(days=61)

    outputDf = pd.DataFrame()
    outputDf['start_dates'] = startDatesList
    outputDf['end_dates'] = endDatesList
    outputDf['vulnerabilities'] = vulnIdsList
    outputDf['vuln_counts'] = vulnCounstList
    outputDf['CPEs'] = CPEsList
    outputDf['CPE_counts'] = CPECountsList
    outputDf['number_attacks'] = numAttacks

    return outputDf


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


def firstMentions_CVE(vulnDf):
    '''

    :param vulnDf:
    :return:
    '''

    ''' Find and store the list of vulnerability mention dates for each one of them '''
    vulnMentions = {}
    for idx, row in vulnDf.iterrows():
        vid = row['vulnerabilityid']
        if  vid not in vulnMentions:
            vulnMentions[vid] = []

        if type(row['posteddate']) is not str:
            continue
        vulnMentions[vid].append(datetime.datetime.strptime(row['posteddate'], '%Y-%m-%d'))


    ''' Return the sorted dates of mentions for each vulnerability '''
    for v in vulnMentions:
        datesMentions = list(set(vulnMentions[v]))
        sortedDates = sorted(datesMentions)

        # Consider the first few mentions of each CVE
        if 0.5 * len(sortedDates) > 1:
            vulnMentions[v] = sortedDates[:int(0.1*len(sortedDates))]

    return vulnMentions


def weeklyCVE_fisrtMentions_event_corr(eventsDf, vulnInfo, start_date, end_date, vulnDates):
    '''

    :param eventsDf:
    :param vulnInfo:
    :param start_date:
    :param end_date:
    :param vulnDates:
    :return:
    '''
    eventsDf['date'] = pd.to_datetime(eventsDf['date'])
    vulnInfo['posteddate'] = pd.to_datetime(vulnInfo['posteddate'])

    startWeek = start_date
    endWeek = startWeek + datetime.timedelta(days=7)

    # DS Structures
    startDatesList = []
    endDatesList = []
    vulnCounstListFirst = []
    vulnCounstList = []
    numAttacks = []

    sum_attacks = 0
    total_attack_days = 0
    while(endWeek < end_date):

        ''' For the darkweb data '''
        vulnsCount = 0
        vulnsCountsFirst = 0
        # try:
        vulWeek = vulnInfo[vulnInfo['posteddate'] >= startWeek]
        vulWeek = vulWeek[vulWeek['posteddate'] < endWeek]

        for idx, row in vulWeek.iterrows():
            cve = row['vulnerabilityid']
            vulnsCount += 1
            for dm in vulnDates[cve]:
                if dm >= startWeek and dm < endWeek:
                    vulnsCountsFirst += 1
                    break

        # except:
        #     pass

        ''' For the armstrong data '''
        eventsCurr = eventsDf[eventsDf['date'] >= startWeek]
        eventsCurr = eventsCurr[eventsCurr['date'] < endWeek]
        total_count = pd.DataFrame(eventsCurr.groupby(['date']).sum())
        count_attacks = np.sum(total_count['count'].values)


        if count_attacks == 0:
            attacksCurr = 0
        else:
            sum_attacks += count_attacks
            total_attack_days += 1

            if count_attacks > 5:
                attacksCurr = count_attacks


                startDatesList.append(startWeek)
                endDatesList.append(endWeek)
                vulnCounstList.append(vulnsCount)
                vulnCounstListFirst.append(vulnsCountsFirst)
                numAttacks.append(attacksCurr)

        startWeek = endWeek
        endWeek = startWeek + datetime.timedelta(days=7)

    outputDf = pd.DataFrame()
    outputDf['start_dates'] = startDatesList
    outputDf['end_dates'] = endDatesList
    outputDf['vuln_counts'] = vulnCounstList
    outputDf['vuln_counts_first'] = vulnCounstListFirst
    outputDf['number_attacks'] = numAttacks

    print("Attack average: ", sum_attacks, total_attack_days, sum_attacks/total_attack_days)
    return outputDf


def main():
    amEvents = pd.read_csv('../../data/Armstrong_data/amEvents_11_17.csv')
    amEvents_malware = amEvents[amEvents['type'] == 'malicious-email']

    vuln_df = pd.read_csv('../../data/DW_data/VulnInfo_11_17.csv', encoding='ISO-8859-1', engine='python')
    cve_cpe_map = pickle.load(open('../../data/DW_data/new_DW/cve_cpe_map_new.pickle', 'rb'))

    trainStart_date = datetime.datetime.strptime('2016-10-01', '%Y-%m-%d')
    trainEnd_date = datetime.datetime.strptime('2017-09-01', '%Y-%m-%d')

    # outputDf = weeklyCVE_event_corr(amEvents_malware, vuln_df, cve_cpe_map, trainStart_date, trainEnd_date)

    vulnDatesList = firstMentions_CVE(vuln_df)
    outputDf = weeklyCVE_fisrtMentions_event_corr(amEvents_malware, vuln_df,  trainStart_date,
                                                  trainEnd_date, vulnDatesList)

    print(outputDf)
    # pickle.dump(outputDf, open('../../data/DW_data/CVE_mentions_events_corr_me.pickle', 'wb'))

    # cve_eventsDf = pd.read_pickle('../../data/DW_data/CPE_events_corr.pickle')
    # outputDf = analyse_events(cve_eventsDf)
    # pickle.dump(outputDf, open('../../data/DW_data/CPE_events_corr.pickle', 'wb'))


if __name__ == "__main__":
    main()

