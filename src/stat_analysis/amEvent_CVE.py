import datetime as dt
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt
import operator


def probBoxPlots(dataDict):
    '''
    dict: key - month; values = list of time differences
    :param dict:
    :return:
    '''

    titles = []
    data_to_plot = []
    sortedByDate = sorted(dataDict.items(), key=operator.itemgetter(0))

    for d, v in sortedByDate:
        titles.append(d)
        data_to_plot.append(v)

    # Create the box_plots
    fig = plt.figure(1, figsize=(10, 8))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True)

    for box in bp['boxes']:
        # change outline color
        box.set(color='#0000FF', linewidth=2)
        # change fill color
        box.set(facecolor='#FFFFFF')

        ## change color and linewidth of the whiskers
        # for whisker in bp['whiskers']:
        #     whisker.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the caps
        # for cap in bp['caps']:
        #     cap.set(color='#7570b3', linewidth=2)

        ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#FF0000', linewidth=4)

        ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

    # ax.set_title('Motif transition:' + str(m4) + '-->' + str(m5))
    ax.set_ylabel('Interval')
    # ax.set_ylim([0, 0.05])
    ax.set_xticklabels(titles)

    third_quartile = [item.get_ydata()[0] for item in bp['whiskers']]
    third_quartile = max(third_quartile)

    # dir_save = '../plots/motif_transition_plots/12_08/v2/inhib'
    # if not os.path.exists(dir_save):
    #     os.makedirs(dir_save)
    # file_save = dir_save + '/' + 'mt_inhib_' + str(m4) + '_' + str(m5) + '.png'

    plt.tick_params('y', labelsize=20)
    plt.tick_params('x', labelsize=20)
    plt.xlabel('Month in consideration', fontsize=25)
    plt.ylabel('Days', fontsize=25)
    plt.title('Time difference b/w amEvent & prev. 10 CVE mentions', fontsize=25)
    # if m4 not in limits_y_steep:
    #     limits_y_steep[m4] = {}
    # try:
    #     limits_y_steep[m4][m5] = third_quartile + math.pow(10, int(math.log10(third_quartile)))
    # except:
    #     limits_y_steep[m4][m5] = 0

    plt.grid(True)
    plt.show()
    # plt.savefig(file_save)
    plt.close()


def timeDiffCVE_amEvent(cveData, amData):
    '''
    Check the time difference distribution between the amstrong event and the most recent at most 10 CVEs
    that occured prior to that event
    :param cveData:
    :param amData:
    :return:
    '''
    # print(amData[:10])
    probDistTemporal = {}
    curr_month = 0
    curr_year = 0
    tempDist = []
    for idx, row in amData.iterrows():
        date_event = row['date']
        if idx == 0:
            curr_month = date_event.month
            curr_year = date_event.year

        if date_event.month != curr_month or date_event.year != curr_year:
            if curr_month < 10:
                strCurrMonth = str(0) + str(curr_month)
            else:
                strCurrMonth = str(curr_month)
            probDistTemporal[str(date_event.year) + '_' + strCurrMonth] = tempDist
            tempDist = []
            curr_month = date_event.month
            curr_year = date_event.year

        # print("Event: ")
        vuln_previous = cveData[cveData['postedDate'] < date_event]
        vuln_previous = vuln_previous.sort('postedDate')[-10: ]
        vuln_previousDates = vuln_previous['postedDate'].tolist()# 10 is a parameter

        for vp in vuln_previousDates:
            time_diff = date_event - vp
            time_diffStr = str(time_diff.days)
            tempDist.append(int(time_diffStr))
            # print(time_diff)
            # print(vuln_previous)

    return probDistTemporal


if __name__ == "__main__":
    read_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    amEvents = pd.read_csv(read_path)
    amEvents['date'] = pd.to_datetime(amEvents['date'], format="%Y-%m-%d")
    amEvents = amEvents[amEvents['date'] < pd.to_datetime('2016-12-01')]

    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))
    vulData['postedDate'] = pd.to_datetime(vulData['postedDate'], format="%Y-%m-%d")
    # print(vulData)
    # print(amEvents)

    startDate = pd.to_datetime("2016-04-01", format="%Y-%m-%d")
    endDate = pd.to_datetime("2016-09-01", format="%Y-%m-%d")

    vulDataFiltered = vulData[vulData['postedDate'] >= startDate]
    vulDataFiltered = vulDataFiltered[vulDataFiltered['postedDate'] < endDate]

    probDistDict = timeDiffCVE_amEvent(vulDataFiltered, amEvents)
    probBoxPlots(probDistDict)