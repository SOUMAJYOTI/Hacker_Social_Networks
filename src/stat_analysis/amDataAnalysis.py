import datetime as dt
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt
import operator
import datetime
import numpy as np

def plot_bars(data, titles):
    width=0.35
    ind = np.arange(len(data))  # the x locations for the groups
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ## the bars
    # rects1 = ax.bar(ind, data_mean, width,
    #                 color='#C0C0C0')
    rects1 = ax.bar(ind, data, width,
                    color='#0000ff')  # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    # ax.set_ylim(0, 1)
    ax.set_ylabel('Number of evnets', size=40)
    ax.set_xlabel('Date frame (start of each week)', size=30)
    ax.set_title('Weekly distribution of events (each month)', size=30)
    xTickMarks = titles
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=5, ha='right')
    plt.grid(True)
    plt.xticks(size=25)
    plt.yticks(size=25)
    plt.subplots_adjust(left=0.13, bottom=0.30, top=0.9)
    ## add a legend
    # ax.legend( (rects1[0], ('Men', 'Women') )

    plt.show()


def plot_week_events(df_plot):
    # df_plot = df_plot[df_plot['date'] < '2016-10-01']
    # df_plot['week'] = df_plot['date'].map(lambda x: x.isocalendar()[1])
    # df_plot['year'] = df_plot['date'].map(lambda x: x.isocalendar()[0])
    # data = df_plot.groupby(['year', 'week']).count()

    titles_list = []
    start_year = '2016'
    start_day = 1
    start_month = 4
    daysMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    count_Events = []
    while True:
        if start_day < 10:
            start_dayStr = str('0') + str(start_day)
        else:
            start_dayStr = str(start_day)

        if start_month < 10:
            start_mnStr = str('0') + str(start_month)
        else:
            start_mnStr = str(start_month)

        numDaysCurrMonth = daysMonths[start_month-1]
        start_date = str(start_year)+'-'+str(start_mnStr)+'-'+start_dayStr

        end_day = start_day + 7
        if end_day > numDaysCurrMonth:
            end_day = numDaysCurrMonth

        if end_day < 10:
            end_dayStr = str('0') + str(end_day)
        else:
            end_dayStr = str(end_day)
        end_date = str(start_year) + '-' + str(start_mnStr) + '-' + end_dayStr

        df_week = df_plot[df_plot['date'] >= start_date]
        df_week= df_week[df_week['date'] < end_date]

        # print(start_date, end_date)
        count_Events.append(len(df_week))
        titles_list.append(start_date)
        start_day = end_day
        if start_day >= 29:
            start_month += 1
            start_day = 1

        if start_month == 10:
            break

    print(titles_list)
    plot_bars(count_Events, titles_list)
    exit()
    data['date'].plot(kind="bar")
    plt.grid(True)
    plt.xticks(rotation=60, size=20)
    plt.yticks(size=20)
    plt.xlabel("Date Timeframe (Year, Week #)", size=25)
    plt.ylabel("# of events", size=25)
    plt.title('Weekly distribution of events', size=25)
    plt.show()


if __name__ == "__main__":
    read_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    amEvents = pd.read_csv(read_path)
    amEvents['date'] = pd.to_datetime(amEvents['date'], format="%Y-%m-%d")
    # print(amEvents)
    # amEvents = amEvents[amEvents['date'] < pd.to_datetime('2016-12-01')]
    plot_week_events(amEvents)

    # print(len(list(set(amEvents['event_type']))))
