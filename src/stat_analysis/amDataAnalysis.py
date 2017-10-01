import datetime as dt
import pandas as pd
import requests
import pickle
import matplotlib.pyplot as plt
import operator


def plot_week_events(df_plot):
    df_plot['week'] = df_plot['date'].map(lambda x: x.isocalendar()[1])
    df_plot['year'] = df_plot['date'].map(lambda x: x.isocalendar()[0])
    data = df_plot.groupby(['year', 'week']).count()
    print(data)

    # exit()
    data['date'].plot(kind="bar")
    plt.grid(True)
    plt.xticks(rotation=45, size=20)
    plt.yticks(size=20)
    plt.xlabel("Date Timeframe (Year, Week #)", size=25)
    plt.ylabel("# of events", size=25)
    plt.title('Weekly distribution of events', size=25)
    plt.show()


if __name__ == "__main__":
    read_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    amEvents = pd.read_csv(read_path)
    amEvents['date'] = pd.to_datetime(amEvents['date'], format="%Y-%m-%d")
    # amEvents = amEvents[amEvents['date'] < pd.to_datetime('2016-12-01')]
    # plot_week_events(amEvents)

    print(len(list(set(amEvents['event_type']))))
