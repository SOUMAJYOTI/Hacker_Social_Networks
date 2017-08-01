import datetime as dt
import pandas as pd
import requests
import json
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import os
import pycyr3con


# Plot utilities for bars chart
def plot_bars(x, y, x_label, y_label, col, xTitles=[]):
    width = 1
    plt.bar(x, y, width, color=col)
    if len(xTitles) > 0:
        major_ticks = np.arange(0, len(xTitles), 2)
        labels = []
        for i in major_ticks:
            labels.append(str(xTitles[i])[:10])

        plt.xticks(major_ticks, labels, rotation=45, size=20)
    else:
        plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel(x_label, size=25)
    plt.ylabel(y_label, size=25)
    # plt.title('Month-wise post counts', size=20)

    plt.subplots_adjust(left=0.13, bottom=0.25, top=0.95)
    plt.grid(True)
    plt.show()


def plot_am_data(df_plot):
    df_plot.groupby([df_plot["date"].dt.year, df_plot["date"].dt.month]).count().plot(kind="bar")
    plt.grid(True)
    plt.xticks(rotation=45, size=20)
    plt.yticks(size=20)
    plt.xlabel("Date Timeframe", size=25)
    plt.ylabel("# of events", size=25)
    plt.show()


def load_am_data(filePaths):
    """ filePath - path and file of the json/txt/csv file """
    dates_occurred = []
    attacker_filenames = []
    for fp in filePaths:
        ext = os.path.splitext(fp)[-1].lower()
        if ext == ".json":
            with open(fp) as data_file:
                data = json.load(data_file)

            for e in data["events"]:
                dates_occurred.append(e["occurred"][:10])
                if "files" in e:
                    if e["files"][0]["filename"] not in attacker_filenames:
                        attacker_filenames.append(e["files"][0]["filename"])
                elif "other_files" in e:
                    if e["other_files"][0]["filename"] not in attacker_filenames:
                        attacker_filenames.append(e["other_files"][0]["filename"])

    df = pd.DataFrame()
    # df["date"] = dates_occurred
    # df["date"] = df['date'].astype('datetime64')
    df["attack_file"] = attacker_filenames

    #plot_am_data(df)
    return df


# if __name__ == "__main__":
#     fPaths = ['../data/Armstrong_data/release/release/data.json',
#               '../data/Armstrong_data/release/release2/data.json']
#     # load_am_data(fPaths)
#     load_data_sdk()
