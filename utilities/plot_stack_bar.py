import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns


def stack_plot(df):
    fig, ax = plt.subplots(figsize=(15, 10))

    margin_bottom = np.zeros(len(df['start_date']))
    colors = ["2F4F4F", "808080", "A52A2A", "DEB887", "1E90FF", "B0E0E6", "FFD700"]

    for idx, col in enumerate(df.columns):
        values = list(df[df['Month'] == month].loc[:, 'Value'])

        df.plot.bar(x='start_date', y=col, ax=ax, stacked=True,
                                          bottom=margin_bottom, color=colors[idx], label=col)
        margin_bottom += values

    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    data_plot = pd.read_csv('../data/DW_data/new_DW/cpe_Counts_plot_weekly_main.csv')

    stack_plot(data_plot)