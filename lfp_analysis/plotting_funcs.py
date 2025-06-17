import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import time as dttime
import seaborn as sns

TIME_INDEX = [dttime(i // 60, i % 60) for i in range(0, 1440, 10)]

def plot_daily_r2(pt, df):

    colors = np.array(['gold', 'r', 'orange', 'b', 'gray'])
    pt_df = df.query('pt_id==@pt and (lead_location=="VC/VS" or lead_location=="OTHER")')
    pt_df_head = pt_df.groupby(pd.Grouper(key='CT_timestamp', freq='D')).head(1)

    plt.figure(figsize=(15,3))
    plt.scatter(pt_df_head['days_since_dbs'], pt_df_head['lfp_left_day_r2_OvER'], s=10, c=colors[pt_df_head['state_label']])

    plt.grid()
    plt.gca().set(xlabel='Days since DBS', ylabel='Daily R2', title=f'{pt_df.loc[pt_df.index[0], "pt_id"]}')
    plt.gca().set_xticks(plt.gca().get_xticks())
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha='right')
    plt.show()
    return

def plot_lfp_heatmap(pt, df):
    pt_df = df.query('pt_id==@pt and (lead_location=="VC/VS" or lead_location=="OTHER")')
    days = pt_df['days_since_dbs'].drop_duplicates().dropna()
    pt_lfp = pd.DataFrame(columns = days, index = [dttime(i // 60, i % 60) for i in range(0, 1440, 10)])
    for day in days:
        day_df = pt_df[pt_df['days_since_dbs'] == day]
        day_df.loc[:,'time_bin'] = day_df['time_bin'] - pd.Timedelta(6, unit='h')
        day_df['time_bin'] = day_df['time_bin'].dt.time
        for time_bin, value in day_df[['time_bin', 'lfp_left_z_scored_OvER']].values:
            if time_bin in pt_lfp.index:
                pt_lfp.loc[time_bin, day] = value

    pt_lfp = pt_lfp.astype(float)
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(len(pt_lfp.columns)*1, 7))
    p = axs.imshow(pt_lfp, aspect=1.2, cmap='jet')
    cbar = plt.colorbar(p, ax=axs)
    cbar.set_label('9 Hz LFP (z-scored)', rotation=270)

    plt.xlabel('Day')
    plt.ylabel('Time of Day')
    #plt.xticks(np.arange(0, len(pt_lfp.columns), len(pt_lfp.columns) // 10), pt_lfp.columns[::11])
    plt.yticks(np.arange(0, len(pt_lfp.index), 71), ['0:00', '12:00', '24:00'])
    plt.title(pt)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.tight_layout()
    plt.show()