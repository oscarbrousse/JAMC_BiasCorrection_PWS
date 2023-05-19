# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:13:45 2021

@author: oscar
"""
#%%
import pandas as pd
import numpy as np
# import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime

### Chose the name of your city of interest
city = 'London'
year = 2018
datadir_MIDAS = r'' + city + '/Filtered/'
savedir = r''

MIDAS_by_ID_df = pd.read_csv(datadir_MIDAS + 'Per_Variable/MIDAS_T_stations_' + city + '_Filt_by_ID.csv', 
                                  index_col=0)
MIDAS_by_ID_df = MIDAS_by_ID_df.set_index(pd.DatetimeIndex(MIDAS_by_ID_df.index))
MIDAS_by_ID_df = MIDAS_by_ID_df[(MIDAS_by_ID_df.index >= '2018-06-01') & (MIDAS_by_ID_df.index < '2018-09-01')]


fig, ax = plt.subplots(figsize = (12,4))

ax.plot(MIDAS_by_ID_df.groupby(MIDAS_by_ID_df.index.date).mean().mean(axis=1),
         linewidth=2, marker = '.', color = 'lightgrey', label = 'MIDAS', zorder = 2)
### Select St James park out of ID
ax.plot(MIDAS_by_ID_df['697'].groupby(MIDAS_by_ID_df.index.date).mean(),
         linewidth=2, marker = '.', color = 'darkgrey', label = 'St-James Park', zorder = 2)

ax.fill_between(np.unique(MIDAS_by_ID_df.index.date),
                MIDAS_by_ID_df.groupby(MIDAS_by_ID_df.index.date).min().mean(axis=1),
                MIDAS_by_ID_df.groupby(MIDAS_by_ID_df.index.date).max().mean(axis=1),
         facecolor = 'lightgrey', zorder = 0, alpha = 0.5,
         linewidth = 1, linestyle = '--', color = 'lightgrey')
ax.fill_between(np.unique(MIDAS_by_ID_df.index.date),
                MIDAS_by_ID_df['697'].groupby(MIDAS_by_ID_df.index.date).min(),
                MIDAS_by_ID_df['697'].groupby(MIDAS_by_ID_df.index.date).max(),
         facecolor = 'darkgrey', zorder = 1, alpha = 0.7,
         linewidth = 1, linestyle = '--', color = 'darkgrey')

ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1, 30, 30)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax.set_ylabel('T (°C)')
ax.legend()

fig.savefig(savedir + 'MIDAS_T_Jun-Aug_vs_StJames.png', dpi=300)
fig.savefig(savedir + 'MIDAS_T_Jun-Aug_vs_StJames.pdf')

# %%
