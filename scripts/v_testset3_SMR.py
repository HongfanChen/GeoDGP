#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import pandas as pd
import datetime
from datetime import datetime
import numpy as np
import datetime
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
def convert_to_datetime(row):
    return pd.to_datetime(datetime.datetime(year=int(row[0]), month=int(row[1]), day=int(row[2]),
                             hour=int(row[3]), minute=int(row[4]), second=int(row[5])
                                           ))
smr = pd.read_csv(root_path + "/data/Input/TestSet3/SMR.txt",
                 skiprows=1, delimiter='\s+')
smr['Datetime'] = smr.apply(convert_to_datetime, axis=1)
smr.set_index('Datetime', inplace=True)
min_time = smr.index[np.where(smr['dst_sm'] == np.min(smr['dst_sm']))[0][0]]

def convert_to_datetime(row):
    return pd.to_datetime(datetime.datetime(year=int(row[0]), month=int(row[1]), day=int(row[2]),
                             hour=int(row[3]), minute=int(row[4]), second=int(row[5])
                                           ))
df = pd.read_csv(root_path + "/data/Input/TestSet3/IMF_ballistic_WIND_fixBz.dat",
                 skiprows=12, delimiter='\s+', header=None)
df.columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond',
              'Bx, nT (GSE, GSM)', 'By, nT (GSM)', 'Bz, nT (GSM)',
              'Vx Velocity, km/s, GSE', 'Vy', 'Vz',
             'Proton Density, n/cc', 'Temperature, K']
df['Datetime'] = df.apply(convert_to_datetime, axis=1)
df.set_index('Datetime', inplace=True)
regular_time = [datetime.datetime(2024, 5, 10, 8, 0) + datetime.timedelta(minutes = i) for i in range(3840)]
index_list = []
for time in regular_time:
    time_diff = df.index - time
    min_index = np.where(np.abs(time_diff) == np.min(np.abs(time_diff)))[0][0]
    index_list.append(df.index[min_index])
df = df.loc[index_list]
df['Datetime'] = regular_time
df.set_index('Datetime', inplace=True)
WIND = df[['Bx, nT (GSE, GSM)', 'By, nT (GSM)', 'Bz, nT (GSM)', 'Vx Velocity, km/s, GSE',
         'Proton Density, n/cc', 'Temperature, K']]

def convert_to_datetime(row):
    return pd.to_datetime(datetime.datetime(year=int(row[0]), month=int(row[1]), day=int(row[2]),
                             hour=int(row[3]), minute=int(row[4]), second=int(row[5]),
                            microsecond = int(row[6])
                                           ))
df = pd.read_csv(root_path + "/data/Input/TestSet3/IMF.dat", skiprows=7, delimiter='\s+', header=None)
df.columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond',
              'Bx, nT (GSE, GSM)', 'By, nT (GSM)', 'Bz, nT (GSM)',
              'Vx Velocity, km/s, GSE', 'Vy', 'Vz',
             'Proton Density, n/cc', 'Temperature, K', 'F107']
df['Datetime'] = df.apply(convert_to_datetime, axis=1)
df.set_index('Datetime', inplace=True)
regular_time = [datetime.datetime(2024, 5, 10, 8, 0) + datetime.timedelta(minutes = i) for i in range(3840)]
index_list = []
for time in regular_time:
    time_diff = df.index - time
    min_index = np.where(np.abs(time_diff) == np.min(np.abs(time_diff)))[0][0]
    index_list.append(df.index[min_index])
df = df.loc[index_list]
df['Datetime'] = regular_time
df.set_index('Datetime', inplace=True)
ACE = df[['Bx, nT (GSE, GSM)', 'By, nT (GSM)', 'Bz, nT (GSM)', 'Vx Velocity, km/s, GSE',
         'Proton Density, n/cc', 'Temperature, K']]
Joint_index = smr.index.intersection(ACE.index)
Joint_index = Joint_index.intersection(WIND.index)
time_start=datetime.datetime(2024,5,10,16)
time_end=datetime.datetime(2024,5,11,6)
time_index = smr['dst_sm'][Joint_index][time_start:time_end].index
smr = smr.loc[time_index]
ACE = ACE.loc[time_index]
WIND = WIND.loc[time_index]
ylabel_size = 30
yticks_size = 25
xticks_size = 23
rotate_angle = 15
linewidth = 2
alpha=0.7
ACE_color = 'red'
WIND_color = 'mediumblue'
vline_color = 'black'
hline_color = 'red'
plt.figure(figsize=(30, 24))
plt.subplot(3,2,1)
plt.plot(time_index, smr['dst_sm'], color='black', linewidth=linewidth)
plt.axvline(x = smr.index[np.where(smr['dst_sm'] == np.min(smr['dst_sm']))[0][0]],
            color = vline_color, linestyle='dotted', linewidth=3, alpha=alpha)
plt.axhline(y = np.min(smr['dst_sm']),
            color = hline_color, linestyle='dotted', linewidth=3, alpha=0.7)
plt.annotate('%0.1f nT' % np.min(smr['dst_sm']), xy=(0.82, np.min(smr['dst_sm'])+12), xytext=(8, 0), 
             xycoords=('axes fraction', 'data'), textcoords='offset points', size=25, color=hline_color)
plt.annotate(min_time.strftime("%Y-%m-%d %H:%M:%S UT"), xy=(0.1, 260), xytext=(8, 0), 
             xycoords=('axes fraction', 'data'), textcoords='offset points', size=35, color=vline_color)
plt.ylabel('SMR [nT]',size=ylabel_size)
plt.ylim(-500,250)
plt.xticks(size= xticks_size)
plt.xticks(rotation=rotate_angle)
plt.yticks(size= yticks_size)
plt.subplot(3,2,2)
plt.plot(time_index, ACE['Bx, nT (GSE, GSM)'],linewidth=linewidth, color=ACE_color, label='ACE')
plt.plot(time_index, WIND['Bx, nT (GSE, GSM)'],linewidth=linewidth, color=WIND_color, label='WIND')
plt.ylabel('$B_x$ [nT]',size=ylabel_size)
plt.axvline(x = smr.index[np.where(smr['dst_sm'] == np.min(smr['dst_sm']))[0][0]],
            color = vline_color, linestyle='dotted', linewidth=3, alpha=alpha)
plt.xticks(size= xticks_size)
plt.xticks(rotation=rotate_angle)
plt.yticks(size= yticks_size)
plt.subplot(3,2,3)
plt.plot(time_index, ACE['Proton Density, n/cc'],linewidth=linewidth, color=ACE_color, label='ACE')
plt.plot(time_index, WIND['Proton Density, n/cc'],linewidth=linewidth, color=WIND_color, label='WIND')
plt.ylabel('$N_p$ [n/cc]',size=ylabel_size)
plt.axvline(x = smr.index[np.where(smr['dst_sm'] == np.min(smr['dst_sm']))[0][0]],
            color = vline_color, linestyle='dotted', linewidth=3, alpha=alpha)
plt.xticks(size= xticks_size)
plt.xticks(rotation=rotate_angle)
plt.yticks(size= yticks_size)
# plt.legend(fontsize=15)
plt.subplot(3,2,4)
plt.plot(time_index, ACE['By, nT (GSM)'],linewidth=linewidth, color=ACE_color, label='ACE')
plt.plot(time_index, WIND['By, nT (GSM)'],linewidth=linewidth, color=WIND_color, label='WIND')
plt.ylabel('$B_y$ [nT]',size=ylabel_size)
plt.axvline(x = smr.index[np.where(smr['dst_sm'] == np.min(smr['dst_sm']))[0][0]],
            color = vline_color, linestyle='dotted', linewidth=3, alpha=alpha)
plt.xticks(size= xticks_size)
plt.xticks(rotation=rotate_angle)
plt.yticks(size= yticks_size)
plt.subplot(3,2,5)
plt.plot(time_index, ACE['Vx Velocity, km/s, GSE'], linewidth=linewidth, color=ACE_color, label='ACE')
plt.plot(time_index, WIND['Vx Velocity, km/s, GSE'], linewidth=linewidth, color=WIND_color, label='WIND')
plt.ylabel('$V_x$ [km/s]',size=ylabel_size)
plt.axvline(x = smr.index[np.where(smr['dst_sm'] == np.min(smr['dst_sm']))[0][0]],
            color = vline_color, linestyle='dotted', linewidth=3, alpha=alpha)
plt.xticks(size= xticks_size)
plt.xticks(rotation=rotate_angle)
plt.yticks(size= yticks_size)
plt.subplot(3,2,6)
plt.plot(time_index, ACE['Bz, nT (GSM)'],linewidth=linewidth, color=ACE_color, label='ACE')
plt.plot(time_index, WIND['Bz, nT (GSM)'],linewidth=linewidth, color=WIND_color, label='WIND')
plt.ylabel('$B_z$ [nT]',size=ylabel_size)
# plt.xlabel('Time',size=ylabel_size)
plt.axvline(x = smr.index[np.where(smr['dst_sm'] == np.min(smr['dst_sm']))[0][0]],
            color = vline_color, linestyle='dotted', linewidth=3, alpha=alpha)
plt.xticks(size= xticks_size)
plt.xticks(rotation=rotate_angle)
plt.yticks(size= yticks_size)

## Legend, ticks and title: -----------------------
params = {'legend.fontsize': 40,
      'legend.handlelength': 4}
plt.rcParams.update(params)
date_format = DateFormatter('%m-%d %H')
for ax in plt.gcf().axes:
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8)) 
    ax.spines['top'].set_linewidth(2)  # Adjust the thickness of the top spine
    ax.spines['bottom'].set_linewidth(2)  # Adjust the thickness of the bottom spine
    ax.spines['left'].set_linewidth(2)  # Adjust the thickness of the left spine
    ax.spines['right'].set_linewidth(2)
# Define custom legend handles with desired line width
custom_handles = [
    Line2D([0], [0], color=ACE_color, linewidth=5, label='ACE'),
    Line2D([0], [0], color=WIND_color, linewidth=5, label='WIND')
]

# Update the figure legend
plt.figlegend(custom_handles, ['ACE', 'WIND'], loc='lower center', ncol=2, frameon=False)
# handles, labels = plt.gca().get_legend_handles_labels()
# plt.figlegend(handles, labels, loc = 'lower center', ncol=2, labelspacing=1, markerscale=5, frameon=False)
plt.savefig(root_path + "/figure/TestSet3/SMR.png", bbox_inches='tight', dpi=300)
