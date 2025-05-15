import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

hour = 2
path = root_path + "/data/test_station/FFT/heating_%sh/" % hour

linewidth = 1.2
station_font_size = 21
label_size = 19
stations = ['ABK', 'OTT', 'FRD']
models = ['GeoDGP', 'Geospace']
plt.tight_layout()
plt.rcParams.update({'font.size': 15})
fig, axes = plt.subplots(3, 1, figsize=(12, 16))
for station in stations:
    data = []
    index = np.where(np.array(stations) == station)[0][0]
    for model in models:
        filename = path+ 'fft_event20_%s_%s_integral.txt' % (model, station)
        df_model = pd.read_csv(filename,delimiter='\s+')
        df_model.columns = ['Time', 'Obs', '%s' % model]
        data.append(df_model)
    data = pd.merge(data[0], data[1])
    UQ_file_path = path + '../../FFT_withUQ/E_integral_UQ/fft_event20_GeoDGP200_%s_integral.txt' % station
    station_UQ_df = pd.read_csv(UQ_file_path, delimiter='\s+')
    station_array = station_UQ_df.pivot_table(
        index=station_UQ_df.groupby('realization').cumcount(),
        columns='realization',
        values='Integral_sim'
    ).to_numpy().T
#     std = np.std(station_array, axis=0, ddof=1)
#     qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
    q25 = np.quantile(station_array, 0.025, axis=0)
    q975 = np.quantile(station_array, 0.975, axis=0)
    mean = np.mean(station_array, axis=0)
    ax = axes.flat[index]
    ax.plot(data['Time'], data['Geospace'], color = "royalblue", label = "Geospace", linewidth=linewidth)
    ax.plot(data['Time'], mean, color = "red", label = "GeoDGP", linewidth=linewidth)
#     ax.plot(data['Time'], data['GeoDGP'], color = "red", label = "GeoDGP_dBHpredmean", linewidth=linewidth)
    ax.plot(data['Time'], data['Obs'], color = "black", label = "Station", linewidth=linewidth)
#     ax.fill_between(data['Time'],
#                     np.clip(data['GeoDGP'] - qnorm_975*std, 0, np.inf),
#                     data['GeoDGP'] + qnorm_975*std,
#                     color='gray', alpha=0.3, label='95% PI')
    ax.fill_between(data['Time'],
                    q25,
                    q975,
                    color='gray', alpha=0.3)
    ax.text(0.82, .95, station, ha='left', va='top', transform=ax.transAxes, fontsize=station_font_size)
    if index == len(stations)-1:
        params = {'legend.fontsize': label_size, 'legend.handlelength': 3}
        plt.rcParams.update(params)
        handles, labels = plt.gca().get_legend_handles_labels()
        leg = plt.figlegend(handles, labels, loc = 'lower center', ncol=3,
                            labelspacing=12, frameon=False)
        for legobj in leg.legend_handles:
            legobj.set_linewidth(3)
        if hour == 2:
            fig.text(0.05, 0.5, '$E\'^2$ Bi-Hourly Integral', va='center', rotation='vertical', fontsize=label_size)
        else:
            fig.text(0.02, 0.5, '$E^2$ Hourly Integral', va='center', rotation='vertical', fontsize=label_size)
        fig.text(0.5, 0.07, 'Hours from May 10 2024', ha='center', rotation='horizontal', fontsize=label_size)
        plt.savefig(root_path + '/figure/TestSet3/TestSet3_FFT_Heating_%sh.png' % hour,
                    bbox_inches='tight', dpi=300)
        plt.close()