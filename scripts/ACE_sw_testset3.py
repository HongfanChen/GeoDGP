#!/usr/bin/env python
# coding: utf-8

# # Let's start from here if locally
## Note: ignore the variable naming here. The data is from ACE.
import time
from rich.progress import track
import pandas as pd
import datetime
import pytplot
import geopack.geopack as gp
from datetime import datetime
import numpy as np
import datetime
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.download import *

def convert_to_datetime(row):
    return pd.to_datetime(datetime.datetime(year=int(row[0]), month=int(row[1]), day=int(row[2]),
                             hour=int(row[3]), minute=int(row[4]), second=int(row[5]),
                            microsecond = int(row[6])
                                           ))
## this is the ACE measuremernt after ballistic propagation.
df = pd.read_csv(root_path+"/data/Input/TestSet3/IMF.dat", skiprows=7, delimiter='\s+', header=None)
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
df = df[['Bx, nT (GSE, GSM)', 'By, nT (GSM)', 'Bz, nT (GSM)', 'Vx Velocity, km/s, GSE',
         'Proton Density, n/cc', 'Temperature, K']]
df = df.replace(999.99, np.nan)
df = df.replace(9999.99, np.nan)
df = df.replace(99999.9, np.nan)
df = df.replace(9999999., np.nan)
for col in df.columns:
    mask = df[col].notna()
    a = mask.ne(mask.shift()).cumsum()
    df = df[(a.groupby(a).transform('size') < 15) | mask]
df = df.ffill()

## start time
epoch_time = datetime.datetime(1970,1,1)
## Time since 1970-01-01
delta = df.index - epoch_time
## universal time
ut = delta.total_seconds()
## I should rewrite gp.recalc in the future if I have time. Only dipole tilt angle is desired here. 
ps_list = []
for i in track(range(len(ut.values))):
    ps = gp.recalc(ut.values[i])
    ps_list.append(ps)
df["Dipole Tilt Angle"] = ps_list
lowres_data = np.genfromtxt(root_path+"/data/omni/low_res_1h/omni2_all_years.dat",
                     skip_header=0)
lowres_time = np.apply_along_axis(get_month_day, 1, lowres_data[:, 0:3])
F10_7 = pd.DataFrame(lowres_data[:, 50])
F10_7.columns = ["f10.7_index"]
F10_7.index = lowres_time
F10_7 = F10_7.replace(999.9, np.nan).ffill()
F10_7 = F10_7.resample('1T').ffill()
df_joint = df.join(F10_7)
# Python package pyspedas provides a easy way to download dst data.
## downloading data from https://wdc.kugi.kyoto-u.ac.jp/index.html
dst_vars = my_dst(trange=['2024-05-01', '2024-05-31'])
dst = pytplot.data_quants['kyoto_dst']
df_dst = pd.DataFrame(dst.to_numpy())
df_dst.columns = ["Dst"]
df_dst.index = dst.time.to_numpy()
df_dst = df_dst.resample('1T').ffill()
df_joint_dst = df_joint.join(df_dst)
## now let's fill all missing data with the nearest data prior to this point, so we avoid using data from future.
df_joint_dst = df_joint_dst.fillna(method='ffill').loc["2024-05-01":"2024-05-31"].dropna()

OMNI_moredst = pd.read_pickle(root_path+"/Input/Gannon_storm/OMNI_PhysicsInformedCols_May10.pkl")
OMNI = df_joint_dst
OMNI = pd.concat([OMNI_moredst['2024-05-09 20:00':'2024-05-10 07:59'].resample('1T').ffill(), OMNI], axis=0)
lagged_data = []
for lag in [i*5 for i in range(0,12)]:
    for covariate in OMNI.columns[0:len(OMNI.columns)-3]:  # 60-minute history, do not lag dipole tilt angle.
        lagged_data.append(OMNI[covariate].shift(lag).rolling('5T', min_periods=3).median().rename(f'{covariate}_lagmd_{1*lag}'))
Dst_history = True
## up to 12 hours of Dst.
if Dst_history:
    ## lagged dst
    lagged_dst = []
    covariate = 'Dst'
    for lag in [i*60 for i in range(1,13)]:
        lagged_dst.append(OMNI[covariate].shift(lag).rename(f'{covariate}_lag_{1*lag}'))
    OMNI = pd.concat([OMNI] + lagged_dst, axis=1)
OMNI = pd.concat([OMNI] + lagged_data, axis=1)
# ## first order difference of dst up to 12 hours.
covariate = 'Dst'
diff_dst = []
dst_minutes = [i*60 for i in range(1,13)]
for j in range(len(dst_minutes)):
    if j == 0:
        diff_dst.append((OMNI['Dst'] - OMNI[f'{covariate}_lag_{1*dst_minutes[j]}']).rename(f'{covariate}_diff_{j+1}'))
    else:
        diff_dst.append((OMNI[f'{covariate}_lag_{1*dst_minutes[j-1]}'] -
                        OMNI[f'{covariate}_lag_{1*dst_minutes[j]}']).rename(f'{covariate}_diff_{j+1}'))
OMNI = pd.concat([OMNI] + diff_dst, axis=1)
for lag in [i*60 for i in range(1,13)]:
    OMNI = OMNI.drop(f'{covariate}_lag_{1*lag}', axis=1)
OMNI = OMNI.dropna()
OMNI.to_pickle(root_path+"/data/Input/OMNI_May10_5m_feature_ACE.pkl")