
from rich.progress import track
import os
import pandas as pd
import datetime
import pytplot
import geopack.geopack as gp
from datetime import datetime
import numpy as np
import datetime
import requests
from matplotlib import pyplot as plt
import random
from urllib.parse import urljoin
from scipy.signal import argrelextrema
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.download import *
# %%
# # Step 1: Download Data
base_url = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/modified/monthly_1min/"

save_dir = root_path+"/data/omni/high_res_1min/"

# Check and create directory if not exists
os.makedirs(save_dir, exist_ok=True)

# Download everything from 1995 to 2022
for year in range(1995, 2023):
    for month in range(1, 13):
        # Construct the filename and its URL
        filename = f"omni_min{year}{str(month).zfill(2)}.asc"
        file_url = urljoin(base_url, filename)

        # Attempt to download the file
        response = requests.get(file_url, stream=True)

        # If the download is successful, save the file
        if response.status_code == 200:
            with open(os.path.join(save_dir, filename), 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Failed to download {filename}.")

print("Download complete.")


# # Step 2: OMNI features
# Define the directory where downloaded files are saved
save_dir = root_path+"/data/omni/high_res_1min/"

# Define the time range
start_year, start_month = 1995, 1
end_year, end_month = 2022, 12

# Prepare an empty DataFrame to hold the data
df = pd.DataFrame()

# Loop through each year and month
for year in track(range(start_year, end_year+1)):
    for month in range(1, 13):
        if year == start_year and month < start_month:
            continue
        if year == end_year and month > end_month:
            break
        # Construct the filename
        filename = f"omni_min{year}{str(month).zfill(2)}.asc"
        file_path = os.path.join(save_dir, filename)

        # Check if the file exists
        if os.path.isfile(file_path):
            # Read the file into a DataFrame
            df_temp = np.loadtxt(file_path)  # Add delimiter or other arguments if needed
            # Append the data to the main DataFrame
#           df = df.append(pd.DataFrame(df_temp))
            df = pd.concat([df, pd.DataFrame(df_temp)], ignore_index=True)
        else:
            print(f"{filename} not found.")
# Apply the function to each row in the DataFrame
df['Datetime'] = df.apply(convert_to_datetime, axis=1)

# Set the new datetime column as the index
df.set_index('Datetime', inplace=True)
# Rename columns
df.drop(columns=df.columns[:4], axis=1, inplace=True)
df.columns = ['ID for IMF spacecraft', 'ID for SW Plasma spacecraft', '# of points in IMF averages', 
              '# of points in Plasma averages', 'Percent interp', 'Timeshift, sec', 'RMS, Timeshift', 
              'RMS, Phase front normal', 'Time btwn observations, sec', 'Field magnitude average', 
              'Bx, nT (GSE, GSM)', 'By, nT (GSE)', 'Bz, nT (GSE)', 'By, nT (GSM)', 'Bz, nT (GSM)', 'RMS SD B scalar',
              'RMS SD field vector', 'Flow speed, km/s', 'Vx Velocity, km/s, GSE', 'Vy Velocity, km/s, GSE', 
              'Vz Velocity, km/s, GSE', 'Proton Density, n/cc', 'Temperature, K', 'Flow pressure, nPa', 
              'Electric field, mV/m', 'Plasma beta', 'Alfven mach number', 'X(s/c), GSE, Re', 'Y(s/c), GSE, Re', 
              'Z(s/c), GSE, Re', 'BSN location, Xgse, Re', 'BSN location, Ygse, Re', 'BSN location, Zgse, Re',
              'AE-index, nT', 'AL-index, nT', 'AU-index, nT', 'SYM/D index, nT', 'SYM/H index, nT', 
              'ASY/D index, nT', 'ASY/H index, nT', 'Na/Np Ratio', 'Magnetosonic mach number']
# df.to_pickle(path_prefix+"/Paper_deltaB/data/omni/omni_preprocessing/OMNI_step2.pkl")
print("Data loading complete.")


# # Step 3: Dipole Tilt Angle
# df = pd.read_pickle(path_prefix+"/Paper_deltaB/data/omni/omni_preprocessing/OMNI_step2.pkl")
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


# In[ ]:


## I should rewrite gp.recalc in the future if I have time. Only dipole tilt angle is desired here. 
ps_list = []
for i in track(range(len(ut.values))):
    ps = gp.recalc(ut.values[i])
    ps_list.append(ps)
df["Dipole Tilt Angle"] = ps_list

# # Step 4: F10.7

# data available at https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/
# https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat
lowres_data = np.genfromtxt(root_path+"/data/omni/low_res_1h/omni2_all_years.dat", skip_header=0)
# In[ ]:


lowres_time = np.apply_along_axis(get_month_day, 1, lowres_data[:, 0:3])
F10_7 = pd.DataFrame(lowres_data[:, 50])
F10_7.columns = ["f10.7_index"]
F10_7.index = lowres_time
F10_7 = F10_7.replace(999.9, np.nan).ffill()
F10_7 = F10_7.resample('1T').ffill()
df_joint = df.join(F10_7)


# ## Step 5: Dst
# Python package pyspedas provides a easy way to download dst data.
## downloading data from https://wdc.kugi.kyoto-u.ac.jp/index.html
dst_vars = my_dst(trange=['1995-01-01', '2023-01-02'])
dst = pytplot.data_quants['kyoto_dst']
df_dst = pd.DataFrame(dst.to_numpy())
df_dst.columns = ["Dst"]
df_dst.index = dst.time.to_numpy()
df_dst = df_dst.resample('1T').ffill()
df_joint_dst = df_joint.join(df_dst)
## now let's fill all missing data with the nearest data prior to this point, so we avoid using data from future.
df_joint_dst = df_joint_dst.fillna(method='ffill').loc["1995-01-01":"2023"].dropna()

dst = pytplot.data_quants['kyoto_dst']
df_dst = pd.DataFrame(dst.to_numpy())
df_dst.columns = ["Dst"]
df_dst.index = dst.time.to_numpy()
df_dst.to_pickle(root_path+"/data/Input/Dst_1995_2022.pkl")


# # Step 6: 5 minute median data
OMNI = df_joint_dst
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
OMNI.to_pickle(root_path+"/data/Input/OMNI_1995_2022_5m_feature.pkl")


# # Step 7: Storm Only Data
## We find the local minima using a 5-hour window.
df_dst = pd.read_pickle(root_path+"/data/Input/Dst_1995_2022.pkl")
Dst_localMin = df_dst.iloc[argrelextrema(df_dst.Dst.values,
                                         np.less_equal,
                                         order=36)[0]]
threshold = 50
validation_threshold = 100
Storm_date = Dst_localMin[Dst_localMin['Dst'] < -validation_threshold].index
random.seed(20240811)
random_index = random.sample(range(1, len(Storm_date)), int(len(Storm_date)*0.09))
validation_date = [Storm_date[i].to_pydatetime() for i in random_index]

val_storm_date_list = [(storm_date - datetime.timedelta(hours=60),
                         storm_date + datetime.timedelta(hours=60)) for storm_date in validation_date]
val_storm = Dst_localMin[Dst_localMin['Dst'] < -100].reset_index(drop=True, inplace=False).iloc[random_index, :]
val_storm.index = validation_date

Storm_date = Dst_localMin[Dst_localMin['Dst'] < -threshold].index
Storm_data = []
for i in range(len(Storm_date)):
    if i < len(Storm_date)-1:
        if Storm_date[i+1] - Storm_date[i] > datetime.timedelta(hours=120):
            start_date = Storm_date[i] - datetime.timedelta(hours=60)
            end_date = Storm_date[i] + datetime.timedelta(hours=60)
            Storm_data.append(OMNI.loc[start_date:end_date])
        else:
            start_date = Storm_date[i] - datetime.timedelta(hours=60)
            end_date = Storm_date[i+1] - datetime.timedelta(hours=60) - datetime.timedelta(minutes=1)
            Storm_data.append(OMNI.loc[start_date:end_date])
    else:
        start_date = Storm_date[i] - datetime.timedelta(hours=60)
        end_date = Storm_date[i] + datetime.timedelta(hours=60)
        Storm_data.append(OMNI.loc[start_date:end_date])
Storm_data = pd.concat(Storm_data, axis=0)

Storm_data.to_pickle(root_path+"/data/Input/OMNI_1995_2022_5m_feature_Storms_%snT.pkl" % threshold)

