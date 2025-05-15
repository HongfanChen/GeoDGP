import pandas as pd
import netCDF4 as nc
import numpy as np
from sunpy.coordinates import frames
import astropy.units as u
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
file_path = root_path + '/data/Input/Gannon_storm/20241015-20-50-supermag.netcdf'
dataset = nc.Dataset(file_path)
## get the numebr of observations
N_obs = dataset.variables['id'].shape[0]
## get station ids
raw_id = dataset.variables['id'][0,:].reshape(-1)
single_letter = raw_id.data[[not mask for mask in raw_id.mask]]
str_letter = [str(x) for x in single_letter]
str_letter = [x[2] for x in str_letter]
N_gap = 3
station_id = [''.join(map(str, str_letter[i:i+N_gap])) for i in range(0, len(str_letter), N_gap)]
## get time
time_yr = dataset.variables['time_yr']
time_mo = dataset.variables['time_mo']
time_dy = dataset.variables['time_dy']
time_hr = dataset.variables['time_hr']
time_mt = dataset.variables['time_mt']
datetimes = pd.to_datetime(
    {
        'year': time_yr[:],
        'month': time_mo[:],
        'day': time_dy[:],
        'hour': time_hr[:],
        'minute': time_mt[:],
    }
)
all_stations_year = []
for i in range(N_obs):
    mlat = dataset.variables['mlat'][i, :].data
    mlon = dataset.variables['mlon'][i, :].data
    obs_time = datetimes[i]
    MAG_coord = frames.Geomagnetic(lon=mlon*u.deg, lat=mlat*u.deg, obstime=obs_time)
    target_coord = frames.SolarMagnetic(magnetic_model='igrf13', obstime=obs_time)
    SM_coord = MAG_coord.transform_to(target_coord)
    dBH = np.sqrt(dataset.variables['dbe_nez'][i, :].data **2 + dataset.variables['dbn_nez'][i, :].data**2)
    all_stations_Time_t = pd.DataFrame([pd.Series(station_id),
                                        pd.Series(dataset.variables['dbe_nez'][i, :].data),
                                        pd.Series(dataset.variables['dbn_nez'][i, :].data),
                                        pd.Series(dBH),
                                        pd.Series(SM_coord.lon.value),
                                        pd.Series(SM_coord.lat.value)]).T
    all_stations_Time_t['Time'] = obs_time
    all_stations_Time_t.columns = ["ID", "dBE", "dBN", "dBH", "SM_lon", "SM_lat", "Time"]
    all_stations_year.append(all_stations_Time_t)
all_stations_year = pd.concat(all_stations_year, axis=0)
all_stations_year = all_stations_year.reset_index(drop=True, inplace=False)
for station in station_id:
    output_dir = root_path+"/data/Input/Gannon_storm/AllStations_Gannon/"
    output_filename = "%s.pkl" % (station)
    output_path = output_dir+output_filename
    all_stations_year[all_stations_year["ID"] == station].drop("ID", axis=1).to_pickle(output_path)

