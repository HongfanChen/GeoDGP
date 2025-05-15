
def load_AllStations_YearbyYear(file_path_list, file_name_list, i, output_dir):
    import pandas as pd
    import netCDF4 as nc
    import numpy as np
    import pickle
    from sunpy.coordinates import frames
    import astropy.units as u
    import os
    import time
    # file_dir = "C:/Users/feixi/Desktop/deltaB/data/AllStations_OneYear/"
    # file_name = "all_stations_all2015.netcdf"
    file_path = file_path_list[i]
    filename = file_name_list[i]
    dataset = nc.Dataset(file_path)
    N_obs = dataset.variables['id'].shape[0]
    station_id = dataset.variables['id'][0,:].reshape(-1)
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
    ## vectorize the transformation from geomagnetic to solar magnetic
    all_stations_year = []
    for i in range(N_obs):
        mlat = dataset.variables['mlat'][i, :].data
        mlon = dataset.variables['mlon'][i, :].data
        obs_time = datetimes[i]
        MAG_coord = frames.Geomagnetic(lon=mlon*u.deg, lat=mlat*u.deg, obstime=obs_time)
        target_coord = frames.SolarMagnetic(magnetic_model='igrf13', obstime=obs_time)
        SM_coord = MAG_coord.transform_to(target_coord)
        dBH = np.sqrt(dataset.variables['dbe_nez'][i, :].data**2 + dataset.variables['dbn_nez'][i, :].data**2)
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
    ## now save file as each station
    for station in station_id:
    #     output_dir = "C:/Users/feixi/Desktop/deltaB/data/AllStations_OneYear_1min/"
        output_filename = "%s_%s.pkl" % (station, filename[16:20])
        output_path = output_dir+output_filename
        all_stations_year[all_stations_year["ID"] == station].drop("ID", axis=1).to_pickle(output_path)


