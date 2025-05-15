import pandas as pd
import datetime
import astropy
import astropy.units as u
from sunpy.coordinates import frames
import numpy as np
import re
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.download import *

def get_score_MidLat(HSS_results, method = "Deep GP"):
    score_level = ["HSS_L1", "HSS_L2", "HSS_L3", "MAE", "SignRate"]
    level_name = ['50 nT', '100 nT', '200 nT',
                  'MAE', 'SignRate']
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) < 50) &
                                       (HSS_results["Method"] == method)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df.index = level_name
    return score_df
def get_score_HighLat(HSS_results, method = "Deep GP"):
    score_level = ["HSS_L1", "HSS_L3", "HSS_L4", "HSS_L5", "MAE", "SignRate"]
    level_name = ['50 nT',  '200 nT', '300 nT', '400 nT',
                  'MAE', 'SignRate']
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) >= 50) &
                                       (HSS_results["Method"] == method)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df.index = level_name
    return score_df
def get_score_AuroralLat(HSS_results, method = "Deep GP"):
    score_level = ["HSS_L1", "HSS_L3", "HSS_L4", "HSS_L5", "MAE", "SignRate"]
    level_name = ['50 nT',  '200 nT', '300 nT', '400 nT',
                  'MAE', 'SignRate']
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) >= 60) & 
                                 (np.abs(HSS_results["MagLat"]) <= 70) &
                                 (HSS_results["Method"] == method)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df.index = level_name
    return score_df
def get_score_AllLat(HSS_results, method = "Deep GP"):
    score_level = ["HSS_L1", "HSS_L3", "HSS_L4", "HSS_L5", "MAE", "SignRate"]
    level_name = ['50 nT',  '200 nT', '300 nT', '400 nT',
                  'MAE', 'SignRate']
    score_df = []
    for score in score_level:
        score_data = HSS_results[(HSS_results["Method"] == method)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df.index = level_name
    return score_df
def get_table(HSS_results, region = 'Mid', multihour_ahead=True):
    table = []
    if multihour_ahead:
        models = ['GeoDGP', 'GeoDGP_persistence', 'Observation_persistence']
    else:
        models = ['Geospace', 'GeoDGP', 'Observation_persistence']
    for model in models:
        if region == 'Mid':
            model_score = get_score_MidLat(HSS_results, method = model)
        if region == 'High':
            model_score = get_score_HighLat(HSS_results, method = model)
        if region == 'Auroral':
            model_score = get_score_AuroralLat(HSS_results, method = model)
        if region == 'All':
            model_score = get_score_AllLat(HSS_results, method = model)
        table.append(model_score['Median'])
    table = pd.concat(table, axis=1).round(2)
    table.columns = models
    return table

## ----------------------------------------- 2015 test storms --------------------------------------------
QoIs = "dBH"
# path = "J:/deltaB/data/test_station/1min_With1minModel/%s/" % QoIs
multihour_ahead = False
path = root_path + "/data/test_station/TestSet3_IMF_ACE/10T/%s/" % QoIs
if multihour_ahead:
    path = root_path + "/data/test_station/TestSet3_IMF_ACE/10T_1h_ahead/%s/" % QoIs
# filename = "2015_03_17_withDst_F107.csv"
station_info = pd.read_csv(root_path + "/data/Input/station_info.csv")
station_location = station_info.iloc[:,0:3]
station_location.columns = ["Station", "GEOLON", "GEOLAT"]
file_list = sorted([x for x in os.listdir(path) if bool(re.match(r'^HSS_', x))])

Metrics = []
for filename in file_list:
    HSS = pd.read_csv(path + filename)
    Station_withScore = pd.merge(station_location, HSS, on='Station', how='inner')
    year = int(filename[4:8])
    month = int(filename[9:11])
    day = int(filename[12:14])
    obs_time = datetime.datetime(year, month, day)
    start_time = obs_time - datetime.timedelta(hours=6)
    end_time = obs_time + datetime.timedelta(hours=48)
    Earth_loc = astropy.coordinates.EarthLocation(lat=Station_withScore["GEOLAT"].values*u.deg,
                                     lon=Station_withScore["GEOLON"].values*u.deg)
    Geographic = astropy.coordinates.ITRS(x=Earth_loc.x, y=Earth_loc.y, z=Earth_loc.z, obstime=obs_time)
    target_coord = frames.Geomagnetic(magnetic_model='igrf13', obstime=obs_time)
    GeoMagnetic = Geographic.transform_to(target_coord)
    GeoMAG = pd.concat([pd.Series(GeoMagnetic.lon.value),
                        pd.Series(GeoMagnetic.lat.value)],
                       axis = 1)
    GeoMAG.columns = ["MagLon", "MagLat"]
    Station_GeoMAG = pd.concat([Station_withScore, GeoMAG], axis=1)
    Metrics.append(Station_GeoMAG)
Metrics = pd.concat(Metrics, axis=0).reset_index()


# # Statistics of Heidke Skill Score, 2024 May 10 Storm

# ## Mid Latitude Stations (Maglat<50)

Num_station = len(np.unique(Metrics[np.abs(Metrics["MagLat"]) < 50]["Station"]))
print("The total number of Mid latitude stations included in this study is %d" % Num_station)
get_table(Metrics, region='Mid', multihour_ahead=multihour_ahead)


# ## High Latitude Stations (MagLat > 50)

Num_station = len(np.unique(Metrics[np.abs(Metrics["MagLat"]) >= 50]["Station"]))
print("The total number of High latitude stations included in this study is %d" % Num_station)
get_table(Metrics, region = 'High', multihour_ahead=multihour_ahead)


# ## Auroral Latitude Stations (MagLat  60-70)


Num_station = len(np.unique(Metrics[(np.abs(Metrics["MagLat"]) >= 60) & 
                                       (np.abs(Metrics["MagLat"]) <= 70)]["Station"]))
print("The total number of auroral latitude stations included in this study is %d" % Num_station)
get_table(Metrics, region = 'Auroral', multihour_ahead=multihour_ahead)


# ## All Latitude Stations (MagLat  -90 - 90)

Num_station = len(np.unique(Metrics["Station"]))
print("The total number of stations of all latitude included in this study is %d" % Num_station)
get_table(Metrics, region = 'All', multihour_ahead=multihour_ahead)

