# %%
import pandas as pd
import datetime
import astropy
import astropy.units as u
from sunpy.coordinates import frames
import numpy as np
import pandas as pd
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.download import *

def get_score_MidLat(HSS_results, method = "Deep GP", hemisphere = 'North'):
    score_level = ["HSS_L1", "HSS_L2", "HSS_L3", "MAE"]
    level_name = ['50 nT', '100 nT', '200 nT', 
                  'MAE']
    score_df = []
    for score in score_level:
        if hemisphere == 'North':
            score_data = HSS_results[(HSS_results["MagLat"] < 50) &
                                     (HSS_results["MagLat"] >= 0) &
                                     (HSS_results["Method"] == method)][score].dropna()
        else:
            score_data = HSS_results[(HSS_results["MagLat"] < 0) &
                                     (HSS_results["MagLat"] >= -50) &
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
def get_score_HighLat(HSS_results, method = "Deep GP", hemisphere = 'North'):
    score_level = ["HSS_L1", "HSS_L3", "HSS_L4", "HSS_L5", "MAE"]
    level_name = ['50 nT','200 nT', '300 nT', '400 nT', 
                  'MAE']
    score_df = []
    for score in score_level:
        if hemisphere == 'North':
            score_data = HSS_results[(HSS_results["MagLat"] >= 50) &
                                           (HSS_results["Method"] == method)][score].dropna()
        else:
            score_data = HSS_results[(HSS_results["MagLat"] <= -50) &
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
def get_score_AuroralLat(HSS_results, method = "Deep GP", hemisphere = 'North'):
    score_level = ["HSS_L1", "HSS_L3", "HSS_L4", "HSS_L5", "MAE"]
    level_name = ['50 nT','200 nT', '300 nT', '400 nT', 
                  'MAE']
    score_df = []
    for score in score_level:
        if hemisphere == 'North':
            score_data = HSS_results[(HSS_results["MagLat"] >= 60) & 
                                     (HSS_results["MagLat"] <= 70) &
                                     (HSS_results["Method"] == method)][score].dropna()
        else:
            score_data = HSS_results[(HSS_results["MagLat"] <= -60) & 
                                     (HSS_results["MagLat"] >= -70) &
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
def get_score_AllLat(HSS_results, method = "Deep GP", hemisphere = 'North'):
    score_level = ["HSS_L1", "HSS_L3", "HSS_L4", "HSS_L5", "MAE"]
    level_name = ['50 nT','200 nT', '300 nT', '400 nT', 
                  'MAE']
    score_df = []
    for score in score_level:
        if hemisphere == 'North':
            score_data = HSS_results[(HSS_results["MagLat"] >= 0) &
                                     (HSS_results["Method"] == method)][score].dropna()
        else:
            score_data = HSS_results[(HSS_results["MagLat"] <= 0) &
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
def get_table(HSS_results, region = 'Mid'):
    table = []
    for model in ['Geospace', 'GeoDGP']:
        for hemis in ['North', 'South']:
            if region == 'Mid':
                model_score = get_score_MidLat(HSS_results, method = model, hemisphere = hemis)
            if region == 'High':
                model_score = get_score_HighLat(HSS_results, method = model, hemisphere = hemis)
            if region == 'Auroral':
                model_score = get_score_AuroralLat(HSS_results, method = model, hemisphere = hemis)
            if region == 'All':
                model_score = get_score_AllLat(HSS_results, method = model, hemisphere = hemis)
            table.append(model_score['Median'])
    table = pd.concat(table, axis=1).round(2)
    table.columns = ['Geospace N', 'Geospace S', 'GeoDGP N', 'GeoDGP S']
    return table


# In[3]:


## ----------------------------------------- 2015 test storms --------------------------------------------
QoIs = "dBH"
path = root_path + "/data/test_station/TestSet1/10T/%s/" % QoIs
station_info = pd.read_csv(root_path+"/data/Input/station_info.csv")
station_location = station_info.iloc[:,0:3]
station_location.columns = ["Station", "GEOLON", "GEOLAT"]
HSS_results = pd.read_csv(path + 'HSS.csv')
HSS_results = pd.merge(station_location, HSS_results, on='Station', how='inner')
obs_time = datetime.datetime(2015, 6, 1)
Earth_loc = astropy.coordinates.EarthLocation(lat=HSS_results["GEOLAT"].values*u.deg,
                                     lon=HSS_results["GEOLON"].values*u.deg)
Geographic = astropy.coordinates.ITRS(x=Earth_loc.x, y=Earth_loc.y, z=Earth_loc.z, obstime=obs_time)
target_coord = frames.Geomagnetic(magnetic_model='igrf13', obstime=obs_time)
GeoMagnetic = Geographic.transform_to(target_coord)
GeoMAG = pd.concat([pd.Series(GeoMagnetic.lon.value),
                    pd.Series(GeoMagnetic.lat.value)],
                   axis = 1)
GeoMAG.columns = ["MagLon", "MagLat"]
HSS_results = pd.concat([HSS_results, GeoMAG], axis=1)
Num_N = len(np.unique(HSS_results[(HSS_results["MagLat"] < 50) & (HSS_results["MagLat"] >= 0)]["Station"]))
Num_S = len(np.unique(HSS_results[(HSS_results["MagLat"] < 0) & (HSS_results["MagLat"] >= -50)]["Station"]))
print("# Mid latitude stations North: %d; South: %d" % (Num_N, Num_S))
get_table(HSS_results, region = 'Mid')
Num_N = len(np.unique(HSS_results[HSS_results["MagLat"] >= 50]["Station"]))
Num_S = len(np.unique(HSS_results[HSS_results["MagLat"] <= -50]["Station"]))
print("# High latitude stations North: %d; South: %d" % (Num_N, Num_S))
get_table(HSS_results, region = 'High')
Num_N = len(np.unique(HSS_results[(HSS_results["MagLat"] >= 60) & 
                                       (HSS_results["MagLat"] <= 70)]["Station"]))
Num_S = len(np.unique(HSS_results[(HSS_results["MagLat"] >= -70) & 
                                       (HSS_results["MagLat"] <= -60)]["Station"]))
print("# Auroral latitude stations North: %d; South: %d" % (Num_N, Num_S))
get_table(HSS_results, region = 'Auroral')
Num_N = len(np.unique(HSS_results[(HSS_results["MagLat"] >= 0)]['Station']))
Num_S = len(np.unique(HSS_results[(HSS_results["MagLat"] < 0)]['Station']))
print("# All latitude stations North: %d; South: %d" % (Num_N, Num_S))
get_table(HSS_results, region = 'All')


# %%
