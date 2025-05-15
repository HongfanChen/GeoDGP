import pandas as pd
import datetime
import astropy
import astropy.units as u
from sunpy.coordinates import frames
import numpy as np
import os
import re
import pytplot
import re

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.download import *

def get_score_MidLat(HSS_results, method = "Deep GP"):
    score_level = ["HSS_L1", "HSS_L2", "HSS_L3", "MAE", "SignRate"]
    level_name = ['50 nT', '100 nT', '200 nT', 'MAE', 'SignRate']
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
    level_name = ['50 nT', '200 nT', '300 nT', '400 nT', 
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
    level_name = ['50 nT', '200 nT', '300 nT', '400 nT', 
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
    level_name = ['50 nT', '200 nT', '300 nT', '400 nT', 
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
def get_table(HSS_results, region = 'Mid'):
    table = []
    models = ['Dagger', 'GeoDGP_ahead', 'GeoDGP_persistence', 'Observation_persistence']
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

## download dst data
dst_year2015 = my_dst(trange=['2015-03-14', '2015-03-18'])
dst = pytplot.data_quants['kyoto_dst']
df_dst = pd.DataFrame(dst.to_numpy())
print("minimum Dst of 2015-03-17 storm is %s nT" % df_dst.min().item())
## download dst data
dst_year2015 = my_dst(trange=['2011-08-03', '2011-08-07'])
dst = pytplot.data_quants['kyoto_dst']
df_dst = pd.DataFrame(dst.to_numpy())
print("minimum Dst of 2011-08-05 storm is %s nT" % df_dst.min().item())

## ----------------------------------------- 2015 test storms --------------------------------------------
QoIs = "dBE"
# path = "J:/deltaB/data/test_station/1min_With1minModel/%s/" % QoIs
path = root_path + "/data/test_station/TestSet2/10T/%s/" % QoIs
# filename = "2015_03_17_withDst_F107.csv"
station_info = pd.read_csv(root_path + "/data/Input/station_info.csv")
station_location = station_info.iloc[:,0:3]
station_location.columns = ["Station", "GEOLON", "GEOLAT"]
file_list = sorted([x for x in os.listdir(path) if bool(re.match(r'^ModelComp_HSS_', x))])

HSS = []
for i in range(2):
    HSS.append(pd.read_csv(path + file_list[i]))
HSS = pd.concat(HSS, axis=0)
HSS.reset_index(drop=True, inplace=True)
HSS_summary = HSS.groupby(['Station', 'Method']).sum()
num_thresholds = 5
for i in range(num_thresholds):
    a = 0+4*i
    b = 1+4*i
    c = 2+4*i
    d = 3+4*i
    HSS_summary['HSS_L%s'%(i+1)] = HSS_summary.apply(lambda x: 2 *(x.iloc[a]*x.iloc[d] - x.iloc[b]*x.iloc[c]) / 
                                                 ((x.iloc[a]+x.iloc[b])*(x.iloc[b]+x.iloc[d]) + 
                                                 (x.iloc[a]+x.iloc[c])*(x.iloc[c]+x.iloc[d])), axis=1)
HSS_summary['SignRate'] = HSS_summary.apply(lambda x: x.iloc[num_thresholds*4 +2]/x.iloc[num_thresholds*4 +1], axis=1)
HSS['AE'] = HSS.apply(lambda x: x.iloc[num_thresholds*4] * x.iloc[num_thresholds*4+1], axis=1)
MAE = HSS[['Station', 'Method', 'AE', 'N']].groupby(['Station', 'Method']).sum().apply(lambda x: 
                                                                                       round(x.iloc[0]/x.iloc[1]),axis=1)
MAE = pd.DataFrame(MAE)
MAE.columns = ['MAE']
Metrics = pd.merge(HSS_summary[['HSS_L1', 'HSS_L2', 'HSS_L3', 'HSS_L4', 'HSS_L5', 'SignRate']], MAE,
                   left_index=True, right_index=True)
Metrics = pd.merge(Metrics.reset_index(drop=False, inplace=False), station_location)
obs_time = datetime.datetime(2013, 1, 1)
Earth_loc = astropy.coordinates.EarthLocation(lat=Metrics["GEOLAT"].values*u.deg,
                                     lon=Metrics["GEOLON"].values*u.deg)
Geographic = astropy.coordinates.ITRS(x=Earth_loc.x, y=Earth_loc.y, z=Earth_loc.z, obstime=obs_time)
target_coord = frames.Geomagnetic(magnetic_model='igrf13', obstime=obs_time)
GeoMagnetic = Geographic.transform_to(target_coord)
GeoMAG = pd.concat([pd.Series(GeoMagnetic.lon.value),
                    pd.Series(GeoMagnetic.lat.value)],
                   axis = 1)
GeoMAG.columns = ["MagLon", "MagLat"]
Metrics = pd.concat([Metrics, GeoMAG], axis=1)


# # Statistics of Heidke Skill Score

# ## Mid Latitude Stations (Maglat<50)

Num_station = len(np.unique(Metrics[np.abs(Metrics["MagLat"]) < 50]["Station"]))
print("The total number of Mid latitude stations included in this study is %d" % Num_station)
get_table(Metrics, region = 'Mid')


# ## High Latitude Stations (MagLat > 50)

Num_station = len(np.unique(Metrics[np.abs(Metrics["MagLat"]) >= 50]["Station"]))
print("The total number of High latitude stations included in this study is %d" % Num_station)
get_table(Metrics, region = 'High')


# ## Auroral Latitude Stations (MagLat  60-70)
Num_station = len(np.unique(Metrics[(np.abs(Metrics["MagLat"]) >= 60) & 
                                       (np.abs(Metrics["MagLat"]) <= 70)]["Station"]))
print("The total number of auroral latitude stations included in this study is %d" % Num_station)
get_table(Metrics, region = 'Auroral')


# ## All Latitude Stations (MagLat  -90 - 90)

Num_station = len(np.unique(Metrics["Station"]))
print("The total number of stations of all latitude included in this study is %d" % Num_station)
get_table(Metrics, region = 'All')

