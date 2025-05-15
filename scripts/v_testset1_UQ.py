import pandas as pd
import datetime
import astropy
import astropy.units as u
from sunpy.coordinates import frames
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.download import *

def get_score_MidLat(HSS_results, quantity):
    score_level = [quantity]
    level_name = [quantity]
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) < 50)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    Num_station = len(np.unique(HSS_results[np.abs(HSS_results["MagLat"]) < 50]["Station"]))
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df['#Station'] = Num_station
    score_df['region'] = 'MidLat'
    score_df.index = level_name
    return score_df
def get_score_HighLat(HSS_results, quantity):
    score_level = [quantity]
    level_name = [quantity]
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) >= 50)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    Num_station = len(np.unique(HSS_results[np.abs(HSS_results["MagLat"]) >= 50]["Station"]))
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df['region'] = 'HighLat'
    score_df['#Station'] = Num_station
    score_df.index = level_name
    return score_df
def get_score_AuroralLat(HSS_results, quantity):
    score_level = [quantity]
    level_name = [quantity]
    score_df = []
    for score in score_level:
        score_data = HSS_results[(np.abs(HSS_results["MagLat"]) >= 60) & 
                                 (np.abs(HSS_results["MagLat"]) <= 70)][score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    Num_station = len(np.unique(HSS_results[(np.abs(HSS_results["MagLat"]) >= 60) & 
                                       (np.abs(HSS_results["MagLat"]) <= 70)]["Station"]))
    score_df = pd.concat(score_df, axis=0)
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df['region'] = 'AuroralLat'
    score_df['#Station'] = Num_station
    score_df.index = level_name
    return score_df
def get_score_AllLat(HSS_results, quantity):
    score_level = [quantity]
    level_name = [quantity]
    score_df = []
    for score in score_level:
        score_data = HSS_results[score].dropna()
        HSS_median = np.nanmedian(score_data)
        HSS_Q1 = np.nanquantile(score_data, 0.25)
        HSS_Q3 = np.nanquantile(score_data, 0.75)
        score_pd = pd.DataFrame([HSS_Q1, HSS_median, HSS_Q3]).T
        score_df.append(score_pd)
    score_df = pd.concat(score_df, axis=0)
    Num_station = len(np.unique(HSS_results["Station"]))
    score_df.columns = ["Q1", "Median", "Q3"]
    score_df['region'] = 'AllLat'
    score_df['#Station'] = Num_station
    score_df.index = level_name
    return score_df
def get_table(HSS_results, quantity, region):
    table = []
    if region == 'Mid':
        model_score = get_score_MidLat(HSS_results, quantity)
    if region == 'High':
        model_score = get_score_HighLat(HSS_results, quantity)
    if region == 'Auroral':
        model_score = get_score_AuroralLat(HSS_results, quantity)
    if region == 'All':
        model_score = get_score_AllLat(HSS_results, quantity)
    return model_score


def clean_quantity(quantity_file, station_file):
    station_info = pd.read_csv(station_file)
    station_location = station_info.iloc[:,0:3]
    station_location.columns = ["Station", "GEOLON", "GEOLAT"]
    quantity_df = pd.read_csv(quantity_file)
    quantity_df = pd.merge(station_location, quantity_df, on='Station', how='inner')
    obs_time = datetime.datetime(2015, 6, 1)
    Earth_loc = astropy.coordinates.EarthLocation(lat=quantity_df["GEOLAT"].values*u.deg,
                                         lon=quantity_df["GEOLON"].values*u.deg)
    Geographic = astropy.coordinates.ITRS(x=Earth_loc.x, y=Earth_loc.y, z=Earth_loc.z, obstime=obs_time)
    target_coord = frames.Geomagnetic(magnetic_model='igrf13', obstime=obs_time)
    GeoMagnetic = Geographic.transform_to(target_coord)
    GeoMAG = pd.concat([pd.Series(GeoMagnetic.lon.value),
                        pd.Series(GeoMagnetic.lat.value)],
                       axis = 1)
    GeoMAG.columns = ["MagLon", "MagLat"]
    quantity_df = pd.concat([quantity_df, GeoMAG], axis=1)
    return quantity_df
def make_table(data, quantity):
    df = pd.concat([get_table(data, quantity, region = 'Mid'),
           get_table(data, quantity, region = 'High'),
           get_table(data, quantity, region = 'Auroral'),
           get_table(data, quantity, region = 'All')],
          axis=0)
    return df
## ----------------------------------------- 2015 test storms --------------------------------------------
QoIs = "dBN"
multihour_ahead = False
hour = 1
WithDst=True
if not WithDst:
    multihour_ahead = False
    QoIs = 'dBH'
if multihour_ahead:
    path = root_path + "/data/test_station/TestSet1/10T_%sh_ahead/%s/" % (hour, QoIs)
elif WithDst:
    path = root_path + "/data/test_station/TestSet1/10T/%s/" % QoIs
else:
    path = root_path + "/data/test_station/TestSet1/NoDst/10T/%s/" % QoIs
station_file = root_path + "/data/Input/station_info.csv"
coverage_file = path + 'coverage.csv'
score_file = path + 'interval_score.csv'
width_file = path+'interval_width.csv'


coverage = clean_quantity(coverage_file, station_file)
interval_score = clean_quantity(score_file, station_file)
interval_width = clean_quantity(width_file, station_file)
coverage_tab = make_table(coverage, quantity = 'coverage')
scoretab = make_table(interval_score, quantity = 'interval_score')
widthtab = make_table(interval_width, quantity = 'interval_width')

coverage_tab.round(2)

widthtab.round(0)

scoretab.round(0)


if QoIs == 'dBH':
    if not multihour_ahead:
        scatter_size = 60
        label_size = 38
        ticks_size = 28
        shaded_color = "gray"  # Dark Green with transparency
        shade_alpha=0.15
        dashed_width = 3
        line_style = (0, (5, 5))
        line_alpha = 0.6
        plt.figure(figsize=(38, 10))
        params = {'legend.fontsize': 28,
              'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.subplot(1,3,1)
        plt.scatter(coverage['MagLat'], coverage['coverage'],
                        marker='o', s = scatter_size, color = 'black')
        plt.fill_betweenx(y=[0.7, 1], x1=60, x2=70, color=shaded_color, alpha=shade_alpha)
        plt.fill_betweenx(y=[0.7, 1], x1=-60, x2=-70, color=shaded_color, alpha=shade_alpha)
        for x in [-50, 50]:
            plt.vlines(x=x, ymin=0.7, ymax=1, color=shaded_color, linewidth=dashed_width, alpha=line_alpha, linestyle=line_style)
        plt.hlines(y=0.98, xmin=-85, xmax=85, color = 'black', linewidth=dashed_width,
                   linestyle = '-', label = 'Empirical coverage (98%)',
                   alpha = 1)
        plt.hlines(y=0.95, xmin=-85, xmax=85, color = 'red', linewidth=dashed_width,
                   linestyle = '-', alpha = 0.5, label = 'Nominal rate (95%)')
        plt.xlabel('Magenetic Latitude', fontsize = label_size)
        plt.ylabel('d$B_H$ Coverage Rate', fontsize = label_size)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
        plt.legend()
        plt.subplot(1,3,2)
        plt.scatter(interval_width['MagLat'], interval_width['interval_width'],
                        marker='o', s = scatter_size, color = 'black')
        plt.fill_betweenx(y=[110, 200], x1=60, x2=70, color=shaded_color, alpha=shade_alpha)
        plt.fill_betweenx(y=[110, 200], x1=-60, x2=-70, color=shaded_color, alpha=shade_alpha)
        for x in [-50, 50]:
            plt.vlines(x=x, ymin=110, ymax=200, color=shaded_color, linewidth=dashed_width, alpha=line_alpha, linestyle=line_style)
        plt.xlabel('Magenetic Latitude', fontsize = label_size)
        plt.ylabel('d$B_H$ Average Interval Width', fontsize = label_size)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.subplot(1,3,3)
        plt.scatter(interval_score['MagLat'], interval_score['interval_score'],
                        marker='o', s = scatter_size, color = 'black')
        plt.fill_betweenx(y=[0, 1300], x1=60, x2=70, color=shaded_color, alpha=shade_alpha)
        plt.fill_betweenx(y=[0, 1300], x1=-60, x2=-70, color=shaded_color, alpha=shade_alpha)
        for x in [-50, 50]:
            plt.vlines(x=x, ymin=0, ymax=1300, color=shaded_color, linewidth=dashed_width, alpha=line_alpha, linestyle=line_style)
        plt.xlabel('Magenetic Latitude', fontsize = label_size)
        plt.ylabel('d$B_H$ Average Interval Score', fontsize = label_size)
        plt.xticks(fontsize = ticks_size)
        plt.yticks(fontsize = ticks_size)
        plt.savefig(root_path+"/figure/TestSet1/TestSet1_UQ.png",
                            bbox_inches='tight', dpi=300)

