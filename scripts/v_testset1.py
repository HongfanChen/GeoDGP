import pandas as pd
import datetime
import astropy
import astropy.units as u
from sunpy.coordinates import frames
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
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
    level_name = ['50 nT','200 nT', '300 nT', '400 nT', 
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
    level_name = ['50 nT','200 nT', '300 nT', '400 nT', 
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
    level_name = ['50 nT','200 nT', '300 nT', '400 nT', 
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
def get_table(HSS_results, region = 'Mid', multihour=True, WithDst=True):
    table = []
    if multihour:
        models = ['GeoDGP_Ahead', 'GeoDGP_persistence', 'Observation_persistence']
    else:
        if WithDst:
            models = ['Geospace', 'GeoDGP', 'Observation_persistence']
        else:
            models = ['GeoDGP']
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
station_info = pd.read_csv(root_path + "/data/Input/station_info.csv")
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


# ## Mid Latitude Stations (-50<MagLat < 50)
Num_station = len(np.unique(HSS_results[np.abs(HSS_results["MagLat"]) < 50]["Station"]))
print("The total number of Mid latitude stations included in this study is %d" % Num_station)
get_table(HSS_results, region = 'Mid', multihour=multihour_ahead, WithDst=WithDst)


# ## High Latitude Stations (MagLat > 50):


Num_station = len(np.unique(HSS_results[np.abs(HSS_results["MagLat"]) >= 50]["Station"]))
print("The total number of High latitude stations included in this study is %d" % Num_station)
get_table(HSS_results, region = 'High', multihour=multihour_ahead, WithDst=WithDst)


# ## Auroral Latitude Stations (MagLat  60-70)

Num_station = len(np.unique(HSS_results[(np.abs(HSS_results["MagLat"]) >= 60) & 
                                       (np.abs(HSS_results["MagLat"]) <= 70)]["Station"]))
print("The total number of auroral latitude stations included in this study is %d" % Num_station)
get_table(HSS_results, region = 'Auroral', multihour=multihour_ahead, WithDst=WithDst)


# ## All Latitude Stations (MagLat  -90 - 90)

Num_station = len(np.unique(HSS_results["Station"]))
print("The total number of stations of all latitude included in this study is %d" % Num_station)
get_table(HSS_results, region = 'All', multihour=multihour_ahead, WithDst=WithDst)


# ## Scatter plot of Heidke Skill Score over storms in 2015 (dBH only)
scatter_size = 60
label_size = 35
ticks_size = 25
shaded_color = "gray"  # Dark Green with transparency
shade_alpha=0.15
dashed_width = 4
line_style = (0, (5, 5))
line_alpha = 0.6
if (not multihour_ahead) & (QoIs == "dBH") & WithDst:
    plt.tight_layout()
    plt.figure(figsize=(30, 20))
    ## Level2: ----------------------
    plt.subplot(2,2,1)
    score = 'HSS_L1'
    HSS_dropna = HSS_results[['Method','Station', 'MagLat', score]].dropna()
    median = HSS_dropna.groupby(['Method', 'Station']).agg({score : 'median', 'MagLat' : 'median'})
    median.columns = ['Median', 'MagLat']
    Q1 = HSS_dropna.groupby(['Method', 'Station']).agg({score : lambda x: np.nanquantile(x, 0.25)})
    Q1.columns = ['Q1']
    Q3 = HSS_dropna.groupby(['Method', 'Station']).agg({score : lambda x: np.nanquantile(x, 0.75)})
    Q3.columns = ['Q3']
    IQR = median.join(Q1).join(Q3)
    plt.scatter(IQR.loc['GeoDGP']['MagLat'], IQR.loc['GeoDGP']['Median'],
                marker='o', s = scatter_size, color = 'red', label = 'GeoDGP')
    plt.scatter(IQR.loc['Geospace']['MagLat'], IQR.loc['Geospace']['Median'],
                marker='s', s = scatter_size, color = 'royalblue', label = 'Geospace')
    plt.fill_betweenx(y=[-0.05, 1], x1=60, x2=70, color=shaded_color, alpha=shade_alpha)
    plt.fill_betweenx(y=[-0.05, 1], x1=-60, x2=-70, color=shaded_color, alpha=shade_alpha)
    for x in [-50, 50]:
        plt.vlines(x=x, ymin=-0.05, ymax=1, color=shaded_color, linewidth=dashed_width, alpha=line_alpha, linestyle=line_style)
    plt.xlabel('Magenetic Latitude', fontsize = label_size)
    plt.ylabel('HSS (50 nT)', fontsize = label_size)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    ## Level2: ----------------------
    plt.subplot(2,2,2)
    score = 'HSS_L3'
    HSS_dropna = HSS_results[['Method','Station', 'MagLat', score]].dropna()
    median = HSS_dropna.groupby(['Method', 'Station']).agg({score : 'median', 'MagLat' : 'median'})
    median.columns = ['Median', 'MagLat']
    Q1 = HSS_dropna.groupby(['Method', 'Station']).agg({score : lambda x: np.nanquantile(x, 0.25)})
    Q1.columns = ['Q1']
    Q3 = HSS_dropna.groupby(['Method', 'Station']).agg({score : lambda x: np.nanquantile(x, 0.75)})
    Q3.columns = ['Q3']
    IQR = median.join(Q1).join(Q3)
    plt.scatter(IQR.loc['GeoDGP']['MagLat'], IQR.loc['GeoDGP']['Median'],
                marker='o', s = scatter_size, color = 'red', label = 'GeoDGP')
    plt.scatter(IQR.loc['Geospace']['MagLat'], IQR.loc['Geospace']['Median'],
                marker='s', s = scatter_size, color = 'royalblue', label = 'Geospace')
    plt.fill_betweenx(y=[-0.05, 1], x1=60, x2=70, color=shaded_color, alpha=shade_alpha)
    plt.fill_betweenx(y=[-0.05, 1], x1=-60, x2=-70, color=shaded_color, alpha=shade_alpha)
    for x in [-50, 50]:
        plt.vlines(x=x, ymin=-0.05, ymax=1, color=shaded_color, linewidth=dashed_width, alpha=line_alpha, linestyle=line_style)
    plt.xlabel('Magenetic Latitude', fontsize = label_size)
    plt.ylabel('HSS (200 nT)', fontsize = label_size)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    ## Level3: ----------------------
    plt.subplot(2,2,3)
    score = 'HSS_L4'
    HSS_dropna = HSS_results[['Method','Station', 'MagLat', score]].dropna()
    median = HSS_dropna.groupby(['Method', 'Station']).agg({score : 'median', 'MagLat' : 'median'})
    median.columns = ['Median', 'MagLat']
    Q1 = HSS_dropna.groupby(['Method', 'Station']).agg({score : lambda x: np.nanquantile(x, 0.25)})
    Q1.columns = ['Q1']
    Q3 = HSS_dropna.groupby(['Method', 'Station']).agg({score : lambda x: np.nanquantile(x, 0.75)})
    Q3.columns = ['Q3']
    IQR = median.join(Q1).join(Q3)
    plt.scatter(IQR.loc['GeoDGP']['MagLat'], IQR.loc['GeoDGP']['Median'],
                marker='o', s = scatter_size, color = 'red', label = 'GeoDGP')
    plt.scatter(IQR.loc['Geospace']['MagLat'], IQR.loc['Geospace']['Median'],
                marker='s', s = scatter_size, color = 'royalblue', label = 'Geospace')
    plt.fill_betweenx(y=[-0.05, 1], x1=60, x2=70, color=shaded_color, alpha=shade_alpha)
    plt.fill_betweenx(y=[-0.05, 1], x1=-60, x2=-70, color=shaded_color, alpha=shade_alpha)
    for x in [-50, 50]:
        plt.vlines(x=x, ymin=-0.05, ymax=1, color=shaded_color, linewidth=dashed_width, alpha=line_alpha, linestyle=line_style)
    plt.xlabel('Magenetic Latitude', fontsize = label_size)
    plt.ylabel('HSS (300 nT)', fontsize = label_size)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    ## Level4: ----------------------
    plt.subplot(2,2,4)
    score = 'HSS_L5'
    HSS_dropna = HSS_results[['Method','Station', 'MagLat', score]].dropna()
    median = HSS_dropna.groupby(['Method', 'Station']).agg({score : 'median', 'MagLat' : 'median'})
    median.columns = ['Median', 'MagLat']
    Q1 = HSS_dropna.groupby(['Method', 'Station']).agg({score : lambda x: np.nanquantile(x, 0.25)})
    Q1.columns = ['Q1']
    Q3 = HSS_dropna.groupby(['Method', 'Station']).agg({score : lambda x: np.nanquantile(x, 0.75)})
    Q3.columns = ['Q3']
    IQR = median.join(Q1).join(Q3)
    plt.scatter(IQR.loc['GeoDGP']['MagLat'], IQR.loc['GeoDGP']['Median'],
                marker='o', s = scatter_size, color = 'red', label = 'GeoDGP')
    plt.scatter(IQR.loc['Geospace']['MagLat'], IQR.loc['Geospace']['Median'],
                marker='s', s = scatter_size, color = 'royalblue', label = 'Geospace')
    plt.fill_betweenx(y=[-0.05, 1], x1=60, x2=70, color=shaded_color, alpha=shade_alpha)
    plt.fill_betweenx(y=[-0.05, 1], x1=-60, x2=-70, color=shaded_color, alpha=shade_alpha)
    for x in [-50, 50]:
        plt.vlines(x=x, ymin=-0.05, ymax=1, color=shaded_color, linewidth=dashed_width, alpha=line_alpha, linestyle=line_style)
#         plt.vlines(x=x, ymin=-0.05, ymax=1, color='black', linewidth=3)
    plt.xlabel('Magenetic Latitude', fontsize = label_size)
    plt.ylabel('HSS (400 nT)', fontsize = label_size)
    plt.xticks(fontsize = ticks_size)
    plt.yticks(fontsize = ticks_size)
    ## Legend and title: -----------------------
    params = {'legend.fontsize': 40,
          'legend.handlelength': 5}
    plt.rcParams.update(params)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc = 'lower center', ncol=2, labelspacing=1, markerscale=5, frameon=False)
#     plt.suptitle('Test set 1: dBH Heidke Skill Score With Four Thresholds', fontsize=50, y=0.93)
    # plt.legend(prop={'size': 15}, markerscale=4)
    if not multihour_ahead:
        plt.savefig(root_path+"/figure/TestSet1/TestSet1_HSS_MagLat.png",
                    bbox_inches='tight', dpi=300)


# # Distribution of HSS

Polar_scatter_size = 85
color_HighLat = 'black'

if (not multihour_ahead) & (QoIs == "dBH") & WithDst:
    plt.style.use('default')
    score_level = "HSS_L1"
    if score_level == "HSS_L1":
        print_score_level = "50 nT"
    elif score_level == "HSS_L2":
        print_score_level = "100 nT"
    elif score_level == "HSS_L3":
        print_score_level = "200 nT"
    elif score_level == "HSS_L4":
        print_score_level = "300 nT"
    elif score_level == "HSS_L5":
        print_score_level = "400 nT"
    HSS_station = HSS_results[HSS_results['Method'] == 'GeoDGP'][[score_level, 'GEOLAT', 'GEOLON']]
    Earth_loc = astropy.coordinates.EarthLocation(lat=HSS_station["GEOLAT"].values*u.deg,
                                         lon=HSS_station["GEOLON"].values*u.deg)
    station_lons = Earth_loc.lon.value  # Replace with your station longitudes
    station_lats = Earth_loc.lat.value  # Replace with your station latitudes
    center_lon=0
    # cmap = plt.get_cmap("RdYlBu_r", 12)
    cmap = plt.get_cmap("Reds", 10)
    fig = plt.figure(figsize=(18, 10))
    crs = ccrs.PlateCarree(central_longitude=center_lon)
    ax = plt.axes(projection=crs)
    ax.set_extent([-180, 180, 90, -90], ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='black', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'black', 'rotation':0}
    gl.ylabel_style = {'size': 15, 'color': 'black'}
    ax.coastlines(resolution='110m', color='black', linewidth=1)
    plt.draw()
    for ea in gl.left_label_artists+gl.right_label_artists:
        ea.set_visible(True)
    # plt.pcolormesh(geolon, geolat, z, transform=ccrs.PlateCarree(), cmap = cmap, vmin=0, vmax=1500)
    plt.scatter(
        x=station_lons,
        y=station_lats,
        c=HSS_station[score_level],
        cmap=cmap,  #this is the changes
        s=Polar_scatter_size,
        alpha=1,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=1,
        marker="o",
        edgecolors='black'
    )
# Define longitude range
    extent_lon_min = -180
    extent_lon_max = 179
    lon = np.linspace(extent_lon_min, extent_lon_max, 360)

    # Store bands' coordinates
    bands = {}

    for lat in [50, 60, 70, -50, -60, -70]:
        maglon, maglat = np.meshgrid(lon, lat)
        MAG_coord = frames.Geomagnetic(lon=maglon*u.deg, lat=maglat*u.deg, obstime=obs_time)
        target_coord = astropy.coordinates.ITRS(obstime=obs_time)
        geographic_coord = MAG_coord.transform_to(target_coord)

        pd_GEO = pd.DataFrame({'GEOLON': geographic_coord.spherical.lon.value.flatten(),
                               'GEOLAT': geographic_coord.spherical.lat.value.flatten()})
        pd_GEO.sort_values('GEOLON', inplace=True)

        # Store data for filling
        if lat in [60, 70, -60, -70]:  
            bands[lat] = pd_GEO

        # Plot latitude lines
        if lat in [50, -50]:  # High Latitude Lines
            plt.plot(pd_GEO['GEOLON'].values, pd_GEO['GEOLAT'].values, 
                     color=color_HighLat, linestyle='-', linewidth=2, 
                     transform=ccrs.PlateCarree())
    # Fill the area between 60° and 70° (and -60° to -70°)
    for lat in [60, -60]:  
        upper_band = bands[lat + np.sign(lat)*10]  # 70° or -70°
        lower_band = bands[lat]  # 60° or -60°

        # Ensure the longitude values align before filling
        plt.fill(np.concatenate([lower_band['GEOLON'].values, upper_band['GEOLON'].values[::-1]]), 
                 np.concatenate([lower_band['GEOLAT'].values, upper_band['GEOLAT'].values[::-1]]), 
                 color=shaded_color, alpha=shade_alpha, transform=ccrs.PlateCarree())
    ax.set_title("HSS (50 nT)",
                 fontsize=25)
    plt.text(-50, 35, r'MAG $50^{\circ}$N', color=color_HighLat, size = 18)
    plt.text(-30, -52, r'MAG $50^{\circ}$S', color=color_HighLat, size = 18)
    cbar = plt.colorbar(fraction=0.087, pad=0.05, orientation="horizontal", aspect=50)
    cbar.ax.tick_params(labelsize=17)
    if not multihour_ahead:
        plt.savefig(root_path+"/figure/TestSet1/TestSet1_HSS_%s_%s.png" % (print_score_level.split()[0], QoIs),
                    bbox_inches='tight', dpi=300)

if (not multihour_ahead) & (QoIs == "dBH") & WithDst:
    plt.style.use('default')
    score_level = "HSS_L3"
    if score_level == "HSS_L1":
        print_score_level = "50 nT"
    elif score_level == "HSS_L2":
        print_score_level = "100 nT"
    elif score_level == "HSS_L3":
        print_score_level = "200 nT"
    elif score_level == "HSS_L4":
        print_score_level = "300 nT"
    elif score_level == "HSS_L5":
        print_score_level = "400 nT"
    HSS_station = HSS_results[HSS_results['Method'] == 'GeoDGP'][[score_level, 'GEOLAT', 'GEOLON']]
    Earth_loc = astropy.coordinates.EarthLocation(lat=HSS_station["GEOLAT"].values*u.deg,
                                         lon=HSS_station["GEOLON"].values*u.deg)
    station_lons = Earth_loc.lon.value  # Replace with your station longitudes
    station_lats = Earth_loc.lat.value  # Replace with your station latitudes
    center_lon=0
    # cmap = plt.get_cmap("RdYlBu_r", 12)
    cmap = plt.get_cmap("Reds", 10)
    fig = plt.figure(figsize=(8, 10))
    crs = ccrs.NorthPolarStereo(central_longitude=center_lon)
    ax = plt.axes(projection=crs)
    ax.set_extent([-180, 180, 90, 35], ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1.5, color='black', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'black', 'rotation':0}
    gl.ylabel_style = {'size': 17, 'color': 'black'}
    ax.coastlines(resolution='110m', color='black', linewidth=1)
    plt.draw()
    for ea in gl.left_label_artists+gl.right_label_artists:
        ea.set_visible(True)
    latitude_label_lon = -25
    for i in range(13,len(gl.label_artists)):
        position = gl.label_artists[i].get_position()
        gl.label_artists[i].set_position((latitude_label_lon, position[1]))
    # plt.pcolormesh(geolon, geolat, z, transform=ccrs.PlateCarree(), cmap = cmap, vmin=0, vmax=1500)
    plt.scatter(
        x=station_lons,
        y=station_lats,
        c=HSS_station[score_level],
        cmap=cmap,  #this is the changes
        s=Polar_scatter_size,
        alpha=1,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=1,
        marker="o",
        edgecolors='black'
    )
    extent_lon_min = -180
    extent_lon_max = 179
    lon = np.linspace(extent_lon_min, extent_lon_max, 360)

    # Store bands' coordinates
    bands = {}

    for lat in [50, 60, 70]:
        maglon, maglat = np.meshgrid(lon, lat)
        MAG_coord = frames.Geomagnetic(lon=maglon*u.deg, lat=maglat*u.deg, obstime=obs_time)
        target_coord = astropy.coordinates.ITRS(obstime=obs_time)
        geographic_coord = MAG_coord.transform_to(target_coord)

        pd_GEO = pd.DataFrame({'GEOLON': geographic_coord.spherical.lon.value.flatten(),
                               'GEOLAT': geographic_coord.spherical.lat.value.flatten()})
        pd_GEO.sort_values('GEOLON', inplace=True)

        # Store data for filling
        if lat in [60, 70]:  
            bands[lat] = pd_GEO

        # Plot latitude lines
        if lat in [50]:  # High Latitude Lines
            plt.plot(pd_GEO['GEOLON'].values, pd_GEO['GEOLAT'].values, 
                     color=color_HighLat, linestyle='-', linewidth=2, 
                     transform=ccrs.PlateCarree())
    # Fill the area between 60° and 70° (and -60° to -70°)
    for lat in [60]:  
        upper_band = bands[lat + np.sign(lat)*10]  # 70° or -70°
        lower_band = bands[lat]  # 60° or -60°

        # Ensure the longitude values align before filling
        plt.fill(np.concatenate([lower_band['GEOLON'].values, upper_band['GEOLON'].values[::-1]]), 
                 np.concatenate([lower_band['GEOLAT'].values, upper_band['GEOLAT'].values[::-1]]), 
                 color=shaded_color, alpha=shade_alpha, transform=ccrs.PlateCarree())
    ax.set_title("HSS (200 nT)",
                 fontsize=25)
    plt.text(-148, 45, r'MAG $50^{\circ}$N', color=color_HighLat, size = 18, transform=ccrs.PlateCarree())
    ax.axis('off')
    if not multihour_ahead:
        plt.savefig(root_path+"/figure/TestSet1/TestSet1_HSS_%s_NorthPole_%s.png" % (print_score_level.split()[0], QoIs),
                    bbox_inches='tight', dpi=300)


# In[ ]:


if (not multihour_ahead) & (QoIs == "dBH") & WithDst:
    plt.style.use('default')
    score_level = "HSS_L3"
    if score_level == "HSS_L1":
        print_score_level = "50 nT"
    elif score_level == "HSS_L2":
        print_score_level = "100 nT"
    elif score_level == "HSS_L3":
        print_score_level = "200 nT"
    elif score_level == "HSS_L4":
        print_score_level = "300 nT"
    elif score_level == "HSS_L5":
        print_score_level = "400 nT"
    HSS_station = HSS_results[HSS_results['Method'] == 'GeoDGP'][[score_level, 'GEOLAT', 'GEOLON']]
    Earth_loc = astropy.coordinates.EarthLocation(lat=HSS_station["GEOLAT"].values*u.deg,
                                         lon=HSS_station["GEOLON"].values*u.deg)
    station_lons = Earth_loc.lon.value  # Replace with your station longitudes
    station_lats = Earth_loc.lat.value  # Replace with your station latitudes
    center_lon=0
    # cmap = plt.get_cmap("RdYlBu_r", 12)
    cmap = plt.get_cmap("Reds", 10)
    fig = plt.figure(figsize=(8, 10))
    crs = ccrs.SouthPolarStereo(central_longitude=center_lon)
    ax = plt.axes(projection=crs)
    ax.set_extent([-180, 180, -90, -35], ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1.5, color='black', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'black', 'rotation':0}
    gl.ylabel_style = {'size': 17, 'color': 'black'}
    ax.coastlines(resolution='110m', color='black', linewidth=1)
    plt.draw()
    for ea in gl.left_label_artists+gl.right_label_artists:
        ea.set_visible(True)
    latitude_label_lon = -155
    for i in range(13,len(gl.label_artists)):
        position = gl.label_artists[i].get_position()
        gl.label_artists[i].set_position((latitude_label_lon, position[1]))
    # plt.pcolormesh(geolon, geolat, z, transform=ccrs.PlateCarree(), cmap = cmap, vmin=0, vmax=1500)
    plt.scatter(
        x=station_lons,
        y=station_lats,
        c=HSS_station[score_level],
        cmap=cmap,  #this is the changes
        s=Polar_scatter_size,
        alpha=1,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=1,
        marker="o",
        edgecolors='black'
    )
    extent_lon_min = -180
    extent_lon_max = 179
    lon = np.linspace(extent_lon_min, extent_lon_max, 360)

    # Store bands' coordinates
    bands = {}

    for lat in [-50, -60, -70]:
        maglon, maglat = np.meshgrid(lon, lat)
        MAG_coord = frames.Geomagnetic(lon=maglon*u.deg, lat=maglat*u.deg, obstime=obs_time)
        target_coord = astropy.coordinates.ITRS(obstime=obs_time)
        geographic_coord = MAG_coord.transform_to(target_coord)

        pd_GEO = pd.DataFrame({'GEOLON': geographic_coord.spherical.lon.value.flatten(),
                               'GEOLAT': geographic_coord.spherical.lat.value.flatten()})
        pd_GEO.sort_values('GEOLON', inplace=True)

        # Store data for filling
        if lat in [-60, -70]:  
            bands[lat] = pd_GEO

        # Plot latitude lines
        if lat in [50, -50]:  # High Latitude Lines
            plt.plot(pd_GEO['GEOLON'].values, pd_GEO['GEOLAT'].values, 
                     color=color_HighLat, linestyle='-', linewidth=2, 
                     transform=ccrs.PlateCarree())
    # Fill the area between 60° and 70° (and -60° to -70°)
    for lat in [-60]:  
        upper_band = bands[lat + np.sign(lat)*10]  # 70° or -70°
        lower_band = bands[lat]  # 60° or -60°

        # Ensure the longitude values align before filling
        plt.fill(np.concatenate([lower_band['GEOLON'].values, upper_band['GEOLON'].values[::-1]]), 
                 np.concatenate([lower_band['GEOLAT'].values, upper_band['GEOLAT'].values[::-1]]), 
                 color=shaded_color, alpha=shade_alpha, transform=ccrs.PlateCarree())
    plt.text(20, -50, r'MAG $50^{\circ}$S', color=color_HighLat, size = 18, transform=ccrs.PlateCarree())
    ax.set_title("HSS (200 nT)", fontsize=25)
    ax.axis('off')
    ax.spines['left'].set_visible(False)
    if not multihour_ahead:
        plt.savefig(root_path+"/figure/TestSet1/TestSet1_HSS_%s_SouthPole_%s.png" % (print_score_level.split()[0], QoIs),
                    bbox_inches='tight', dpi=300)


