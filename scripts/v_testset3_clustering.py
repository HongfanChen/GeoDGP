import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from rich.progress import track
import datetime
import torch.nn as nn
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import astropy
import astropy.units as u
from sunpy.coordinates import frames
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)


def test_general(file_path, QoIs,
                 storm_date = datetime.datetime(2015, 3, 17, 4, 7),
                 lower = 72,
                 upper = 72):
    station_data = pd.read_pickle(file_path)
    station_data = station_data.dropna()
    station_data.index = station_data['Time']
    station_data = station_data[[QoIs, "SM_lon", "SM_lat"]]
    start_date = storm_date - datetime.timedelta(hours=lower)
    end_date = storm_date + datetime.timedelta(hours=upper)
    test_SuperMAG_raw = station_data.loc[start_date:end_date]
    test_Y_raw = test_SuperMAG_raw[QoIs]
    return test_Y_raw

path = root_path+"/data/train/"
QoIs = "dBH"
station_path = path + '../Input/Gannon_storm/AllStations_Gannon/'
station_file = os.listdir(station_path)
May10_storm = [datetime.datetime(2024, 5, 11, 3, 0)]
upper = 36
lower = 18
station_obs = []
for i in range(len(station_file)):
    file_path = station_path + station_file[i]
    storm_Y = test_general(file_path, QoIs,
                                    storm_date = May10_storm[0],
                                    lower = lower,
                                    upper = upper)
    time_idx = storm_Y.index
    if not storm_Y.shape[0] == 0:
        station_obs.append(pd.DataFrame(storm_Y, index = time_idx, columns = ['dBH']))
corr_m_obs = np.zeros((len(station_obs),len(station_obs)))
for i in track(range(len(station_obs))):
    for j in range(i, len(station_obs)):
        data_i = station_obs[i]
        data_j = station_obs[j]
        data_joint = pd.merge(data_i, data_j, left_index=True, right_index=True)
        matrix_join = np.matrix([data_joint['dBH_x'], data_joint['dBH_y']])
        matrix_join = matrix_join.astype(np.float64)
        if matrix_join.shape[1] >= 100:
            corr_m_obs[i,j] = np.corrcoef(matrix_join)[0,1]
            corr_m_obs[j,i] = corr_m_obs[i,j]
station_ID = [station_file[i][0:3] for i in range(len(station_file))]
corr_pd_obs = pd.DataFrame(corr_m_obs, columns = station_ID, index = station_ID)

dissimilarity = np.sqrt(2*(1-corr_m_obs))
np.fill_diagonal(dissimilarity, 0)
Z = linkage(squareform(dissimilarity), 'complete')
threshold = 1.15
labels = fcluster(Z, threshold, criterion='distance')
station_info = pd.read_csv(root_path+"/data/Input/station_info.csv")
station_location = station_info.iloc[:,0:3]
station_location.columns = ["Station", "GEOLON", "GEOLAT"]
labeled_station = pd.DataFrame([station_ID, labels]).T
labeled_station.columns = ['Station', 'Cluster']
station = labeled_station.merge(station_location)
# dendrogram(Z, labels=corr_pd_obs.columns, orientation = 'top') 

obs_time = datetime.datetime(2024,5,10,0,0)
Earth_loc = astropy.coordinates.EarthLocation(lat=station["GEOLAT"].values*u.deg,
                                     lon=station["GEOLON"].values*u.deg)
station_lons = Earth_loc.lon.value  # Replace with your station longitudes
station_lats = Earth_loc.lat.value  # Replace with your station latitudes
center_lon=0
cmap = plt.get_cmap("RdYlBu_r", len(np.unique(labels)))
fig = plt.figure(figsize=(18, 10))
crs = ccrs.PlateCarree(central_longitude=center_lon)
ax = plt.axes(projection=crs)
ax.set_extent([-180, 179, 90, -90], ccrs.PlateCarree())
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
plt.scatter(
    x=station_lons,
    y=station_lats,
    c=station['Cluster'],
    cmap=cmap,
    s=100,
    alpha=1,
    transform=ccrs.PlateCarree(),
    marker="o",
    edgecolors='black'
)
for lat in [-60, -30, 0, 30, 60]:
    center_lon = 0
    # Define the extent of the map
    extent_lon_min = -180
    extent_lon_max = 179
    lon = np.linspace(extent_lon_min, extent_lon_max, 360)
    maglon, maglat = np.meshgrid(lon, lat)
    MAG_coord = frames.Geomagnetic(lon=maglon*u.deg, lat=maglat*u.deg, obstime=obs_time)
    target_coord = astropy.coordinates.ITRS(obstime=obs_time)
    geographic_coord = MAG_coord.transform_to(target_coord)
    pd_GEO = pd.DataFrame({'GEOLON': geographic_coord.spherical.lon.value.flatten(),
                           'GEOLAT': geographic_coord.spherical.lat.value.flatten()})
    pd_GEO.sort_values('GEOLON', inplace=True)
    plt.plot(pd_GEO['GEOLON'].values, pd_GEO['GEOLAT'].values, color='black', linestyle='-', linewidth=2,
            transform=ccrs.PlateCarree())
cbar = plt.colorbar(fraction=0.023, pad=0.05)
cbar.set_ticks([])
cbar.set_label(label='Clusters',size=20)
plt.title(r'Hierarchical Clustering Based on Station $dB_{H}$ Correlations', fontsize = 25)
plt.savefig(root_path+"/figure/TestSet3/clustering.png", bbox_inches='tight', dpi=300)

