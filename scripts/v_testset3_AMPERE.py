import netCDF4 as nc
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import griddata
from cartopy.feature.nightshade import Nightshade
import datetime
import aacgmv2
import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

def transform_longitude_np(longitudes):
    """
    Transform a NumPy array of longitudes from [-180, 180] to [0, 360].
    
    Parameters:
        longitudes (numpy.ndarray): Array of longitudes in the range [-180, 180].
        
    Returns:
        numpy.ndarray: Array of longitudes transformed to the range [0, 360].
    """
    return (longitudes + 360) % 360

file_path = root_path + "/data/AMPERE/20240510.2230.600.600.north.grd.ncdf"
dataset = nc.Dataset(file_path)
obs_time=datetime.datetime(2024, 5, 10, 22, 36)

# mlon = dataset.variables['mlt_hr'][:].data.reshape(-1) * 15
mlt = dataset.variables['mlt_hr'][:].data.reshape(-1)
mlon = np.array(aacgmv2.convert_mlt(mlt, obs_time, m2a=True))
mlon = transform_longitude_np(mlon)
mlat = 90-dataset.variables['cLat_deg'][:].data.reshape(-1)
z = dataset.variables['jPar'][:].data.reshape(-1)

## longitude augmentation: ---------------------------------------------------
left = np.where(mlon<45)
right = np.where(mlon>315)
## data for right append
mlon_right_append = mlon[left]+360
mlat_right_append = mlat[left]
z_right_append = z[left]
## data for left append
mlon_left_append = mlon[right]-360
mlat_left_append = mlat[right]
z_left_append = z[right]
## right append
mlon_aug = np.concatenate((mlon, mlon_right_append))
mlat_aug = np.concatenate((mlat, mlat_right_append))
z_aug = np.concatenate((z, z_right_append))
## left append
mlon_aug = np.concatenate((mlon_left_append, mlon_aug))
mlat_aug = np.concatenate((mlat_left_append, mlat_aug))
z_aug = np.concatenate((z_left_append, z_aug))

## create grids
extent_lon_min = 0
extent_lon_max = 359
extent_lat_min = 40
extent_lat_max = 90
## Transformation using AACGM
geolon, geolat = np.meshgrid(np.linspace(extent_lon_min, extent_lon_max, 360),
                                 np.linspace(extent_lat_min, extent_lat_max, 51))
# Initialize output arrays with the same shape as the input
aacgm_lons = np.empty_like(geolon, dtype=np.float64)
aacgm_lats = np.empty_like(geolat, dtype=np.float64)

# Convert each element
for i in range(geolon.shape[0]):
    for j in range(geolat.shape[1]):
        lat = geolat[i, j]
        lon = geolon[i, j]
        aacgm_lat, aacgm_lon, _ = aacgmv2.get_aacgm_coord(lat, lon, 780, obs_time)  # Altitude is 780 km
        aacgm_lons[i, j] = aacgm_lon
        aacgm_lats[i, j] = aacgm_lat
## [0,360]
aacgm_lons = transform_longitude_np(aacgm_lons)
## interpolation in AACGM lon and lat.
z_grid_geo = griddata((mlon_aug, mlat_aug), z_aug, (aacgm_lons, aacgm_lats), method='cubic')

file_path = root_path+"/data/AMPERE/20240510.2230.600.600.south.grd.ncdf"
dataset = nc.Dataset(file_path)
obs_time=datetime.datetime(2024, 5, 10, 22, 36)


# In[ ]:


# mlon = dataset.variables['mlt_hr'][:].data.reshape(-1) * 15
mlt = dataset.variables['mlt_hr'][:].data.reshape(-1)
mlon = np.array(aacgmv2.convert_mlt(mlt, obs_time, m2a=True))
mlon = transform_longitude_np(mlon)
mlat = 90-dataset.variables['cLat_deg'][:].data.reshape(-1)
z = dataset.variables['jPar'][:].data.reshape(-1)


# In[ ]:


## longitude augmentation: ---------------------------------------------------
left = np.where(mlon<45)
right = np.where(mlon>315)
## data for right append
mlon_right_append = mlon[left]+360
mlat_right_append = mlat[left]
z_right_append = z[left]
## data for left append
mlon_left_append = mlon[right]-360
mlat_left_append = mlat[right]
z_left_append = z[right]
## right append
mlon_aug = np.concatenate((mlon, mlon_right_append))
mlat_aug = np.concatenate((mlat, mlat_right_append))
z_aug = np.concatenate((z, z_right_append))
## left append
mlon_aug = np.concatenate((mlon_left_append, mlon_aug))
mlat_aug = np.concatenate((mlat_left_append, mlat_aug))
z_aug = np.concatenate((z_left_append, z_aug))
## create grids
extent_lon_min = 0
extent_lon_max = 359
extent_lat_min = -90
extent_lat_max = -40
## Transformation using AACGM
geolon_S, geolat_S = np.meshgrid(np.linspace(extent_lon_min, extent_lon_max, 360),
                                 np.linspace(extent_lat_min, extent_lat_max, 51))
# Initialize output arrays with the same shape as the input
aacgm_lons = np.empty_like(geolon_S, dtype=np.float64)
aacgm_lats = np.empty_like(geolat_S, dtype=np.float64)

# Convert each element
for i in range(geolon_S.shape[0]):
    for j in range(geolat_S.shape[1]):
        lat = geolat_S[i, j]
        lon = geolon_S[i, j]
        aacgm_lat, aacgm_lon, _ = aacgmv2.get_aacgm_coord(lat, lon, 780, obs_time)  # Altitude is 780 km
        aacgm_lons[i, j] = aacgm_lon
        aacgm_lats[i, j] = aacgm_lat
## [0,360]
aacgm_lons = transform_longitude_np(aacgm_lons)
## interpolation in AACGM lon and lat.
z_grid_geo_S = griddata((mlon_aug, mlat_aug), z_aug, (aacgm_lons, aacgm_lats), method='cubic')
# Define parameters
center_lon = 0
cmap = 'Reds'
# Define projections for each subplot
projections = [ccrs.NorthPolarStereo(central_longitude=center_lon),
               ccrs.SouthPolarStereo(central_longitude=center_lon)]  # North and South Polar projections

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': projections[0]})

for i, ax in enumerate(axes):
    # Set the appropriate projection for each axis
    ax.projection = projections[i]

    # Set extent based on projection
    if isinstance(projections[i], ccrs.NorthPolarStereo):
        ax.set_extent([0, 359, 90, 40], ccrs.PlateCarree())
    elif isinstance(projections[i], ccrs.SouthPolarStereo):
        ax.set_extent([0, 359, -40, -90], ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1.5, color='black', alpha=0.7, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 16, 'color': 'black', 'rotation': 0}
    gl.ylabel_style = {'size': 16, 'color': 'black', 'weight': 'bold'}
    ax.coastlines(resolution='50m', color='black', linewidth=1.2)
    plt.draw()
    for ea in gl.left_label_artists+gl.right_label_artists:
        ea.set_visible(True)
    if i == 0:
    # Plot data
        for j in range(13,len(gl.label_artists)):
            position = gl.label_artists[j].get_position()
            gl.label_artists[j].set_position((-145, position[1]))
        im = ax.pcolormesh(geolon, geolat, np.absolute(z_grid_geo),
                           cmap=cmap, vmin=0.3, vmax=2, transform=ccrs.PlateCarree())
    else:
        im = ax.pcolormesh(geolon_S, geolat_S, np.absolute(z_grid_geo_S),
                           cmap=cmap, vmin=0.3, vmax=2, transform=ccrs.PlateCarree())
    ax.add_feature(Nightshade(obs_time, alpha=0.15))
# Add a single colorbar for both subplots
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, aspect=50, orientation="horizontal")
cbar.set_label(label='Current density in absolute value [$\mu$A/m$^2$]', size=30)
cbar.ax.tick_params(labelsize=25)
fig.suptitle('AMPERE Current Density 2024-05-10 22:30:00 - 22:40:00 UT', fontsize=35, y=0.95)
# plt.tight_layout() 
plt.show()
fig.savefig(root_path+"/figure/TestSet3/AMPERE_FACs.png", bbox_inches='tight', dpi=300)



