# %%
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy
import datetime
import pickle
from torch.utils.data import DataLoader
import astropy
import astropy.units as u
import astropy.coordinates
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from sunpy.coordinates import frames
from cartopy.feature.nightshade import Nightshade
import time
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
from src.utils import *
from src.model import *


path = root_path + "/data/train/"
QoIs = "dBH"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
yeojohnson_lmbda = 1
window_max = 10
with open(path + "scaler/scalerX_%s.pkl" % window_max,'rb') as f:
    scalerX = pickle.load(f)
with open(path + "scaler/scalerY_%s_%s.pkl" % (QoIs, window_max),'rb') as f:
    scalerY = pickle.load(f)
X_shape = 96
make_image=True
test_time = False
OMNI = pd.read_pickle(path + "../Input/OMNI_May10_5m_feature_ACE.pkl").dropna()
torch.manual_seed(20241030)
model = DeepGaussianProcess(X_shape,
                            DEVICE,
                            num_hidden1_dims = 20,
                            num_hidden2_dims = 10,
                            num_hidden3_dims = 10)
state_dict = torch.load(path + "model/paper/model_state_dBH_10T.pth")
model.load_state_dict(state_dict)
if torch.cuda.is_available():
    model = model.cuda()

## numbers in paper
if test_time:
    repeat_num = 10
    ### location input can be calculated in advance:-------------------------------------------------------------
    obs_time=datetime.datetime(2024, 5, 10, 22, 36)
    # Storm_time = pd.DataFrame(OMNI.loc[obs_time]).T.index
    # Define the center of the map
    center_lon = 0
    # Define the extent of the map
    extent_lon_min = 0
    extent_lon_max = 359
    extent_lat_min = -90
    extent_lat_max = 90
    lat = np.linspace(extent_lat_min, extent_lat_max, 181)
    lon = np.linspace(extent_lon_min, extent_lon_max, 180)
    geolon, geolat = np.meshgrid(lon, lat)
    time_evaluation = []
    for j in range(repeat_num):
        start_time = time.time()
        Earth_loc = astropy.coordinates.EarthLocation(lat=geolat*u.deg, lon=geolon*u.deg)
        Geographic = astropy.coordinates.ITRS(x=Earth_loc.x, y=Earth_loc.y, z=Earth_loc.z, obstime=obs_time)
        target_coord = frames.SolarMagnetic(magnetic_model='igrf13', obstime=obs_time)
        SM_coord = Geographic.transform_to(target_coord)
        SM_lat = SM_coord.lat.value
        SM_lon_cos = np.cos(SM_coord.lon.value/ 180 * np.pi)
        SM_lon_sine = np.sin(SM_coord.lon.value / 180 * np.pi)
        SM_df = np.column_stack((SM_lat.flatten(), SM_lon_cos.flatten(), SM_lon_sine.flatten()))
        base = OMNI.loc[obs_time].to_numpy().reshape(1,-1).repeat(SM_df.shape[0], axis=0)
        Grid_X = np.hstack((base, SM_df))
        Grid_X = torch.tensor(scalerX.transform(Grid_X), dtype=torch.float32)
        Grid_dataset = SuperMAGDataset_fromtensor(Grid_X, torch.zeros(Grid_X.shape[0], dtype=torch.float32))
        Grid_loader = DataLoader(Grid_dataset,
                                batch_size=1024)
        ## Model Evaluation
        model.eval()
        grid_mean, grid_var, grid_lls = model.predict(Grid_loader)
        pred_Y_DGP = grid_mean.mean(0).cpu().numpy().reshape(-1,1)
        ## -------------------------------
        sigma = sigma_GMM(grid_mean, grid_var)
        qnorm_975 = scipy.stats.norm.ppf(0.975, loc=0, scale=1)
        pred_Y_DGP_lr = pred_Y_DGP - qnorm_975*sigma.reshape(-1,1)
        pred_Y_DGP_up = pred_Y_DGP + qnorm_975*sigma.reshape(-1,1)
        z = scalerY.inverse_transform(pred_Y_DGP).reshape(SM_lat.shape)
        z_lr = scalerY.inverse_transform(pred_Y_DGP_lr).reshape(SM_lat.shape)
        z_up = scalerY.inverse_transform(pred_Y_DGP_up).reshape(SM_lat.shape)
        # Record the end time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        time_evaluation.append(elapsed_time)
    print("The average evaluation time on a 360 (longitude) x 180 (latitude) grids is %.2f" % np.mean(time_evaluation))
# %%
if make_image:
    obs_time=datetime.datetime(2024, 5, 10, 22, 36)
    # Storm_time = pd.DataFrame(OMNI.loc[obs_time]).T.index
    # Define the center of the map
    center_lon = 0
    # Define the extent of the map
    extent_lon_min = 0
    extent_lon_max = 359
    extent_lat_min = -90
    extent_lat_max = 90
    lat = np.linspace(extent_lat_min, extent_lat_max, 181)
    lon = np.linspace(extent_lon_min, extent_lon_max, 360)
    geolon, geolat = np.meshgrid(lon, lat)
    Earth_loc = astropy.coordinates.EarthLocation(lat=geolat*u.deg, lon=geolon*u.deg)
    Geographic = astropy.coordinates.ITRS(x=Earth_loc.x, y=Earth_loc.y, z=Earth_loc.z, obstime=obs_time)
    target_coord = frames.SolarMagnetic(magnetic_model='igrf13', obstime=obs_time)
    SM_coord = Geographic.transform_to(target_coord)
    SM_lat = SM_coord.lat.value
    SM_lon_cos = np.cos(SM_coord.lon.value/ 180 * np.pi)
    SM_lon_sine = np.sin(SM_coord.lon.value / 180 * np.pi)
    SM_df = np.column_stack((SM_lat.flatten(), SM_lon_cos.flatten(), SM_lon_sine.flatten()))
    base = OMNI.loc[obs_time].to_numpy().reshape(1,-1).repeat(SM_df.shape[0], axis=0)
    Grid_X = np.hstack((base, SM_df))
    Grid_X = torch.tensor(scalerX.transform(Grid_X), dtype=torch.float32)
    Grid_dataset = SuperMAGDataset_fromtensor(Grid_X, torch.zeros(Grid_X.shape[0], dtype=torch.float32))
    Grid_loader = DataLoader(Grid_dataset,
                            batch_size=1024)
    ## Model Evaluation
    model.eval()
    grid_mean, grid_var, grid_lls = model.predict(Grid_loader)
    z = scalerY.inverse_transform(grid_mean.cpu().mean(0).numpy().reshape(-1,1)).reshape(-1).reshape(SM_lat.shape)
    ## -------------------------------
    pred_Y_DGP = grid_mean.mean(0).cpu().numpy().reshape(-1,1)
    vmin=0
    vmax=1200
    latitude_label_lon = 130


# In[ ]:


if make_image:
    #station information can be downloaded here: SuperMag---download data--scroll down to bottom
    ## --About SuperMAG Data --Station Information
    station_path = path + '../Input/Gannon_storm/AllStations_Gannon/'
    station_file = sorted(os.listdir(station_path))
    station_info = pd.read_csv(path + '../Input/station_info.csv')
    all_stations = sorted(list(station_info['IAGA']))
    res = []
    for i in range(len(station_file)):
        if station_file[i][0:3] in all_stations:
            file_path = station_path + station_file[i]
            station_data = pd.read_pickle(file_path)
            station_data.index = station_data['Time']
            obs_dBH = station_data.loc[obs_time]['dBH']
            station_dBH = pd.concat(
                [station_info[station_info['IAGA'] == 
                              station_file[i][0:3]][['IAGA', 'GEOLON', 'GEOLAT']].reset_index(drop=True,inplace=False),
                   pd.DataFrame(station_data.loc[obs_time]).T['dBH'].reset_index(drop=True, inplace=False)],
                  axis = 1)
            if not np.isnan(station_dBH['dBH'].item()):
                res.append(station_dBH)
    obsered_dBH = pd.concat(res,axis = 0)
    plt.clf()
    ## Contour plot
    plt.style.use('default')
    cmap = 'RdYlBu_r' 
    fig = plt.figure(figsize=(16, 8))
    crs = ccrs.PlateCarree(central_longitude=center_lon)
    ax = plt.axes(projection=crs)
    ax.coastlines(resolution='110m', color='black', linewidth=0.7)
    gl = ax.gridlines(crs=crs, draw_labels=True, linewidth=1, color='black', linestyle='--')
    ax.set_extent([0, 359, -90, 90], ccrs.PlateCarree())
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 16, 'color': 'black'}
    gl.ylabel_style = {'size': 16, 'color': 'black'}
    plt.pcolormesh(geolon, geolat, z, transform=ccrs.PlateCarree(), cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(fraction=0.014, pad=0.06, aspect=30)
    cbar.set_label(label='d$B_H$ [nT]',size=25)
    cbar.ax.tick_params(labelsize=20)
    plt.title('GeoDGP ' + obs_time.strftime("%Y-%m-%d %H:%M:%S UT"), fontsize = 30)
    plt.scatter(
        x=obsered_dBH["GEOLON"],
        y=obsered_dBH["GEOLAT"],
        c=obsered_dBH["dBH"],
        cmap=cmap,  #this is the changes
        s=75,
        alpha=1,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        marker="o",
        edgecolors='black'
    )
    ax.add_feature(Nightshade(obs_time, alpha=0.15))
    plt.savefig(root_path + "/figure/TestSet3/GeoDGP_global.png", bbox_inches='tight', dpi=300)

if make_image:
#     start_time = time.time()
    cmap = 'RdYlBu_r' 
    station_path = path + '../Input/Gannon_storm/AllStations_Gannon/'
    station_file = sorted(os.listdir(station_path))
    station_info = pd.read_csv(path + '../Input/station_info.csv')
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
            upperhalf_stations = sorted(list(station_info[station_info['GEOLAT']>30]['IAGA']))
        elif isinstance(projections[i], ccrs.SouthPolarStereo):
            ax.set_extent([0, 359, -40, -90], ccrs.PlateCarree())
            upperhalf_stations = sorted(list(station_info[station_info['GEOLAT']<-30]['IAGA']))
        ## scatter plots
        res = []
        for j in range(len(station_file)):
            if station_file[j][0:3] in upperhalf_stations:
                file_path = station_path + station_file[j]
                station_data = pd.read_pickle(file_path)
                station_data.index = station_data['Time']
                obs_dBH = station_data.loc[obs_time]['dBH']
                station_dBH = pd.concat(
                    [station_info[station_info['IAGA'] == 
                                  station_file[j][0:3]][['IAGA', 'GEOLON', 'GEOLAT']].reset_index(drop=True,inplace=False),
                       pd.DataFrame(station_data.loc[obs_time]).T['dBH'].reset_index(drop=True, inplace=False)],
                      axis = 1)
                if not np.isnan(station_dBH['dBH'].item()):
                    res.append(station_dBH)
        obsered_dBH = pd.concat(res,axis = 0)
    #     plt.show()
        # Add gridlines
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1.5, color='black', alpha=0.7, linestyle='--')
        gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 16, 'color': 'black', 'rotation': 0}
#         gl.ylabel_style = {'size': 30, 'color': 'black'}
        gl.ylabel_style = {'size': 16, 'color': 'black', 'weight': 'bold'}
        ax.coastlines(resolution='50m', color='black', linewidth=1.2)
        plt.draw()
        for ea in gl.left_label_artists+gl.right_label_artists:
            ea.set_visible(True)
    # for i in range(13,len(gl.label_artists)):
    #     position = gl.label_artists[i].get_position()
    #     gl.label_artists[i].set_position((-150, position[1]))
        im = ax.pcolormesh(geolon, geolat, z,
                               cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax.scatter(
            x=obsered_dBH["GEOLON"],
            y=obsered_dBH["GEOLAT"],
            c=obsered_dBH["dBH"],
            cmap=cmap,  #this is the changes
            s=150,
            alpha=1,
            transform=ccrs.PlateCarree(),
            vmin=vmin,
            vmax=vmax,
            marker="o",
            edgecolors='black',
            linewidth=1
        )
        ax.add_feature(Nightshade(obs_time, alpha=0.15))
    #     ax.set_title(f'10 May 2024 22:30:00 - 22:40:00 UT {i+1}', fontsize=20)

    # Add a single colorbar for both subplots
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04, aspect=50, orientation="horizontal")
    cbar.set_label(label='d$B_H$ [nT]', size=30)
    cbar.ax.tick_params(labelsize=25)
    fig.suptitle('GeoDGP dB$_H$ ' + obs_time.strftime("%Y-%m-%d %H:%M:%S UT"), fontsize=35, y=0.95)
    # plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()
    fig.savefig(root_path + "/figure/TestSet3/GeoDGP_Polar.png", bbox_inches='tight', dpi=300)
#     end_time = time.time()
#     end_time-start_time
# %%
