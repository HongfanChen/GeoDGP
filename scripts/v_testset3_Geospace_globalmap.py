import spacepy.pybats.bats as pybats
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import pyplot as plt
import datetime
import os
import pandas as pd
import matplotlib.ticker as mticker
from cartopy.feature.nightshade import Nightshade
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
make_image=True
if make_image:
    ## ---------------------------------------------- Map specification -----------------------------------------
    # Define the center of the map
    center_lon = 0
    # Define the extent of the map
    extent_lon_min = 0
    extent_lon_max = 359
    extent_lat_min = 90
    extent_lat_max = -90
    lat = np.linspace(extent_lat_min, extent_lat_max, 181)
    lon = np.linspace(extent_lon_min, extent_lon_max, 360)
    geolon, geolat = np.meshgrid(lon, lat)
    vmin=0
    vmax=1200
    latitude_label_lon = 130
    ## ---------------------------------------------- Generate data -----------------------------------------
    obs_time = datetime.datetime(2024, 5, 10, 22, 36)
    file_prefix =root_path+'/data/Geospace_Gannon_MAGGRID/mag_grid_global_e'
    filename = file_prefix + '%s%02d%02d-%02d%02d26.out' % (obs_time.year, obs_time.month, obs_time.day,
                                                            obs_time.hour, obs_time.minute)
#     print(filename)
    ## ---------------------------------------------- Load Geospace data -----------------------------------------
    data = pybats.Bats2d(filename)
    dbe = np.array(data['dbe'])
    dbn = np.array(data['dbn'])
    dBH = np.sqrt(dbe**2+dbn**2)
    z = np.rot90(dBH)
    ## 2nd order Taylor expansion in the row direction.
    gxx, gyy = np.gradient(z)
    ggxx, _ = np.gradient(gxx)
    for i in range(5):
        z = np.concatenate((np.tile(np.nan, (1,z.shape[1])), z), axis=0)
        z[0, :] = z[1, :] + (-1)*gxx[0, :]  + (-1) * 0.5 * (ggxx[0,:])
        z = np.concatenate((z, np.tile(np.nan, (1,z.shape[1]))), axis=0)
        z[-1, :] = z[-2, :] + gxx[-1, :] + 0.5 * (ggxx[-1,:])
    ## Contour plot
    plt.style.use('default')
    ## ---------------------------------------- Figure 1: Global Map ----------------------------------------
    cmap = 'RdYlBu_r'
    fig = plt.figure(figsize=(16, 8))
    crs = ccrs.PlateCarree(central_longitude=center_lon)
    ax = plt.axes(projection=crs)
    ax.coastlines(resolution='110m', color='black', linewidth=0.7)
    ax.set_extent([0, 359, 90, -90], ccrs.PlateCarree())
    gl = ax.gridlines(crs=crs, draw_labels=True, linewidth=1, color='black', linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 16, 'color': 'black'}
    gl.ylabel_style = {'size': 16, 'color': 'black'}
    # base_obj = plt.contourf(geolon, geolat, np.rot90(dBH), levels = levels, cmap='RdYlBu_r', origin='upper')
    plt.pcolormesh(geolon, geolat, z, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(fraction=0.014, pad=0.06, aspect=30)
    cbar.set_label(label='d$B_H$ [nT]',size=25)
    cbar.ax.tick_params(labelsize=20)
    ## add stations
    station_path = root_path+'/data/Input/Gannon_storm/AllStations_Gannon/'
    station_file = sorted(os.listdir(station_path))
    station_info = pd.read_csv(root_path+'/data/Input/station_info.csv')
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
    plt.title('Geospace ' + obs_time.strftime("%Y-%m-%d %H:%M:%S UT"), fontsize = 30)
    plt.savefig(root_path+"/figure/TestSet3/Geospace_global.png", bbox_inches='tight', dpi=300)
    plt.close()


if make_image:
    ## ---------------------------------------------- Map specification -----------------------------------------
    # Define the center of the map
    center_lon = 0
    # Define the extent of the map
    extent_lon_min = 0
    extent_lon_max = 359
    extent_lat_min = 90
    extent_lat_max = -90
    lat = np.linspace(extent_lat_min, extent_lat_max, 181)
    lon = np.linspace(extent_lon_min, extent_lon_max, 360)
    geolon, geolat = np.meshgrid(lon, lat)
    vmin=0
    vmax=1200
    latitude_label_lon = 130
    ## ---------------------------------------------- Generate data -----------------------------------------
    station_path = root_path + '/data/Input/Gannon_storm/AllStations_Gannon/'
    station_file = sorted(os.listdir(station_path))
    station_info = pd.read_csv(root_path+'/data/Input/station_info.csv')
    obs_time = datetime.datetime(2024, 5, 10, 22, 36)
    file_prefix =root_path + '/data/Geospace_Gannon_MAGGRID/mag_grid_global_e'
    filename = file_prefix + '%s%02d%02d-%02d%02d26.out' % (obs_time.year, obs_time.month, obs_time.day,
                                                            obs_time.hour, obs_time.minute)
    ## ---------------------------------------------- Load Geospace data -----------------------------------------
    data = pybats.Bats2d(filename)
    dbe = np.array(data['dbe'])
    dbn = np.array(data['dbn'])
    dBH = np.sqrt(dbe**2+dbn**2)
    z = np.rot90(dBH)
    ## 2nd order Taylor expansion in the row direction.
    gxx, gyy = np.gradient(z)
    ggxx, _ = np.gradient(gxx)
    for i in range(5):
        z = np.concatenate((np.tile(np.nan, (1,z.shape[1])), z), axis=0)
        z[0, :] = z[1, :] + (-1)*gxx[0, :]  + (-1) * 0.5 * (ggxx[0,:])
        z = np.concatenate((z, np.tile(np.nan, (1,z.shape[1]))), axis=0)
        z[-1, :] = z[-2, :] + gxx[-1, :] + 0.5 * (ggxx[-1,:])
    # Define parameters
    center_lon = 0
    cmap = 'RdYlBu_r'
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
    fig.suptitle('Geospace dB$_H$ ' + obs_time.strftime("%Y-%m-%d %H:%M:%S UT"), fontsize=35, y=0.95)
    # plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()
    fig.savefig(root_path+"/figure/TestSet3/Geospace_Polar.png", bbox_inches='tight', dpi=300)




