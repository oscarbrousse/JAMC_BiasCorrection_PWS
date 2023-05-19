# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:16:21 2021

@author: Oscar Brousse
"""
# %%
import KDTreeIndex as KDT
import glob
import time
from pathlib import Path
import ast

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as clrs
from matplotlib.colors import CenteredNorm
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.sample
import seaborn as sns
import xarray as xr
from rasterio.warp import Resampling, reproject
from scipy import spatial

from sklearn import metrics as skm
from scipy import stats

##########################
### MISCELLANEOUS INFO ###
##########################

city = 'London'
datadir_WRF = r''
datadir_WRF_geog = r''
savedir = datadir_WRF + ''
datadir_NA = r''
datadir_lcz = r''

startdate = '2018-06-01'
enddate = '2018-08-31'
resample_starth = '2018-05-31 09:00:00'
dates_list = [d.strftime('%Y-%m-%d') for d in pd.date_range(startdate, enddate, freq='1d').to_list()]

#########################
### MODEL SIMULATIONS ###
#########################

geogrid_d01 = datadir_WRF_geog + 'geo_em.d01.nc'
surf_info_d01 = xr.open_dataset(geogrid_d01)

geogrid_d02 = datadir_WRF_geog + 'geo_em.d02.nc'
surf_info_d02 = xr.open_dataset(geogrid_d02)

geogrid_d03 = datadir_WRF_geog + 'geo_em.d03_LCZ_params.nc'
surf_info_d03 = xr.open_dataset(geogrid_d03)

WRF_grid = datadir_WRF_geog + 'geo_em.d03_gridinfo.tif'
WRF_tif = xr.open_rasterio(WRF_grid)

lat_d1=surf_info_d01.XLAT_M.squeeze().values
lon_d1=surf_info_d01.XLONG_M.squeeze().values
min_lat_d1 = np.min(lat_d1); max_lat_d1 = np.max(lat_d1)
min_lon_d1 = np.min(lon_d1); max_lon_d1 = np.max(lon_d1)

lat_d2=surf_info_d02.XLAT_M.squeeze().values
lon_d2=surf_info_d02.XLONG_M.squeeze().values

lat_d3=surf_info_d03.XLAT_M.squeeze().values
lon_d3=surf_info_d03.XLONG_M.squeeze().values

lu_d3 = xr.DataArray(surf_info_d03.LU_INDEX.values.squeeze(), dims=['latitude', 'longitude'],
                            coords = {'lat': (('latitude', 'longitude'), lat_d3),
                                      'lon': (('latitude', 'longitude'), lon_d3)})

LU_INDEX = surf_info_d03.LU_INDEX.values.squeeze()
urb_mask = np.where(LU_INDEX < 30, np.nan, 1)

ground_pixel_tree = KDT.KDTreeIndex(lu_d3)
ds_wrf = xr.open_dataset(datadir_WRF + "WRF_Urb_BouLac_T2-V10-U10_20180525-20180831.nc")


#########################
### AWS OBSERVATIONS  ###
#########################

datadir_MIDAS = r'' + city + '/Filtered/'
spat_at_MIDAS = pd.read_csv(datadir_MIDAS + 'Per_Variable/List_MIDAS_stations_' + city + '_Filt.csv',
                            index_col = 0)
spat_at_MIDAS = spat_at_MIDAS[(spat_at_MIDAS.Lon > np.min(lon_d3)) &
                                (spat_at_MIDAS.Lon < np.max(lon_d3)) &
                                (spat_at_MIDAS.Lat > np.min(lat_d3)) &
                                (spat_at_MIDAS.Lat < np.max(lat_d3))]

#########################
### CWS OBSERVATIONS  ###
#########################

### General attributes per station
df_netatmo_id = pd.read_csv(
    datadir_NA + "List_Netatmo_stations_London_WRF_2015-2020.csv"
)
df_netatmo_id = df_netatmo_id.set_index("ID")

df_na = pd.concat(
    [
        pd.read_csv(p)
        for p in Path(datadir_NA).glob("Netatmo_London_2018-*_filt_temp_by_ID.csv")
    ]
)
df_na = df_na.rename(columns={"Unnamed: 0": "time"})
df_na["time"] = pd.to_datetime(df_na.time)
df_na = df_na.set_index("time")
df_na = df_na[startdate:enddate]
df_na = df_na.dropna(axis=1, how='all')

df_na_filt = pd.concat(df_na.loc[date].dropna(thresh=20, axis=1)
                       for date in dates_list
                       )

spat_at_NA_sub = df_netatmo_id.loc[df_na_filt.columns]
spat_at_NA_sub = spat_at_NA_sub[(spat_at_NA_sub.Lon > np.min(lon_d3)) &
                                (spat_at_NA_sub.Lon < np.max(lon_d3)) &
                                (spat_at_NA_sub.Lat > np.min(lat_d3)) &
                                (spat_at_NA_sub.Lat < np.max(lat_d3))]
NA_filt_df = df_na_filt[spat_at_NA_sub.index]

gdf_netatmo_id = gpd.GeoDataFrame(
    df_netatmo_id,
    geometry=gpd.points_from_xy(df_netatmo_id.Lon, df_netatmo_id.Lat),
    crs="EPSG:4326",
)

### We use KDT closest distance on curvilinear grid. 
### The Xarray sel option only authorizes 1 dimensional indexing

latlon_NA = np.column_stack((np.array(spat_at_NA_sub.Lat).ravel(),
                             np.array(spat_at_NA_sub.Lon).ravel()))
latlon_NA_tuple = list(map(tuple, latlon_NA))

### Locate the Netatmo stations on the WRF grid
ground_pixel_tree = KDT.KDTreeIndex(lu_d3)
stations_index = ground_pixel_tree.query(latlon_NA_tuple)

spat_at_NA_sub['I'] = stations_index[1]
spat_at_NA_sub['J'] = stations_index[0]
spat_at_NA_sub['LU'] = np.nan
for NA_ID in spat_at_NA_sub.index.values.astype(str):
    i = spat_at_NA_sub.I.loc[NA_ID]
    j = spat_at_NA_sub.J.loc[NA_ID]
    spat_at_NA_sub.loc[NA_ID, "LU"] = lu_d3[j,i]
spat_at_NA_sub_urb = spat_at_NA_sub[spat_at_NA_sub.LU > 30]

#########################
###   WRF CWS PARAMS  ###
#########################

gdf_wrf = gpd.read_file(
    datadir_WRF + "wrf_voronoi/wrf_voronoi.shp"
)
gdf_wrf= gdf_wrf.rename(columns = {'south_nort':'south_north'})

## Since we run the regressors on a daily timestep we do not resample prior to joining
## the temperature to the GDF grid info. This is done later within a loop
gdf_wrf= gdf_wrf.set_index(["west_east", "south_north"]).join(ds_wrf[["XLAT","XLONG"]].to_dataframe())

### Link CWS to dataset of predictors
gdf_LU_joined = gdf_wrf.set_index(["XLAT", "XLONG"]).join(
    lu_d3.to_dataframe(name="lu_index")
    .rename(columns={"lon": "XLONG", "lat": "XLAT"})
    .set_index(["XLAT", "XLONG"])
)

gdf_join = gdf_LU_joined.sjoin(gdf_netatmo_id, predicate="contains")
### Need to ensure consistancy between the bias-correction and the evaluation
### Slight discrepancy probably due to projection and methods for localizing PWS on WRF domain
gdf_na_urb = gdf_join[gdf_join.lu_index>30].copy()
spat_at_NA_sub_urb = spat_at_NA_sub_urb[spat_at_NA_sub_urb.index.isin(gdf_na_urb.index_right)]

#%% 
#########################
###  PLOTS BC PAPER   ###
#########################

list_daily_stat = ["mean", "max", "min"]

### Colormap scaler
min_val, max_val = (0.1,1)
n = 10

cmap_temp_orig = plt.cm.Reds.copy()
colors_temp = cmap_temp_orig(np.linspace(min_val, max_val, n))
cmap_temp = clrs.LinearSegmentedColormap.from_list("mycmap", colors_temp)
cmap_temp.set_bad('lightgrey', alpha=1)

cmap_bias = plt.cm.RdBu_r.copy()
cmap_bias.set_bad('lightgrey', alpha=1)

cmap_mb = cmocean.cm.delta.copy()
cmap_bias.set_bad('white', alpha=1)

lcz_colors_dict =  {0:'#FFFFFF', 1:'#910613', 2:'#D9081C', 3:'#FF0A22', 4:'#C54F1E', 5:'#FF6628', 6:'#FF985E', 
                    7:'#FDED3F', 8:'#BBBBBB', 9:'#FFCBAB',10:'#565656', 11:'#006A18', 12:'#00A926', 
                    13:'#628432', 14:'#B5DA7F', 15:'#000000', 16:'#FCF7B1', 17:'#656BFA', 18:'#00ffff'}

lcz_labels = ['Mask', 'Compact High Rise: LCZ 1', 'Compact Mid Rise: LCZ 2', 'Compact Low Rise: LCZ 3', 
              'Open High Rise: LCZ 4', 'Open Mid Rise: LCZ 5', 'Open Low Rise: LCZ 6',
              'Lighweight Lowrise: LCZ 7', 'Large Lowrise: LCZ 8',
              'Sparsely Built: LCZ 9', 'Heavy Industry: LCZ 10',
              'Dense Trees: LCZ A', 'Sparse Trees: LCZ B', 'Bush - Scrubs: LCZ C',
              'Low Plants: LCZ D', 'Bare Rock - Paved: LCZ E', 'Bare Soil - Sand: LCZ F',
              'Water: LCZ G', 'Wetlands: LCZ W']
lcz_labels_dict = dict(zip(list(lcz_colors_dict.keys()),lcz_labels))

cmap_lcz = mpl.colors.ListedColormap(list(lcz_colors_dict.values()))
lcz_classes = list(lcz_colors_dict.keys()); lcz_classes.append(20)
norm_lcz = mpl.colors.BoundaryNorm(lcz_classes, cmap_lcz.N)

### Set colormap of terrain height and LCZ classes
HSURF_d1 = surf_info_d01.HGT_M.values[0,:,:]
min_t = np.floor(np.nanmin(HSURF_d1))
max_t = np.ceil(np.nanmax(HSURF_d1))
rng_t = max_t - min_t

cmap_t = plt.cm.get_cmap('terrain', rng_t).copy()
cmap_ticks = np.int16(np.around(np.linspace(int(min_t),int(max_t),2, endpoint = True)))
cmap_ticks_label = cmap_ticks.astype(str)

### Set colormap for stats visualization
cmap_stat = plt.cm.cividis.copy()
cmap_stat.set_bad('white', alpha=0)

### Set reversed colormap for stats visualization
cmap_stat_r = plt.cm.cividis_r.copy()
cmap_stat_r.set_bad('white', alpha=0)

#%%
##########
### Figure 2 : DOMAIN NESTING WRF
##########

cmap_lcz_d3 = mpl.colors.ListedColormap([
                lcz_colors_dict.get(lcz) for lcz in np.unique((LU_INDEX - 30)*urb_mask)[:-1]]
    )
lcz_classes_d3 = list(np.unique((LU_INDEX - 30)*urb_mask)[:-1].astype(int))
lcz_labels_d3 = [lcz_labels[i] for i in lcz_classes_d3]
lcz_classes_d3.append(11)
norm_lcz_d3 = mpl.colors.BoundaryNorm(lcz_classes_d3, cmap_lcz_d3.N)

### Plot domain nesting in upper panel and urban land use + weather stations in bottom panel

col = 1; row = 2
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(row, col, figsize = (12,4.7*row), subplot_kw=dict(projection=proj))
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.95)
ax = ax.flatten()

im = ax[0].pcolormesh(lon_d1, lat_d1, HSURF_d1, vmin = min_t, vmax = max_t, cmap = cmap_t)
ax[0].set_extent((min_lon_d1-0.5, max_lon_d1+0.5, min_lat_d1-0.5, max_lat_d1-0.5))
ax[0].coastlines(color='k', linewidth=0.5)
# Plot boundaries of 2nd domain
ax[0].plot(lon_d2[0,:],lat_d2[0,:], color = 'grey', transform=ccrs.PlateCarree(), linewidth = 2)
ax[0].plot(lon_d2[:,0],lat_d2[:,0], color = 'grey', transform=ccrs.PlateCarree(), linewidth = 2)
ax[0].plot(lon_d2[-1,:],lat_d2[-1,:], color = 'grey', transform=ccrs.PlateCarree(), linewidth = 2)
ax[0].plot(lon_d2[:,-1],lat_d2[:,-1], color = 'grey', transform=ccrs.PlateCarree(), linewidth = 2)
#Plot boundaries of 3rd domain and ISA in it
ax[0].plot(lon_d3[0,:],lat_d3[0,:], color = 'dimgrey', transform=ccrs.PlateCarree(), linewidth = 2)
ax[0].plot(lon_d3[:,0],lat_d3[:,0], color = 'dimgrey', transform=ccrs.PlateCarree(), linewidth = 2)
ax[0].plot(lon_d3[-1,:],lat_d3[-1,:], color = 'dimgrey', transform=ccrs.PlateCarree(), linewidth = 2)
ax[0].plot(lon_d3[:,-1],lat_d3[:,-1], color = 'dimgrey', transform=ccrs.PlateCarree(), linewidth = 2)

ax[0].text(s="D1", x = np.max(lon_d1)-2.5, y = np.max(lat_d1)-0.5, 
           color = "dimgrey", fontsize = 10, fontweight='bold')
ax[0].text(s="D2", x = np.max(lon_d2)-1, y = np.max(lat_d2)+0.7, 
           color = "dimgrey", fontsize = 10, fontweight='bold')
ax[0].text(s="D3", x = np.max(lon_d3)-0.7, y = np.max(lat_d3)+0.7, 
           color = "dimgrey", fontsize = 10, fontweight='bold')
ax[0].set_title("Domain nesting in WRF (12 km - 3 km - 1 km)",
                color = "dimgrey", fontsize = 12)

ax[0].set_ylabel("Latitude", color = "dimgrey", fontsize = 10, labelpad = -15)
ax[0].set_yticks(np.asarray(ax[0].get_ylim()))
ax[0].set_yticklabels([str(np.around(np.asarray(ax[0].get_ylim())[0], decimals = 2)),
                       str(np.around(np.asarray(ax[0].get_ylim())[1], decimals = 2))],
                       color = "dimgrey", fontsize = 10)
ax[0].set_xlabel("Longitude", color = "dimgrey", fontsize = 10, labelpad=-10)
ax[0].set_xticks(np.asarray(ax[0].get_xlim()))
ax[0].set_xticklabels([str(np.around(np.asarray(ax[0].get_xlim())[0], decimals = 2)),
                       str(np.around(np.asarray(ax[0].get_xlim())[1], decimals = 2))],
                       color = "dimgrey", fontsize = 10)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='3%', pad=0.15, axes_class=plt.Axes)
fig.add_axes(cax)
cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
cbar.ax.set_title("Altitude [m]", loc='center', y = 1.04, color = 'dimgrey', fontsize = 10)

im2 = ax[1].pcolor(lon_d3, lat_d3, (LU_INDEX - 30)*urb_mask, cmap = cmap_lcz_d3, norm = norm_lcz_d3)
ax[1].pcolor(lon_d3, lat_d3, np.where(LU_INDEX == 17, 1, np.nan), color = cmap_t(0.02), zorder = -1)
ax[1].scatter(spat_at_MIDAS.Lon, 
              spat_at_MIDAS.Lat, 
              color = 'dimgrey', marker = 'o', linewidth = 2, 
              edgecolors = 'dimgrey', s=100, zorder = 2, alpha = 0.7,
              label = 'MIDAS AWS')
ax[1].scatter(spat_at_NA_sub_urb.Lon, 
              spat_at_NA_sub_urb.Lat, 
              color = 'darkgrey', marker = '.', s=30, zorder = 2, alpha = 0.5,
              label = 'Netatmo PWS')
ax[1].set_title("Official and personal weather stations over urban LCZ in Domain 3",
                color = "dimgrey", fontsize = 12)
ax[1].set_ylabel("Latitude", color = "dimgrey", fontsize = 10, labelpad = -15)
ax[1].set_yticks([np.min(lat_d3), np.max(lat_d3)])
ax[1].set_yticklabels([str(np.around(np.min(lat_d3), decimals = 2)),
                       str(np.around(np.max(lat_d3), decimals = 2))],
                       color = "dimgrey", fontsize = 10)
ax[1].set_xlabel("Longitude", color = "dimgrey", fontsize = 10, labelpad=-10)
ax[1].set_xticks([np.min(lon_d3), np.max(lon_d3)])
ax[1].set_xticklabels([str(np.around(np.min(lon_d3), decimals = 2)),
                       str(np.around(np.max(lon_d3), decimals = 2))],
                       color = "dimgrey", fontsize = 10)

divider = make_axes_locatable(ax[1])
cax2 = divider.append_axes('right', size='3%', pad=0.15, axes_class=plt.Axes)
fig.add_axes(cax2)
cbar_lcz = fig.colorbar(im2, cax=cax2, orientation='vertical', 
                        ticks = [2.5,4,5.5,7,8.5,9.5,10.5])
cbar_lcz.ax.tick_params(axis='y',which='both',left=False,right=False,labelright=True)
labels_lcz = cbar_lcz.ax.set_yticklabels(lcz_labels_d3, color = 'dimgrey', fontsize = 10)
cbar_lcz.ax.invert_yaxis()

### Create temporary objects for the legend
handles, labels = ax[1].get_legend_handles_labels()
fig.legend(handles, labels, loc = 'lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize = 10)      
fig.savefig(savedir + "Domain_nesting_CWS-AWS_W2W.png", dpi=300)

#%%
##########
### Figure 3 : MODEL PERFORMANCE AT EACH PWS
##########

### We have 5 metrics and 2 model outputs 
# col = 2; row = 5
col = 2; row = 3
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(row, col, figsize = (12,3.5*row), subplot_kw=dict(projection=proj),
                       constrained_layout=True)
ax = ax.flatten()

# list_stat = ["RMSE [°C]", "MAE [°C]", "MB [°C]", "Pearson r²", "Spearman r"]
# stat_max = [3.5, 2.8, 2, 0.8, 0.9]
# stat_min = [2.0, 1.8, -2, 0.6, 0.7]

list_stat = ["RMSE [°C]", "MB [°C]", "Pearson r²"]
stat_max = [3.5, 2, 0.8]
stat_min = [2.0, -2, 0.6]

### Loop over the model output dataset
m=0
for mod in Path(datadir_WRF).glob("WRF_Urb*_T2-V10-U10_20180525-20180831.nc"):
    if m == 0:
        l = np.arange(0,10,2)
    elif m == 1:
        l = np.arange(1,10,2)
    ds_wrf = xr.open_dataset(mod)
    ny_wrf, nx_wrf = np.shape(ds_wrf.T2[0,:,:])
    wrf_T2 = np.array(ds_wrf.T2.loc[startdate:enddate].values - 273.15)

    for NA_ID in spat_at_NA_sub_urb.index.values.astype(str):
        if NA_filt_df[NA_ID].count() == 0:
            continue
        else:
            tmp_NA = np.array(NA_filt_df[NA_ID])
            j = spat_at_NA_sub_urb.J.loc[NA_ID]
            i = spat_at_NA_sub_urb.I.loc[NA_ID]
    
            tmp_WRF = wrf_T2[:,j,i]
            ### Handle potential NaN values in measurements
            tmp_WRF = tmp_WRF[~np.isnan(tmp_NA)]
            tmp_NA = tmp_NA[~np.isnan(tmp_NA)]
            
            spat_at_NA_sub_urb.loc[NA_ID, "RMSE [°C]"] = np.sqrt(skm.mean_squared_error(tmp_NA, tmp_WRF)) 
            # spat_at_NA_sub_urb.loc[NA_ID, "MAE [°C]"] = skm.mean_absolute_error(tmp_NA, tmp_WRF)
            spat_at_NA_sub_urb.loc[NA_ID, "MB [°C]"] = np.mean(tmp_WRF - tmp_NA)
            spat_at_NA_sub_urb.loc[NA_ID, "Pearson r²"] = (stats.pearsonr(tmp_NA, tmp_WRF)[0])**2
            # spat_at_NA_sub_urb.loc[NA_ID, "Spearman r"] = stats.spearmanr(tmp_NA, tmp_WRF)[0]
            del tmp_WRF, tmp_NA
    stat_df = spat_at_NA_sub_urb[list_stat].columns
    ### print the average performance and the standard deviation
    print(mod, 
          spat_at_NA_sub_urb[list_stat].mean(),
          spat_at_NA_sub_urb[list_stat].std()
        )

    k = 0
    for st in stat_df:
        ax[l[k]].coastlines(resolution='10m', alpha=0.1)
        ax[l[k]].contour(lon_d3, lat_d3, lu_d3, levels=[30, 41], colors = 'black', linewidths = 0.1)
        ax[l[k]].set_extent([np.min(lon_d3), np.max(lon_d3), np.min(lat_d3), np.max(lat_d3)])
    
        ### Order the data based on the statistic scores
        if (st == "sp_r") | (st == "r2"):
            order_st = np.flip(np.argsort(spat_at_NA_sub_urb[st]))
        else:
            order_st = np.argsort(spat_at_NA_sub_urb[st])
        if st == 'MB [°C]':
            im = ax[l[k]].scatter(np.array(spat_at_NA_sub_urb.Lon)[order_st],
                               np.array(spat_at_NA_sub_urb.Lat)[order_st],
                               c = np.array(spat_at_NA_sub_urb[st])[order_st], 
                               cmap = cmap_mb, 
                               vmin = stat_min[k],
                               vmax = stat_max[k],
                               s=15, zorder = 1, alpha = 0.7, linewidths = 0)
        elif st in list_stat[-2:]:
            im = ax[l[k]].scatter(np.array(spat_at_NA_sub_urb.Lon)[order_st],
                               np.array(spat_at_NA_sub_urb.Lat)[order_st],
                               c = np.array(spat_at_NA_sub_urb[st])[order_st], 
                               cmap = cmap_stat, 
                               vmin = stat_min[k],
                               vmax = stat_max[k],
                               s=15, zorder = 1, alpha = 0.7, linewidths = 0)
        else:
            im = ax[l[k]].scatter(np.array(spat_at_NA_sub_urb.Lon)[order_st],
                               np.array(spat_at_NA_sub_urb.Lat)[order_st],
                               c = np.array(spat_at_NA_sub_urb[st])[order_st], 
                               cmap = cmap_stat_r, 
                               vmin = stat_min[k],
                               vmax = stat_max[k],
                               s=15, zorder = 1, alpha = 0.7, linewidths = 0)
        divider = make_axes_locatable(ax[l[k]])
        cax = divider.append_axes('right', size='3%', pad=0.15, axes_class=plt.Axes)
        fig.add_axes(cax)
        if l[0]==1:
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
            cbar.ax.set_title(st, loc='center', y = 1.04, color = 'dimgrey', fontsize = 16)
            cbar.ax.tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 14, bottom = False, left = False)
        else:
            cax.remove()
        k+=1
        
    spat_at_NA_sub_urb = spat_at_NA_sub_urb.drop(list_stat, 
                                                 axis=1)
    m+=1

ax[0].set_title("BouLac", color = 'dimgrey', fontsize = 24)
ax[1].set_title("YSU", color = 'dimgrey', fontsize = 24)

ax[-2].set_yticks([np.round(np.min(lat_d3), decimals = 2), np.round(np.max(lat_d3), decimals = 2)])
ax[-2].set_xticks([np.round(np.min(lon_d3), decimals = 2), np.round(np.max(lon_d3), decimals = 2)])
ax[-2].tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 14, bottom = False, left = False)

fig.savefig(savedir + 'Eval_WRF_Netatmo_2018-06-01_2018-08-31.png', dpi=300)
fig.savefig(savedir + 'Eval_WRF_Netatmo_2018-06-01_2018-08-31.pdf')

#%%
### FIGURE APPENDIX (MAE and Spearman r)

col = 2; row = 2
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(row, col, figsize = (12,3.5*row), subplot_kw=dict(projection=proj),
                       constrained_layout=True)
ax = ax.flatten()

list_stat = ["MAE [°C]", "Spearman r"]
stat_max = [2.8, 0.9]
stat_min = [1.8, 0.7]

### Loop over the model output dataset
m=0
for mod in Path(datadir_WRF).glob("WRF_Urb*_T2-V10-U10_20180525-20180831.nc"):
    if m == 0:
        l = np.arange(0,10,2)
    elif m == 1:
        l = np.arange(1,10,2)
    ds_wrf = xr.open_dataset(mod)
    ny_wrf, nx_wrf = np.shape(ds_wrf.T2[0,:,:])
    wrf_T2 = np.array(ds_wrf.T2.loc[startdate:enddate].values - 273.15)

    for NA_ID in spat_at_NA_sub_urb.index.values.astype(str):
        if NA_filt_df[NA_ID].count() == 0:
            continue
        else:
            tmp_NA = np.array(NA_filt_df[NA_ID])
            j = spat_at_NA_sub_urb.J.loc[NA_ID]
            i = spat_at_NA_sub_urb.I.loc[NA_ID]
    
            tmp_WRF = wrf_T2[:,j,i]
            ### Handle potential NaN values in measurements
            tmp_WRF = tmp_WRF[~np.isnan(tmp_NA)]
            tmp_NA = tmp_NA[~np.isnan(tmp_NA)]
            
            # spat_at_NA_sub_urb.loc[NA_ID, "RMSE [°C]"] = np.sqrt(skm.mean_squared_error(tmp_NA, tmp_WRF)) 
            spat_at_NA_sub_urb.loc[NA_ID, "MAE [°C]"] = skm.mean_absolute_error(tmp_NA, tmp_WRF)
            # spat_at_NA_sub_urb.loc[NA_ID, "MB [°C]"] = np.mean(tmp_WRF - tmp_NA)
            # spat_at_NA_sub_urb.loc[NA_ID, "Pearson r²"] = (stats.pearsonr(tmp_NA, tmp_WRF)[0])**2
            spat_at_NA_sub_urb.loc[NA_ID, "Spearman r"] = stats.spearmanr(tmp_NA, tmp_WRF)[0]
            del tmp_WRF, tmp_NA
    stat_df = spat_at_NA_sub_urb[list_stat].columns
    ### print the average performance and the standard deviation
    print(mod, 
          spat_at_NA_sub_urb[list_stat].mean(),
          spat_at_NA_sub_urb[list_stat].std()
        )

    k = 0
    for st in stat_df:
        ax[l[k]].coastlines(resolution='10m', alpha=0.1)
        ax[l[k]].contour(lon_d3, lat_d3, lu_d3, levels=[30, 41], colors = 'black', linewidths = 0.1)
        ax[l[k]].set_extent([np.min(lon_d3), np.max(lon_d3), np.min(lat_d3), np.max(lat_d3)])
    
        ### Order the data based on the statistic scores
        if (st == "sp_r") | (st == "r2"):
            order_st = np.flip(np.argsort(spat_at_NA_sub_urb[st]))
        else:
            order_st = np.argsort(spat_at_NA_sub_urb[st])
        if st == 'MB [°C]':
            im = ax[l[k]].scatter(np.array(spat_at_NA_sub_urb.Lon)[order_st],
                               np.array(spat_at_NA_sub_urb.Lat)[order_st],
                               c = np.array(spat_at_NA_sub_urb[st])[order_st], 
                               cmap = cmap_mb, 
                               vmin = stat_min[k],
                               vmax = stat_max[k],
                               s=15, zorder = 1, alpha = 0.7, linewidths = 0)
        elif st in list_stat[-2:]:
            im = ax[l[k]].scatter(np.array(spat_at_NA_sub_urb.Lon)[order_st],
                               np.array(spat_at_NA_sub_urb.Lat)[order_st],
                               c = np.array(spat_at_NA_sub_urb[st])[order_st], 
                               cmap = cmap_stat, 
                               vmin = stat_min[k],
                               vmax = stat_max[k],
                               s=15, zorder = 1, alpha = 0.7, linewidths = 0)
        else:
            im = ax[l[k]].scatter(np.array(spat_at_NA_sub_urb.Lon)[order_st],
                               np.array(spat_at_NA_sub_urb.Lat)[order_st],
                               c = np.array(spat_at_NA_sub_urb[st])[order_st], 
                               cmap = cmap_stat_r, 
                               vmin = stat_min[k],
                               vmax = stat_max[k],
                               s=15, zorder = 1, alpha = 0.7, linewidths = 0)
        divider = make_axes_locatable(ax[l[k]])
        cax = divider.append_axes('right', size='3%', pad=0.15, axes_class=plt.Axes)
        fig.add_axes(cax)
        if l[0]==1:
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
            cbar.ax.set_title(st, loc='center', y = 1.04, color = 'dimgrey', fontsize = 16)
            cbar.ax.tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 14, bottom = False, left = False)
        else:
            cax.remove()
        k+=1
        
    spat_at_NA_sub_urb = spat_at_NA_sub_urb.drop(list_stat, 
                                                 axis=1)
    m+=1

ax[0].set_title("BouLac", color = 'dimgrey', fontsize = 24)
ax[1].set_title("YSU", color = 'dimgrey', fontsize = 24)

ax[-2].set_yticks([np.round(np.min(lat_d3), decimals = 2), np.round(np.max(lat_d3), decimals = 2)])
ax[-2].set_xticks([np.round(np.min(lon_d3), decimals = 2), np.round(np.max(lon_d3), decimals = 2)])
ax[-2].tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 14, bottom = False, left = False)

fig.savefig(savedir + 'Eval_WRF_Netatmo_2018-06-01_2018-08-31_Supp.png', dpi=300)
fig.savefig(savedir + 'Eval_WRF_Netatmo_2018-06-01_2018-08-31_Supp.pdf')

#%%
##########
### Figure 4 : MODEL PERFORMANCES MATRICES
##########

### NEED TO RUN THE CODE FOR TABLE S3 BELOW BEFORE PLOTTING FIGURE 4
### THE CSV FILE READ HERE HAS BEEN MANUALLY COMPILED USING THE CSVs OUTPUTED IN TABLE S3

stats_df = pd.read_csv(savedir + 
                         "Model_Performance_Figure.csv", sep = ';')
stats_df = stats_df.rename(columns={"Unnamed: 0": "Stat", "Unnamed: 1": "BC_data"})

stat_max = [1.9, 1.5, 1.2, 0.2, 0.45]
stat_min = [0.9, 0.7, -1.2, 0, 0.15]
list_stat = ["RMSE [°C]", "MAE [°C]", "MB [°C]", "Pearson r²", "Spearman r"]

col = 1; row = 5
fig, ax = plt.subplots(row, col, figsize = (12,2.5*row))
ax = ax.flatten()
fig.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.90)

i = 0
for st in stats_df.Stat.unique():
    tmp_ar = np.array(stats_df[stats_df.Stat == st].drop(["Stat", "BC_data"], axis = 1).copy().values)
    if st == 'MB':
        im = ax[i].pcolormesh(tmp_ar, 
                              cmap = cmap_mb,
                              vmin = stat_min[i],
                              vmax = stat_max[i])
    elif st in list_stat[-2:]:
        im = ax[i].pcolormesh(tmp_ar, 
                              cmap = cmap_stat,
                              vmin = stat_min[i],
                              vmax = stat_max[i])
    else:
        im = ax[i].pcolormesh(tmp_ar, 
                              cmap = cmap_stat_r,
                              vmin = stat_min[i],
                              vmax = stat_max[i])
        
    divider = make_axes_locatable(ax[i])
    cax = divider.append_axes('right', size='2%', pad=0.15, axes_class=plt.Axes)
    fig.add_axes(cax)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
    cbar.ax.set_title(list_stat[i], loc='center', y = 1.06, color = 'dimgrey', fontsize = 12)
    i+=1
    
for ax_i in ax[:-1]:
    ax_i.tick_params(axis='both', bottom = False, left = False, right = False, top = False,
                     labelleft = False, labelbottom = False)
ax[-1].set_xticks(np.arange(0.5,13.5,1))
ax[-1].set_xticklabels(['WRF', 'RF$_{avg}$', 'RF$_{tstep}$',
                       'LinReg$_{avg}$', 'LinReg$_{tstep}$',
                       'Ridge$_{avg}$', 'Ridge$_{tstep}$',
                       'Lasso$_{avg}$', 'Lasso$_{tstep}$',
                       'GB$_{avg}$', 'GB$_{tstep}$',
                       'Dummy$_{avg}$', 'Dummy$_{tstep}$'],
                      rotation=45, ha='right', rotation_mode='anchor')
ax[-1].set_yticks(np.arange(0.5,6.5,1))
ax[-1].set_yticklabels(['BouLac$_{mean}$', 'YSU$_{mean}$',
                        'BouLac$_{min}$', 'YSU$_{min}$',
                        'BouLac$_{max}$', 'YSU$_{max}$'])
ax[-1].tick_params(axis='both', bottom = False, left = False, right = False, top = False,
                 labelleft = True, labelbottom = True, labelcolor = 'dimgrey', labelsize = 12)
ax[0].set_title("Model performance before and after bias-correction with different regressions", color = "dimgrey", fontsize = 16, y = 1.1)

fig.savefig(savedir + "Model_Perf_boot_x25_Stats.png", dpi=300)

#%%
##########
### Figure 5 : BIAS CORRECTION PER ML REGRESSOR (Min, Max, Mean / YSU, BouLac)
##########

list_mod_n = ["Random Forest", "Linear Regression", "Ridge Regression",
              "Lasso Regression", "Gradient Boosting", "Dummy Regression"]
for stat in list_daily_stat:
    ds_ysu = xr.open_dataset(savedir + 
                              "WRF_T2_Urb-BC_" + stat + "_" + startdate + "-" + enddate + "_" + city + ".nc")
    ds_bl = xr.open_dataset(savedir +
                            "WRF_BouLac_T2_Urb-BC_" + stat + "_" + startdate + "-" + enddate + "_" + city + ".nc")
    list_mod = list(ds_ysu.keys())[3::2]
    ### Maps for time step bias correction
    col = 3; row = 6
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(row, col, figsize = (8,12), subplot_kw=dict(projection=proj))
    ax = ax.flatten()
    fig.subplots_adjust(left=0.05, bottom=0.10, right=0.90, top=0.90)
    
    list_mod_tmp = list(ds_ysu.keys())[3::2]
    list_mod_tmp.append("ctl_run_nobc")
    
    min_temp = np.nanpercentile(
            (np.array(ds_ysu[list_mod_tmp].mean(dim='time').where(urb_mask == 1, np.nan).to_array()),
             np.array(ds_ysu[list_mod_tmp].mean(dim='time').where(urb_mask == 1, np.nan).to_array())
                                  ), q=5
        )
    max_temp = np.nanpercentile(
            (np.array(ds_ysu[list_mod_tmp].mean(dim='time').where(urb_mask == 1, np.nan).to_array()),
             np.array(ds_ysu[list_mod_tmp].mean(dim='time').where(urb_mask == 1, np.nan).to_array())
                                  ), q=95
        )
    
    del list_mod_tmp
    
    im = ax[0].pcolormesh(lon_d3, lat_d3, ds_ysu.ctl_run_nobc.mean(axis = 0)*urb_mask,
                  vmin = min_temp, vmax = max_temp, cmap = cmap_stat)
    ax[0].set_title("YSU", color = 'dimgrey')
    
    ax[9].pcolormesh(lon_d3, lat_d3, ds_bl.ctl_run_nobc.mean(axis = 0)*urb_mask,
                  vmin = min_temp, vmax = max_temp, cmap = cmap_stat)
    ax[9].set_title("BouLac", color = 'dimgrey')
    
    i = 3
    m = 0
    for mod in list_mod:
        im2 = ax[i].pcolormesh(lon_d3, lat_d3, 
                         ds_ysu[mod].mean(axis = 0)*urb_mask - ds_ysu.ctl_run_nobc.mean(axis = 0)*urb_mask,
                         vmin = -3, vmax = 3, cmap = cmap_bias)
        ax[i].set_title(list_mod_n[m], color = 'dimgrey')
        ax[i+9].pcolormesh(lon_d3, lat_d3, 
                           ds_bl[mod].mean(axis = 0)*urb_mask  - ds_bl.ctl_run_nobc.mean(axis = 0)*urb_mask,
                           vmin = -3, vmax = 3, cmap = cmap_bias)
        ax[i+9].set_title(list_mod_n[m], color = 'dimgrey')

        i+=1
        m+=1
    
    for ax_i in ax:
        ax_i.tick_params(axis='both', bottom = False, left = False, right = False, top = False,
                         labelleft = False, labelbottom = False)
        ax_i.pcolor(lon_d3, lat_d3, np.where(LU_INDEX == 17, 1, np.nan), color = cmap_t(0.02), zorder = 2)
    
    for ax_i in ax[[1,2,10,11]]:
        ax_i.remove()
    
    cax_dif = plt.axes([0.31, 0.39, 0.01, 0.08])
    cbar = fig.colorbar(im, cax=cax_dif, orientation='vertical', extend = 'both')
    cbar.ax.set_title(r'$T2_{'+stat+'}$ [°C]', loc='left', y = 1.06, color = 'dimgrey', fontsize = 12)
    cbar.ax.tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 14)
    
    cax_dif = plt.axes([0.91, 0.115, 0.01, 0.08])
    cbar = fig.colorbar(im2, cax=cax_dif, orientation='vertical', extend = 'both')
    cbar.ax.set_title(r'$\Delta$T2 [°C]', loc='left', y = 1.06, color = 'dimgrey', fontsize = 12)
    cbar.ax.tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 14)
    
    fig.suptitle("Modelled temperatures and respective bias-corrections with multiple regressors",
                 y = 0.94, color = 'dimgrey')
    fig.savefig(savedir + "WRF_YSU-BouLac_" + stat + ".png", dpi=300)

#%%
##########
### Figure 6 : BIAS CORRECTED PRODUCT PER ML REGRESSOR (YSU, BouLac, Netatmo predicted)
##########

tmp_ds_ysu = xr.open_dataset(savedir + 
                         "WRF_T2_Urb-BC_mean_" + startdate + "-" + enddate + "_" + city + ".nc")
list_mod = list(tmp_ds_ysu.keys())[3::2]
del tmp_ds_ysu

list_daily_stat = ["mean", "max", "min"]
list_mod_n = ["Random Forest", "Linear Regression", "Ridge Regression",
              "Lasso Regression", "Gradient Boosting", "Dummy Regression"]

m=0
for mod_tstep in list_mod:
    ### Maps for time step bias correction
    col = 5; row = 3
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(row, col, figsize = (12,4.5), subplot_kw=dict(projection=proj))
    ax = ax.flatten()
    fig.subplots_adjust(left=0.10, bottom=0.10, right=0.90, top=0.90)
    
    i=0
    for stat in list_daily_stat:
        ds_ysu = xr.open_dataset(savedir + 
                                  "WRF_T2_Urb-BC_" + stat + "_" + startdate + "-" + enddate + "_" + city + ".nc")
        ds_bl = xr.open_dataset(savedir +
                                "WRF_BouLac_T2_Urb-BC_" + stat + "_" + startdate + "-" + enddate + "_" + city + ".nc")
        ds_na = xr.open_dataset(savedir +
                                "NA_Pred_T2_WRFgrid_" + stat + "_" + startdate + "-" + enddate + "_" + city + ".nc")
        
        min_temp = np.nanpercentile(
                (ds_na[mod_tstep].mean(axis = 0)*urb_mask,
                  ds_ysu[mod_tstep].mean(axis = 0)*urb_mask,
                  ds_bl[mod_tstep].mean(axis = 0)*urb_mask
                                      ), q=5
            )
        max_temp = np.nanpercentile(
                (ds_na[mod_tstep].mean(axis = 0)*urb_mask,
                  ds_ysu[mod_tstep].mean(axis = 0)*urb_mask,
                  ds_bl[mod_tstep].mean(axis = 0)*urb_mask
                                      ), q=95
            )
        
        if i==0:
            im = ax[i].pcolormesh(lon_d3, lat_d3, ds_na[mod_tstep].mean(axis = 0)*urb_mask,
                          vmin = min_temp, vmax = max_temp, cmap = cmap_stat)
        elif i==5:
            im2 = ax[i].pcolormesh(lon_d3, lat_d3, ds_na[mod_tstep].mean(axis = 0)*urb_mask,
                          vmin = min_temp, vmax = max_temp, cmap = cmap_stat)
        elif i==10:
            im3 = ax[i].pcolormesh(lon_d3, lat_d3, ds_na[mod_tstep].mean(axis = 0)*urb_mask,
                          vmin = min_temp, vmax = max_temp, cmap = cmap_stat)
        ax[i+1].pcolormesh(lon_d3, lat_d3, ds_ysu[mod_tstep].mean(axis = 0)*urb_mask,
                          vmin = min_temp, vmax = max_temp, cmap = cmap_stat)
        ax[i+2].pcolormesh(lon_d3, lat_d3, 
                            ds_ysu[mod_tstep].mean(axis = 0)*urb_mask - ds_na[mod_tstep].mean(axis = 0)*urb_mask,
                            vmin = -2, vmax = 2, cmap = cmap_bias)
        
        ax[i+3].pcolormesh(lon_d3, lat_d3, ds_bl[mod_tstep].mean(axis = 0)*urb_mask,
                          vmin = min_temp, vmax = max_temp, cmap = cmap_stat)
        im4 = ax[i+4].pcolormesh(lon_d3, lat_d3, 
                            ds_bl[mod_tstep].mean(axis = 0)*urb_mask - ds_na[mod_tstep].mean(axis = 0)*urb_mask,
                            vmin = -2, vmax = 2, cmap = cmap_bias)    
        i+=5
    
    i=0
    for ax_i in ax:
        ax_i.tick_params(axis='both', bottom = False, left = False, right = False, top = False,
                         labelleft = False, labelbottom = False)
        ax_i.pcolor(lon_d3, lat_d3, np.where(LU_INDEX == 17, 1, np.nan), color = cmap_t(0.02), zorder = 2)
        divider = make_axes_locatable(ax_i)
        if i==0:
            cax = divider.append_axes('left', size='3%', pad=0.15, axes_class=plt.Axes)
            fig.add_axes(cax)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical', extend = 'both')
            cbar.ax.set_title(r"$T2_{"+list_daily_stat[0]+"}$ [°C]", 
                              loc='left', y = 1.04, color = 'dimgrey', fontsize = 8)
            cbar.ax.tick_params(axis='y',which='both',left=True,right=False,labelright=False,labelleft=True)
        elif i==5:
            cax = divider.append_axes('left', size='3%', pad=0.15, axes_class=plt.Axes)
            fig.add_axes(cax)
            cbar = fig.colorbar(im2, cax=cax, orientation='vertical', extend = 'both')
            cbar.ax.set_title(r"$T2_{"+list_daily_stat[1]+"}$ [°C]", 
                              loc='left', y = 1.04, color = 'dimgrey', fontsize = 8)
            cbar.ax.tick_params(axis='y',which='both',left=True,right=False,labelright=False,labelleft=True)
        elif i==10:
            cax = divider.append_axes('left', size='3%', pad=0.15, axes_class=plt.Axes)
            fig.add_axes(cax)
            cbar = fig.colorbar(im3, cax=cax, orientation='vertical', extend = 'both')
            cbar.ax.set_title(r"$T2_{"+list_daily_stat[2]+"}$ [°C]", 
                              loc='left', y = 1.04, color = 'dimgrey', fontsize = 8)
            cbar.ax.tick_params(axis='y',which='both',left=True,right=False,labelright=False,labelleft=True)
        elif i==(len(ax)-1):
            cax = divider.append_axes('left', size='3%', pad=0.15, axes_class=plt.Axes)
            fig.add_axes(cax)
            cax.remove()
            cax_dif = plt.axes([0.91, 0.13, 0.004, 0.17])
            cbar = fig.colorbar(im4, cax=cax_dif, orientation='vertical', extend = 'both')
            cbar.ax.set_title(r'$\Delta$T2 [°C]', loc='right', y = 1.04, color = 'dimgrey', fontsize = 8)
        else:
            cax = divider.append_axes('left', size='3%', pad=0.15, axes_class=plt.Axes)
            fig.add_axes(cax)
            cax.remove()
        i+=1

    list_plot = ["PWS", "YSU", "YSU - PWS", "BouLac", "BouLac - PWS"]
    i=0
    for ax_i in ax[0:5]:
        ax_i.set_title(list_plot[i], color = "dimgrey", fontsize = 11)
        i+=1
    
    ax[-5].set_yticks([np.round(np.min(lat_d3), decimals = 2), np.round(np.max(lat_d3), decimals = 2)])
    ax[-5].set_xticks([np.round(np.min(lon_d3), decimals = 2), np.round(np.max(lon_d3), decimals = 2)])
    ax[-5].tick_params(axis = "both", labelcolor = 'dimgrey', labelsize = 8, bottom = False, left = False)
    
    fig.suptitle("Bias-correction of WRF BEP-BEM simulations compared to predicted temperatures with " + list_mod_n[m], 
                 y = 0.98,
                 color = "dimgrey", fontsize = 12)
    fig.savefig(savedir + "WRF_YSU-BouLac_vs_NApred_" + mod_tstep[:-6] + ".png", dpi=300)
    
    m+=1

#%%
##########
### Figure S3.1 to S3.3 : MODEL BOOTSTRAPPING 25x DAILY PERFORMANCE
##########

### This is the plot for the averaged daily temperature across all stations
list_daily_stat = ["mean", "max", "min"]
for stat in list_daily_stat:
    ds_wrf_bc_boot_ysu = xr.open_dataset(savedir + 
                                         'T2_Urb-BC_daily_' + stat + '_bootstrap_x25_' + city + '.nc')
    ds_wrf_bc_boot_bl = xr.open_dataset(savedir + 
                                     'T2_BouLac_Urb-BC_daily_' + stat + '_bootstrap_x25_' + city + '.nc')
    
    ax_lim_min = np.floor(np.min([ds_wrf_bc_boot_ysu.min().to_array(), ds_wrf_bc_boot_bl.min().to_array()]))
    ax_lim_max = np.ceil(np.max([ds_wrf_bc_boot_ysu.max().to_array(), ds_wrf_bc_boot_bl.max().to_array()]))
    
    fig, ax = plt.subplots(2, 3, figsize = (12,10))
    ax = ax.flatten()
    j=0
    for bc_mod in list(ds_wrf_bc_boot_ysu.keys())[2::]:
        print(bc_mod)
        
        if 'avg' in bc_mod:
            ### YSU
            ### Raw output
            ax[j].scatter(ds_wrf_bc_boot_ysu.obs, ds_wrf_bc_boot_ysu.wrf, color = 'paleturquoise',
                        label = 'WRF YSU output', alpha = 0.5)
            m, b = np.polyfit(ds_wrf_bc_boot_ysu.obs, ds_wrf_bc_boot_ysu.wrf, 1)
            ax[j].plot(np.arange(ax_lim_min,ax_lim_max + 1), m*np.arange(ax_lim_min,ax_lim_max + 1) + b, 
                      color = 'paleturquoise', linestyle = '--')
            
            ### Average bias correction
            ax[j].scatter(ds_wrf_bc_boot_ysu.obs, ds_wrf_bc_boot_ysu[bc_mod], color = 'lightseagreen',
                        label = 'WRF YSU time-mean correction', alpha = 0.5, marker = "s")
            m, b = np.polyfit(ds_wrf_bc_boot_ysu.obs, ds_wrf_bc_boot_ysu[bc_mod], 1)
            ax[j].plot(np.arange(ax_lim_min,ax_lim_max + 1), m*np.arange(ax_lim_min,ax_lim_max + 1) + b, 
                      color = 'lightseagreen', linestyle = '-.')
            ax[j].set_title(bc_mod[:-4])
            
            ### BOULAC
            ### Raw output
            ax[j].scatter(ds_wrf_bc_boot_bl.obs, ds_wrf_bc_boot_bl.wrf, color = 'mediumorchid',
                        label = 'WRF BouLac output', alpha = 0.5)
            m, b = np.polyfit(ds_wrf_bc_boot_bl.obs, ds_wrf_bc_boot_bl.wrf, 1)
            ax[j].plot(np.arange(ax_lim_min,ax_lim_max + 1), m*np.arange(ax_lim_min,ax_lim_max + 1) + b, 
                      color = 'mediumorchid', linestyle = '--')
            
            ### Average bias correction
            ax[j].scatter(ds_wrf_bc_boot_bl.obs, ds_wrf_bc_boot_bl[bc_mod], color = 'blueviolet',
                        label = 'WRF BouLac time-mean correction', alpha = 0.5, marker = "s")
            m, b = np.polyfit(ds_wrf_bc_boot_bl.obs, ds_wrf_bc_boot_bl[bc_mod], 1)
            ax[j].plot(np.arange(ax_lim_min,ax_lim_max + 1), m*np.arange(ax_lim_min,ax_lim_max + 1) + b, 
                      color = 'blueviolet', linestyle = '-.')
            
        if 'tstep' in bc_mod:
            ### YSU
            ### Timestep daily bias correction
            ax[j].scatter(ds_wrf_bc_boot_ysu.obs, ds_wrf_bc_boot_ysu[bc_mod], color = 'darkcyan',
                        label = 'WRF YSU time-step correction', alpha = 0.5, marker = "*")
            m, b = np.polyfit(ds_wrf_bc_boot_ysu.obs, ds_wrf_bc_boot_ysu[bc_mod], 1)
            ax[j].plot(np.arange(ax_lim_min,ax_lim_max + 1), m*np.arange(ax_lim_min,ax_lim_max + 1) + b, 
                      color = 'darkcyan', linestyle = ':')
            
            ### BOULAC
            ### Timestep daily bias correction
            ax[j].scatter(ds_wrf_bc_boot_bl.obs, ds_wrf_bc_boot_bl[bc_mod], color = 'purple',
                        label = 'WRF BouLac time-step correction', alpha = 0.5, marker = "*")
            m, b = np.polyfit(ds_wrf_bc_boot_bl.obs, ds_wrf_bc_boot_bl[bc_mod], 1)
            ax[j].plot(np.arange(ax_lim_min,ax_lim_max + 1), m*np.arange(ax_lim_min,ax_lim_max + 1) + b, 
                      color = 'purple', linestyle = ':')
            
            ### Obs == Mod isoline
            ax[j].set_ylim(ax_lim_min,ax_lim_max)
            ax[j].set_xlim(ax_lim_min,ax_lim_max)
            ax[j].plot(np.arange(ax_lim_min,ax_lim_max + 1), 
                      np.arange(ax_lim_min,ax_lim_max + 1), color = 'k')
            j+=1
    ax[3].set_ylabel('Modelled')      
    ax[3].set_xlabel('Observed')

    ### Create temporary objects for the legend
    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, 'lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize = 10)      
    fig.suptitle("Average model's bias correction of daily " + stat + " temperature after 25 bootstrap", y = 0.98)
    fig.savefig(savedir + "bootstrapx25_daily_avg_bc_" + stat + ".png", dpi=300)
    del fig, ds_wrf_bc_boot_bl, ds_wrf_bc_boot_ysu

#%%
#########################
###  TABLES BC PAPER  ###
#########################

### NEED TO BE RUN BEFORE RUNNING FIGURE 4
### TABLE S3.1: EVALUATE THE MODEL FIT USING AVERAGE DAILY TEMPERATURE BIAS CORRECTION AT EACH CWS LOC

list_daily_stat = ["mean", "max", "min"]
for stat in list_daily_stat:
    ds_wrf_bc_boot_bl = xr.open_dataset(savedir + 'T2_BouLac_Urb-BC_' + stat + '_bootstrap_x25_' + city + '.nc')
    ds_wrf_bc_boot_ysu = xr.open_dataset(savedir + 'T2_Urb-BC_' + stat + '_bootstrap_x25_' + city + '.nc')
    
    ds_wrf_bc_boot_bl = ds_wrf_bc_boot_bl.dropna(dim='ID')
    ds_wrf_bc_boot_ysu = ds_wrf_bc_boot_ysu.dropna(dim='ID')


    df_stat_bl = pd.DataFrame(columns = ["RMSE", "MAE", "MB", "r2", "r_s"])
    df_stat_ysu = pd.DataFrame(columns = ["RMSE", "MAE", "MB", "r2", "r_s"])
    for bc_mod in list(ds_wrf_bc_boot_ysu.keys())[1::]: 
        RMSE = np.sqrt(skm.mean_squared_error(ds_wrf_bc_boot_ysu.obs, 
                                      ds_wrf_bc_boot_ysu[bc_mod])) 
        MAE = skm.mean_absolute_error(ds_wrf_bc_boot_ysu.obs, 
                                      ds_wrf_bc_boot_ysu[bc_mod])
        MB = np.mean(ds_wrf_bc_boot_ysu[bc_mod] - ds_wrf_bc_boot_ysu.obs 
                                      ).values
        r_2 = stats.pearsonr(ds_wrf_bc_boot_ysu.obs, 
                                      ds_wrf_bc_boot_ysu[bc_mod])[0]**2
        r_spear = stats.spearmanr(ds_wrf_bc_boot_ysu.obs, 
                                      ds_wrf_bc_boot_ysu[bc_mod])[0]
        tmp_ysu = pd.DataFrame({'RMSE':[RMSE], 'MAE':[MAE], 'MB':[MB], 
                               'r2':[r_2], 'r_s':[r_spear]})
        
        df_stat_ysu = pd.concat((df_stat_ysu, tmp_ysu), 
                            axis = 0,
                            ignore_index = True)
        
        RMSE = np.sqrt(skm.mean_squared_error(ds_wrf_bc_boot_bl.obs, 
                                      ds_wrf_bc_boot_bl[bc_mod])) 
        MAE = skm.mean_absolute_error(ds_wrf_bc_boot_bl.obs, 
                                      ds_wrf_bc_boot_bl[bc_mod])
        MB = np.mean(ds_wrf_bc_boot_bl[bc_mod] - ds_wrf_bc_boot_bl.obs 
                                      ).values
        r_2 = stats.pearsonr(ds_wrf_bc_boot_bl.obs, 
                                      ds_wrf_bc_boot_bl[bc_mod])[0]**2
        r_spear = stats.spearmanr(ds_wrf_bc_boot_bl.obs, 
                                      ds_wrf_bc_boot_bl[bc_mod])[0]
        tmp_bl = pd.DataFrame({'RMSE':[RMSE], 'MAE':[MAE], 'MB':[MB], 
                               'r2':[r_2], 'r_s':[r_spear]})
        
        df_stat_bl = pd.concat((df_stat_bl, tmp_bl), 
                            axis = 0,
                            ignore_index = True)
        
        del tmp_bl, tmp_ysu
        
    df_stat_ysu.index = list(ds_wrf_bc_boot_ysu.keys())[1::]
    df_stat_bl.index = list(ds_wrf_bc_boot_bl.keys())[1::]
    
    df_stat_ysu = df_stat_ysu.astype(float).round(decimals = 2)
    df_stat_bl = df_stat_bl.astype(float).round(decimals = 2)
    
    df_stat_bl.to_csv(savedir + 'BouLac_Perf_CWS_bootx25_' + stat +'.csv')
    df_stat_ysu.to_csv(savedir + 'YSU_Perf_CWS_bootx25_' + stat +'.csv')

        
# %%
