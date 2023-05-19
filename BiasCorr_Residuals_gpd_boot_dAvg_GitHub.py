# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:58:34 2022

@author: Oscar Brousse and Charles Simpson
@institute: IEDE at UCL
"""

import glob
from pathlib import Path
import ast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.sample
import xarray as xr
from rasterio.warp import Resampling, reproject
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split


##########################
### MISCELLANEOUS INFO ###
##########################

city = 'London'
datadir_WRF = r''
datadir_WRF_geog = r''
savedir = r''
datadir_NA = r''
datadir_lcz = r''


startdate = '2018-06-01'
enddate = '2018-08-31'
resample_starth = '2018-05-31 09:00:00'
dates_list = [d.strftime('%Y-%m-%d') for d in pd.date_range(startdate, enddate, freq='1d').to_list()]

boulac = True

#########################
### MODEL SIMULATIONS ###
#########################

geogrid_d03 = datadir_WRF_geog + 'geo_em.d03_LCZ_params.nc'
surf_info_d03 = xr.open_dataset(geogrid_d03)

WRF_grid = datadir_WRF_geog + 'geo_em.d03_gridinfo.tif'
WRF_tif = xr.open_rasterio(WRF_grid)

if boulac == True:
    ds_wrf = xr.open_dataset(datadir_WRF + "WRF_Urb_BouLac_T2-V10-U10_20180525-20180831.nc")
else:
    ds_wrf = xr.open_dataset(datadir_WRF + "WRF_Urb_T2-V10-U10_20180525-20180831.nc")
ny_wrf, nx_wrf = np.shape(ds_wrf.T2[0,:,:])

df_wrf = ds_wrf.to_dataframe()
df_wrf = df_wrf.drop(columns=["U10", "V10"])
df_wrf.columns
df_wrf.T2 = df_wrf.T2 - 273.15

df_wrf_filt = pd.concat(df_wrf.loc[date].dropna(thresh=20, axis=1)
                        for date in dates_list
                       )

### We must import a shapefile with the WRF grid polygons otherwise GDF would only create points
### with coordinates equal to the center of the grid point

gdf_wrf = gpd.read_file(
    datadir_WRF + "wrf_voronoi/wrf_voronoi.shp"
)
gdf_wrf= gdf_wrf.rename(columns = {'south_nort':'south_north'})

## Since we run the regressors on a daily timestep we do not resample prior to joining
## the temperature to the GDF grid info. This is done later within a loop
gdf_wrf= gdf_wrf.set_index(["west_east", "south_north"]).join(ds_wrf[["XLAT","XLONG"]].to_dataframe())

lat_d3=surf_info_d03.XLAT_M.squeeze().values
lon_d3=surf_info_d03.XLONG_M.squeeze().values


LU_INDEX = surf_info_d03.LU_INDEX.values.squeeze()

### urban mask to compute the residuals bias correction only in urban areas with CWS
urb_mask = np.where(LU_INDEX < 30, 0, 1)

### Dictionnary of UCP in WRF (from python W2W (see Demuzere et al. (2021)))
ucp_dict = {
       'LP_URB2D'  : 90,
       'MH_URB2D'  : 91,
       'STDH_URB2D': 92,
       'HGT_URB2D' : 93,
       'LB_URB2D'  : 94,
       'LF_URB2D'  : 95,   # 97, 98, 99, for all 4 directions --> Can only extract one if perpedicular E/N orientation
       'HI_URB2D'  : 117,  # Goes on until index 132
   }

#########################
### CWS OBSERVATIONS  ###
#########################

### General attributes per station
datadir_NA = r''
df_netatmo_id = pd.read_csv(
    datadir_NA + "List_Netatmo_stations_London_WRF_2015-2020.csv"
)

gdf_netatmo_id = gpd.GeoDataFrame(
    df_netatmo_id,
    geometry=gpd.points_from_xy(df_netatmo_id.Lon, df_netatmo_id.Lat),
    crs="EPSG:4326",
)

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

##########################
### EARTH OBSERVATIONS ###
##########################

lcz_tif_path = datadir_lcz + 'WUDAPT_EU_LCZ_London_BD.tif'

### Get the LCZ map
lcz_data = xr.open_rasterio(lcz_tif_path)
### Need to reverse Y axis
lcz_data = lcz_data.reindex(y=lcz_data.y[::-1])
lcz_data = lcz_data.loc[1,:,:]
lcz_data = lcz_data.where(lcz_data != 0, 17)

### Create a matrix that counts how many LCZ pixels there are in each WRF pixel

lcz_ones = np.ones(shape=np.shape(lcz_data[:,:]))
WRF_lcz_c = np.empty(shape=np.shape(WRF_tif[0,:,:]))
with rasterio.Env():
    lcz_count_wrf = reproject(lcz_ones,
                              WRF_lcz_c,
                              src_transform=lcz_data.attrs['transform'],
                              src_crs=lcz_data.attrs['crs'],
                              dst_transform=WRF_tif.attrs['transform'],
                              dst_crs=WRF_tif.attrs['crs'],
                              src_nodata = np.nan,
                              dst_nodata = np.nan,
                              resampling=Resampling.sum)[0]

### Dictionnary of each LCZ proportion and its value in the raster file
lcz_dict = {1 : 'LCZ_1_p',
            2 : 'LCZ_2_p',
            3 : 'LCZ_3_p',
            4 : 'LCZ_4_p',
            5 : 'LCZ_5_p',
            6 : 'LCZ_6_p',
            7 : 'LCZ_7_p',
            8 : 'LCZ_8_p',
            9 : 'LCZ_9_p',
            10 : 'LCZ_10_p',
            11 : 'LCZ_11_p',
            12 : 'LCZ_12_p',
            13 : 'LCZ_13_p',
            14 : 'LCZ_14_p',
            15 : 'LCZ_15_p',
            16 : 'LCZ_16_p',
            17 : 'LCZ_17_p' 
            }


ds_LCZp = xr.Dataset({'lu_index': (['latitude', 'longitude'], LU_INDEX),
                      'urb_frac': (['latitude', 'longitude'], surf_info_d03.FRC_URB2D.squeeze().values),
                      'surf_hgt': (['latitude', 'longitude'], surf_info_d03.HGT_M.squeeze().values),                      
                      'bld_hgt': (['latitude', 'longitude'], surf_info_d03.URB_PARAM.squeeze()[ucp_dict.get('HGT_URB2D'),:,:].values),
                      'lambda_p': (['latitude', 'longitude'], surf_info_d03.URB_PARAM.squeeze()[ucp_dict.get('LP_URB2D'),:,:].values),
                      'lambda_f': (['latitude', 'longitude'], surf_info_d03.URB_PARAM.squeeze()[ucp_dict.get('LF_URB2D'),:,:].values),
                      'lambda_b': (['latitude', 'longitude'], surf_info_d03.URB_PARAM.squeeze()[ucp_dict.get('LB_URB2D'),:,:].values)
                      },
                     coords={'y': (('latitude', 'longitude'), lat_d3),
                             'x': (('latitude', 'longitude'), lon_d3)}
    )

for lczi in lcz_dict.keys():
    if np.sum(np.where(lcz_data == lczi, 1, 0)) < 1000:
        continue
    lcz_ones_tmp = np.ones(shape=np.shape(lcz_data[:,:]))
    lcz_ones_tmp = np.where(lcz_data == lczi, lcz_ones_tmp, 0)
    WRF_lcz_c_tmp = np.empty(shape=np.shape(WRF_tif[0,:,:]))
    with rasterio.Env():
        lcz_count_wrf_tmp = reproject(lcz_ones_tmp,
                                      WRF_lcz_c_tmp,
                                      src_transform=lcz_data.attrs['transform'],
                                      src_crs=lcz_data.attrs['crs'],
                                      dst_transform=WRF_tif.attrs['transform'],
                                      dst_crs=WRF_tif.attrs['crs'],
                                      src_nodata = np.nan,
                                      dst_nodata = np.nan,
                                      resampling=Resampling.sum)[0]
    
    ds_LCZp[lcz_dict.get(lczi)] = (('latitude', 'longitude'), np.flip(lcz_count_wrf_tmp / lcz_count_wrf, axis = 0))
    del lcz_ones_tmp, WRF_lcz_c_tmp, lcz_count_wrf_tmp

### THE PROPORTION OF LCZ IS NOT USED IN THIS STUDY BUT COULD BE USED USING THESE COVARIATES
### In WRF, urban LCZ are assigned a value above 30 and up to 41 (LCZ 1 = 31, LCZ 2 = 32...)
for lcz_dummy in np.unique(LU_INDEX[LU_INDEX>30]):
    ds_LCZp['LCZ_' + str(int(lcz_dummy - 30)) + '_m'] = (('latitude', 'longitude'), np.where(LU_INDEX == lcz_dummy, 1, 0))


#########################
###   WRF CWS PARAMS  ###
#########################


### Link CWS to dataset of predictors
gdf_LU_joined = gdf_wrf.set_index(["XLAT", "XLONG"]).join(
    ds_LCZp.to_dataframe()
    .rename(columns={"x": "XLONG", "y": "XLAT"})
    .set_index(["XLAT", "XLONG"])
)

gdf_join = gdf_LU_joined.sjoin(gdf_netatmo_id, predicate="contains")

#########################
###  BIAS CORRECTION  ###
#########################

### We bias correct daily average, minimum and maximum air temperature at 2 meters in urban areas only
list_daily_stat = ["mean", "max", "min"]

### The hypertunning is done in BestMod_Residuals_BiasCorr.py
### Load best parameters for each Machine Learning regressors
df_params = pd.read_csv(savedir + 'GridSearchCV_BestModelParams.csv')

### We will bootstrap 25 times the models against using 80/20 train/test ratio
### This will be performed for both time-mean and time-step bias correction

### Number of bootstrapping to test the models and fraction of data used as testing
bootstrap_n = 25
samp_frac = 0.20

for stat in list_daily_stat:
    print(stat)   
    ### Need to reload the models' list it each time otherwise raises ValueError
    ### There could be scope for this package in R: https://cran.r-project.org/web/packages/ranger/ranger.pdf
    list_models = [RandomForestRegressor(), LinearRegression(), Ridge(), Lasso(), GradientBoostingRegressor(),
                   DummyRegressor()]
    
    if stat == "mean":
        ### For the averaged value we take the daily averaged temperature at date d
        df_na_d = df_na_filt[startdate:enddate].resample('24h').mean()
        T2_WRF_mean = (ds_wrf.loc[{'XTIME':slice(startdate,enddate)}].T2.resample(XTIME="1D").mean() - 273.15).values
        da_T2_WRF = xr.DataArray(T2_WRF_mean, 
                                 dims=['XTIME', 'south_north', 'west_east'],
                                 coords = {'XTIME': pd.DatetimeIndex(pd.date_range(startdate, enddate, freq='1d')),
                                           'lat': (('south_north', 'west_east'), lat_d3),
                                            'lon': (('south_north', 'west_east'), lon_d3)})
        df_wrf_d = da_T2_WRF.to_dataframe(name = "T2")
    elif stat == "max":
        df_na_d = df_na_filt.resample('24h', origin=resample_starth).max()
        ### The resampling method applied here follows Had-UK. We keep dates d - d+1
        df_na_d = df_na_d.loc[startdate:enddate].set_index(pd.DatetimeIndex(dates_list))

        
        ## Calculate the maximum air temperature following Hollis et al. (2019)
        ## (Minimum from 9:00 UTC D-1 to 9:00 UTC D; Maximum from 9:00 UTC D to 9:00 UTC D + 1)
        for d_i in range(len(dates_list)):
            print(dates_list[d_i])
            if d_i == len(dates_list) - 1:
                timeslice_max = slice(dates_list[d_i] + 'T09:00:00',
                                      dates_list[d_i] + 'T23:59:00')
            else:
                timeslice_max = slice(dates_list[d_i] + 'T09:00:00',
                                  dates_list[d_i + 1] + 'T09:00:00')
            if d_i == 0:
                T2_WRF_max = np.max(ds_wrf.loc[dict(XTIME=timeslice_max)].T2.values, axis=0) - 273.15
            else:
                T2_WRF_max = np.dstack((T2_WRF_max,
                                       np.max(ds_wrf.loc[dict(XTIME=timeslice_max)].T2.values, axis=0) - 273.15))
        
        T2_WRF_max = np.moveaxis(T2_WRF_max, -1, 0)
        da_T2_WRF = xr.DataArray(T2_WRF_max, dims=['XTIME', 'south_north', 'west_east'],
                                    coords = {'XTIME': pd.DatetimeIndex(pd.date_range(startdate, enddate, freq='1d')),
                                              'lat': (('south_north', 'west_east'), lat_d3),
                                              'lon': (('south_north', 'west_east'), lon_d3)})
        df_wrf_d = da_T2_WRF.to_dataframe(name = "T2")

    elif stat == "min":
        df_na_d = df_na_filt[startdate:enddate].resample('24h', origin=resample_starth).min()
        ### The resampling method applied here follows Had-UK. We keep dates d-1 - d
        df_na_d = df_na_d.loc[resample_starth:dates_list[-2]].set_index(pd.DatetimeIndex(dates_list))
        ## Calculate the minimum air temperature following Hollis et al. (2019)
        ## (Minimum from 9:00 UTC D-1 to 9:00 UTC D; Maximum from 9:00 UTC D to 9:00 UTC D + 1)
        
        for d_i in range(len(dates_list)):
            print(dates_list[d_i])
            if d_i == 0:
                timeslice_min = slice(dates_list[d_i] + 'T00:00:00',
                                      dates_list[d_i] + 'T8:59:00')
            else:
                timeslice_min = slice(dates_list[d_i -1] + 'T09:00:00',
                                      dates_list[d_i] + 'T09:00:00')
            if d_i == 0:
                T2_WRF_min = np.min(ds_wrf.loc[dict(XTIME=timeslice_min)].T2.values, axis=0) - 273.15
            else:
                T2_WRF_min = np.dstack((T2_WRF_min,
                                       np.min(ds_wrf.loc[dict(XTIME=timeslice_min)].T2.values, axis=0) - 273.15))
        
        T2_WRF_min = np.moveaxis(T2_WRF_min, -1, 0)
        da_T2_WRF = xr.DataArray(T2_WRF_min, dims=['XTIME', 'south_north', 'west_east'],
                                    coords = {'XTIME': pd.DatetimeIndex(pd.date_range(startdate, enddate, freq='1d')),
                                              'lat': (('south_north', 'west_east'), lat_d3),
                                              'lon': (('south_north', 'west_east'), lon_d3)})
        df_wrf_d = da_T2_WRF.to_dataframe(name = "T2")

    
    gdf_wrf_avg = gdf_wrf.join(df_wrf_d.groupby([pd.Grouper(level='south_north'), 
                                               pd.Grouper(level='west_east')]
                                              ).T2.apply(np.nanmean).rename("t2_wrf")).set_index(
                                                  ["XLAT", "XLONG"]).copy()
    gdf_join_avg = gdf_join.join(gdf_wrf_avg.t2_wrf).copy()
    gdf_join_avg = gdf_join_avg.set_index("ID").join(df_na_d.apply(np.nanmean).rename("t2_na")).copy()
    gdf_join_avg["bias"] = gdf_join_avg.t2_wrf - gdf_join_avg.t2_na
    ### We only train over urban netatmo defined as stations located in a modal WRF LCZ
    gdf_join_avg_urb = gdf_join_avg[gdf_join_avg.lu_index > 30].dropna().copy()
    
    for mod in list_models:
        print(mod)
        var = gdf_join_avg_urb.drop(
            columns=["geometry", "index_right", "Unnamed: 0", "Lon", "Lat", "moduleID", "index", "lu_index"]
                    ).copy()
        ### For now, we only train over morphological WRF parameters
        var_n = var.columns[
            (~var.columns.str.contains("LCZ")) & 
            (~var.columns.str.contains("lu_index"))]
        mod_param = df_params.copy()

        reg_mod_n = str(mod)[:-2]
        if reg_mod_n == 'DummyRegressor':
            reg_mod = mod
        else:
            reg_mod = mod.set_params(**ast.literal_eval(
                        mod_param.Model_Params[np.int8(mod_param[mod_param.Model == reg_mod_n].index).squeeze()])
            ) 
        ### Observed bias
        obs_avg = var.bias
        ### Set of covariates to predict the bias
        cov_avg = var.loc[:, var_n].drop(columns=["t2_wrf", "t2_na", "bias"])
        j = 0
        for date in dates_list:
            print(date)
            ### We train the model on each timestep to compare with a temporal average training
            gdf_wrf_tmp = gdf_wrf.join(df_wrf_d.loc[date].T2.rename("t2_wrf")).set_index(
                ["XLAT", "XLONG"]).copy()
            gdf_join_tmp = gdf_join.join(gdf_wrf_tmp.t2_wrf).copy()
            gdf_join_tmp = gdf_join_tmp.set_index("ID").join(df_na_d.loc[date].rename("t2_na")).copy()
            gdf_join_tmp["bias"] = gdf_join_tmp.t2_wrf - gdf_join_tmp.t2_na
            ### We only train over urban netatmo defined as stations located in a modal WRF LCZ
            gdf_join_tmp_urb = gdf_join_tmp[gdf_join_tmp.lu_index > 30].dropna()
            
            
            var_tmp = gdf_join_tmp_urb.drop(
                columns=["geometry", "index_right", "Unnamed: 0", "Lon", "Lat", "moduleID", "index", "lu_index"]
                        ).copy()
           
            ### Observed bias
            obs_tmp = var_tmp.bias
            ### Set of covariates to predict the bias
            cov_tmp = var_tmp.loc[:, var_n].drop(columns=["t2_wrf", "t2_na", "bias"])
            tmp_df = pd.DataFrame(index = var.index) 
            
            for i in range (bootstrap_n):

                cov_train_tstep, cov_test_tstep, obs_train_tstep, obs_test_tstep = train_test_split(cov_tmp, obs_tmp,
                                                                            test_size=samp_frac, random_state=i)
                reg_mod.fit(cov_train_tstep, obs_train_tstep)
                pred_tstep = reg_mod.predict(cov_test_tstep)
                
                ### If we want to compare apples with apples we have to train the model only against the same subsamples
                ### of covariates at each CWS location running on day-d
                cov_train_avg, cov_test_avg, obs_train_avg, obs_test_avg = train_test_split(
                    cov_avg.loc[cov_tmp.index], obs_avg.loc[obs_tmp.index], test_size=samp_frac, random_state=i)
                reg_mod.fit(cov_train_avg, obs_train_avg)
                pred_avg = reg_mod.predict(cov_test_avg)
                  
                tmp_df.loc[cov_test_tstep.index, 'pred_tstep_b' + str(i) + '_t' + str(j)
                           ] = var_tmp.loc[obs_test_tstep.index].t2_wrf - pred_tstep
                tmp_df.loc[cov_test_tstep.index, 'pred_avg_b' + str(i) + '_t' + str(j)
                           ] = var_tmp.loc[obs_test_tstep.index].t2_wrf - pred_avg
                if mod == list_models[0]:
                    tmp_df.loc[cov_test_tstep.index, 'obs_b' + str(i) + '_t' + str(j)] = var_tmp.t2_na
                    tmp_df.loc[cov_test_tstep.index, 'wrf_b' + str(i) + '_t' + str(j)] = var_tmp.t2_wrf
            
                
            if j == 0:
                tstep_boot_ar = np.array([
                    np.nanmean(np.nanmean(tmp_df.to_numpy(), axis = 0)[i::4]) for i in range(4)
                    ])
            else:
                tstep_boot_ar = np.vstack((tstep_boot_ar,
                                        np.array([
                                            np.nanmean(np.nanmean(tmp_df.to_numpy(), axis = 0)[i::4]) for i in range(4)
                                            ])
                                           )
                                         )

            del tmp_df
            j += 1
            

        if mod == list_models[0]:
            ds_bootstrap = xr.Dataset({'obs': ('time', tstep_boot_ar[:,2]),
                                       'wrf': ('time', tstep_boot_ar[:,3])},
                                 coords={'time': pd.DatetimeIndex(dates_list)}
                )
                
        ds_bootstrap[reg_mod_n + '_avg'] = ('time', tstep_boot_ar[:,1])
        ds_bootstrap[reg_mod_n + '_tstep'] = ('time', tstep_boot_ar[:,0])
        ds_bootstrap[reg_mod_n + '_avg'].attrs['long_name'] = "daily " + stat + " BC T2 by t-avg obs (bootstrap x25)"
        ds_bootstrap[reg_mod_n + '_avg'].attrs['units'] = "[°C]"
        ds_bootstrap[reg_mod_n + '_tstep'].attrs['long_name'] = "daily " + stat + " BC T2 by t-step obs (bootstrap x25)"
        ds_bootstrap[reg_mod_n + '_tstep'].attrs['units'] = "[°C]"
        
    ds_bootstrap.attrs['global attr'] = 'Data Source: {}'.format(
        'Daily ' + stat + ' Bias Corrected T2 (Summer 2018 WRF BEP-BEM; bootstrap x25)')

    enc = {x: {"zlib": True} for x in ds_bootstrap}
    if boulac == True:
        ds_bootstrap.to_netcdf(savedir + 'T2_BouLac_Urb-BC_daily_' + stat + '_bootstrap_x25_' + city + '.nc', 
                                encoding=enc)
    else:
        ds_bootstrap.to_netcdf(savedir + 'T2_Urb-BC_daily_' + stat + '_bootstrap_x25_' + city + '.nc', 
                            encoding=enc)
