# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:16:21 2021

@author: Oscar Brousse
"""

from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
import rasterio
import rioxarray
from rasterio.warp import reproject, Resampling
import glob
import time
import geopandas as gpd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split



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

geogrid_d03 = datadir_WRF_geog + 'geo_em.d03_LCZ_params.nc'
surf_info_d03 = xr.open_dataset(geogrid_d03)

WRF_grid = datadir_WRF_geog + 'geo_em.d03_gridinfo.tif'
WRF_tif = xr.open_rasterio(WRF_grid)

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

### For the averaged value we take the daily averaged temperature at date d
df_na_d = df_na_filt[startdate:enddate].resample('24h').mean()

T2_WRF_mean = (ds_wrf.loc[{'XTIME':slice(startdate,enddate)}].T2.resample(XTIME="1D").mean() - 273.15).values
da_T2_WRF = xr.DataArray(T2_WRF_mean, 
                         dims=['XTIME', 'south_north', 'west_east'],
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


###################################
###  SPATIAL MODELS HYPERTUNING ###
###################################

### Set up variables over which to train the ML regressors

var = gdf_join_avg_urb.drop(
    columns=["geometry", "index_right", "Unnamed: 0", "Lon", "Lat", "moduleID", "index", "lu_index"]
            ).copy()
### For now, we only train over morphological WRF parameters
var_n = var.columns[
    (~var.columns.str.contains("LCZ")) & 
    (~var.columns.str.contains("lu_index"))]

### Calculate the observed bias that we aim at predicting
obs_avg = var.bias
### Remove spurious covariates to predict the bias
cov_avg = var.loc[:, var_n].drop(columns=["t2_wrf", "t2_na", "bias"])

### ML, Ridge, Lasso, RF, XGboost

list_models = [LinearRegression(), Ridge(), Lasso(), RandomForestRegressor(), GradientBoostingRegressor()]
list_parameters = [{'normalize':(True, False)},
                   {'normalize':(True, False), 'alpha':np.arange(1,10,1), 'solver':('auto', 'sparse_cg', 'cholesky', 'svd',
                                                                                    'lsqr', 'sag', 'saga'),
                    'tol':10.0**(np.arange(-10, -1)), 'random_state':[42]},
                   {'normalize':(True, False), 'alpha':np.arange(1,10,1), 'random_state':[42],
                    'tol':10.0**(np.arange(-10, -1)), 'selection':('cyclic', 'random')},
                   {'max_features':('auto', 'sqrt', 'log2'),
                    'random_state':[42], 'n_estimators':np.arange(200, 1200, 200),
                    'min_samples_split':np.arange(2,np.ceil(len(list(ds_LCZp.keys())[1::])),4).astype(int),
                    'min_samples_leaf':(np.arange(1,np.ceil(len(list(ds_LCZp.keys())[1::])/2),2)).astype(int)},
                   {'max_features':('auto', 'sqrt', 'log2'),
                    'random_state':[42], 'n_estimators':np.arange(200, 1200, 200),
                    'learning_rate':np.arange(0.2,1.1,0.2),
                    'max_depth':np.arange(3,12,3), 'subsample':np.arange(0.2,1.1,0.2),
                    'min_samples_split':np.arange(2,np.ceil(len(list(ds_LCZp.keys())[1::])),4).astype(int),
                    'min_samples_leaf':(np.arange(2,np.ceil(len(list(ds_LCZp.keys())[1::])/2),2)).astype(int)}
                   ]

best_models=[]
for i in range(len(list_models)):
    start = time.time()
    model = list_models[i]
    params = list_parameters[i]
    
    ### We want to reduce the RMSE in priority to make sure that the model 
    ### minimizes the error in the residuals expected in the extremes. Another metric can be used.
    best_mod_srch = GridSearchCV(model, params, scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'], 
                                 n_jobs = 8, refit = 'neg_root_mean_squared_error', return_train_score=True, cv = 5)
    ### We don't train the model on the land use index values but on dummy variables extracted from it [comment out first element]
    best_mod_srch.fit(cov_avg.copy(), obs_avg.copy())
    best_models.append([str(model)[:-2], best_mod_srch.best_params_, 'Time : ' + str(time.time() - start) + 's'])
    scores_rsl = pd.DataFrame(best_mod_srch.cv_results_)
    scores_rsl.to_csv(savedir + 'GridSearchCV_Scores_' + str(model)[:-2] + '.csv')

    df_best_estims = pd.DataFrame(best_models, columns = ['Model', 'Model_Params', 'Time_BestModel_Eval'])
    df_best_estims.to_csv(savedir + 'GridSearchCV_BestModelParams.csv')
    
    del df_best_estims, scores_rsl



    

    
