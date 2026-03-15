# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 14:39:40 2026

@author: JDawg
"""
import os
from tqdm import tqdm
import glob
import numpy as np
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from evals.IMAGE_comparison import get_IMAGE_maps
import yaml
import utils
import os
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef

#names 'ml_op' 'historical_guvi' 'forecast_guvi'
model_name = 'forecast_guvi'

device = "cuda" if torch.cuda.is_available() else "cpu"
with open("config.yaml") as stream:
    stat_dict =yaml.safe_load(stream)

#inference pipeline
gt = get_IMAGE_maps()[['datetime', 'maps']]
inputs = pd.read_csv(r"E:\ml_aurora\organized_solar_wind.csv")

#can only get from nothern scans
gt['datetime'] = pd.to_datetime(gt['datetime']).dt.round('min') #just round to the nearest minute
inputs['datetime'] = pd.to_datetime(inputs['time'])
inputs['time'] = inputs['datetime']
inputs['doy'] = inputs.datetime.dt.dayofyear
inputs = inputs[inputs.datetime < gt.datetime.max()]


#GATHER SPACE WEATHER INDICES
st = '2001-08-01'
et = '2002-12-01'

print('Downloading AL & AU: auroral electrojet indices...')
url = rf'https://lasp.colorado.edu/space-weather-portal/latis/dap/kyoto_ae_indices.csv?time,al,au&time>={st}T00:00:00Z&time<={et}T00:00:00Z'
ej_df = pd.read_csv(url)
ej_df['time'] = pd.to_datetime(ej_df['time (yyyy-MM-dd HH:mm:ss.SSS)'])
print('done')



print('Downloading Hp30 and Ap30: geomagnetic indice...')
url = fr'https://kp.gfz.de/app/hpodata?startdate={st}&enddate={et}&format=Hp30_txt#hpo-data-download-207'
cols = ['year', 'month', 'day', 'hour', 'minute', 'trash', 'trash1', 'hp30', 'ap30', 'D']
hp_df = pd.read_csv(url, sep=r'\s+', names=cols, header=None)
hp_df['time'] = pd.to_datetime(hp_df[['year', 'month', 'day', 'hour', 'minute']])
hp_df = hp_df.drop(columns=['trash', 'trash1', 'year', 'month', 'day', 'hour', 'minute', 'D'])
print('done')


print('Downloading F10.7: solar activity indice...')
sa_df = pd.read_csv(r'https://celestrak.org/SpaceData/SW-All.csv')[['F10.7_OBS', 'DATE']]
sa_df['time'] = pd.to_datetime(sa_df['DATE'])
sa_df = sa_df[sa_df.time.dt.year >= 2000]
print('done')

#TIME BACKBONE
time_index = pd.date_range(st, et, freq='1min')
time_df = pd.DataFrame({'time': time_index})


#MERGE EVERY DATAFRAME INTO ONE 
out = time_df
out = pd.merge_asof(out, inputs, on = 'time', direction = 'backward')
out = pd.merge_asof(out, sa_df, on = 'time',direction = 'backward')
out = pd.merge_asof(out, hp_df, on = 'time', direction = 'backward')
inputs = pd.merge_asof(out, ej_df, on = 'time',direction = 'backward')


mm = pd.merge(inputs, gt, on = 'datetime', how = 'left').drop_duplicates('time')
idxs = [i for i in range(len(mm)) if not np.isnan(mm.iloc[i].maps).any()]

# four hour context for inputs
hr = 4

if model_name == 'ml_op':
    
    #ML-OP model
    model = utils.FC_to_Conv().to(device)
    model.load_state_dict(torch.load(r'weights/OP_weight.pth', weights_only = True))
    model.eval()
    
    
    # gather input datas and preprocess
    channels = stat_dict['data_statistics']['op_channels']
    inp_mean = np.array(stat_dict['data_statistics']['input_mean'][:5])
    inp_std = np.array(stat_dict['data_statistics']['input_std'][:5])
    img_mean = np.array(stat_dict['data_statistics']['op_img_mean'])
    img_std = np.array(stat_dict['data_statistics']['op_img_std'])
    
    
    inps = np.stack([inputs[['Bx','By','Bz','vel','doy']].iloc[idx - 60*(hr+1): idx-60].to_numpy().astype(np.float32) for idx in tqdm(idxs)])#crude, but just using 60 minutes
    inps[:,:,:4] = (inps[:,:,:4] - inp_mean[None,None,:4])/inp_std[None,None,:4]
    inps[:,:,-1] = inps[:,:,-1]/365
    mask = np.isnan(inps)
    inps[mask] = 0
    
    
    #inference
    inps = torch.tensor(inps, dtype = torch.float32).to(device)
    y_hat = model(inps.reshape(inps.shape[0],-1).unsqueeze(1)).detach().cpu().numpy()
    y_hat = (y_hat * img_std[None,:,None,None]) + img_mean[None,:,None,None]
    
    #electron energy flux
    ej_hat = y_hat[:,::2][:,:3].sum(axis = 1)
    

if model_name == 'forecast_guvi':
    
    #ML-GUVI model
    model = utils.FC_to_Conv(241*10, c_out = 1).to(device)
    # model.load_state_dict(torch.load(r'weights/four_hour_context_0436.pth', weights_only = True))
    model.load_state_dict(torch.load(r'dummy_weight.pth', weights_only = True))

    model.eval()
    
    # gather input datas and preprocess
    inp_mean = np.array(stat_dict['data_statistics']['input_mean'])
    inp_std = np.array(stat_dict['data_statistics']['input_std'])
    img_mean = np.array(stat_dict['data_statistics']['guvi_img_mean'])
    img_std = np.array(stat_dict['data_statistics']['guvi_img_std'])
    
    #here is where i need to add the other indices
    cols = stat_dict['data_statistics']['inputs_cols']
    inps = np.stack([inputs[cols].iloc[idx - 60*(hr+1): idx-60].to_numpy().astype(np.float32) for idx in tqdm(idxs)]) #crude, but just using 60 minutes
    inps = (inps - inp_mean[None,None,:])/inp_std[None,None,:]
    mask = np.isnan(inps)
    inps[mask] = 0
    inps = np.concatenate([inps, np.zeros((inps.shape[0], inps.shape[-1]))[:,None,:]], axis = 1) #keep for guvi

    #inference
    inps = torch.tensor(inps, dtype = torch.float32).to(device)
    y_hat = model(inps.reshape(inps.shape[0],-1).unsqueeze(1)).detach().cpu().numpy()
    ej_hat = np.expm1(y_hat)
    # ej_hat = (y_hat * img_std[None,0,None,None]) + img_mean[None,0,None,None]

if model_name == 'historical_guvi':
    
    #ML-GUVI model
    model = utils.FC_to_Conv(241*10, c_out = 1).to(device)
    model.load_state_dict(torch.load(r'weights/four_hour_historical_0402.pth', weights_only = True))
    model.eval()
    
    # gather input datas and preprocess
    inp_mean = np.array(stat_dict['data_statistics']['input_mean'])
    inp_std = np.array(stat_dict['data_statistics']['input_std'])
    img_mean = np.array(stat_dict['data_statistics']['guvi_img_mean'])
    img_std = np.array(stat_dict['data_statistics']['guvi_img_std'])
    
    #here is where i need to add the other indices
    cols = stat_dict['data_statistics']['inputs_cols']
    inps = np.stack([inputs[cols].iloc[idx - 60*(hr): idx].to_numpy().astype(np.float32) for idx in tqdm(idxs)]) #crude, but just using 60 minutes
    inps = (inps - inp_mean[None,None,:])/inp_std[None,None,:]
    mask = np.isnan(inps)
    inps[mask] = 0
    inps = np.concatenate([inps, np.zeros((inps.shape[0], inps.shape[-1]))[:,None,:]], axis = 1) #keep for guvi

    #inference
    inps = torch.tensor(inps, dtype = torch.float32).to(device)
    y_hat = model(inps.reshape(inps.shape[0],-1).unsqueeze(1)).detach().cpu().numpy()
    ej_hat = np.expm1(y_hat)    
    
#calculate mcc's
maps = np.stack(mm.maps.iloc[idxs].to_numpy())
y_true = maps.astype(int).ravel()
thresholds = [.125, .25, .5 , 1, 2]
mcc_list = []


for t in thresholds:
    y_pred = (ej_hat > t).astype(int).ravel()
    mcc = matthews_corrcoef(y_true, y_pred)
    mcc_list.append(mcc)


plt.figure()
plt.plot(mcc_list)
plt.title('ML-OP vs IMAGE boundaries')
plt.xlabel('energy flux threshold')
plt.ylabel('mcc')
plt.ylim(0,1)
plt.show()

#%%

for i in range(len(ej_hat)):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(ej_hat[i,0] > .25)
    plt.title('ml_guvi')
    
    plt.subplot(1,2,2)
    plt.imshow(maps[i])
    plt.title('IMAGE boundaries')
    plt.suptitle(mm.iloc[idxs[i]].time)
    plt.show()

