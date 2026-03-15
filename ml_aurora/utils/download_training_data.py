# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:07:56 2025

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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed



def download_solar_wind_data(out_dir = r'E:\solar_wind'):
    '''Downloads solar wind data from ACE.'''

    #downloads 1m resolution solar wind data 
    url = r'https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/'
    page = requests.get(url).text
    
    
    file_types = ['swepam', 'mag']
    pattern = re.compile(r'href="([^"]*(?:swepam|mag).*?\.txt)"')
    files = [url + m for m in pattern.findall(page)]
    
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()

    for f in tqdm(files, desc = 'Downloading solar wind data...'):
        fname = os.path.join(out_dir, os.path.basename(f))
    
        if os.path.exists(fname):   # skip existing
            continue
    
        with session.get(f, stream=True) as r:
            r.raise_for_status()
            with open(fname, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fp.write(chunk)
    print(f'Solar Wind Data: Complete Download from: {files[0][-24:-15]} to {files[-4][-24:-15]}')
      


#transform solar wind data
def collate_solar_wind(fp = r'E:\ml_aurora\solar_wind', out = r'E:\ml_aurora\organized_solar_wind.csv'):

    '''Creates time history of solar wind from ACE files. Can be called directly after 
    downloading solar wind data.'''
    mag_data = []
    vel_data = []
    
    for f in tqdm(glob.glob(os.path.join(fp, '*.txt')), desc = 'Loading in Solar Wind data...'):
        
    
        if 'mag' in f:
            with open(f, 'r') as fh:
                for i,line in enumerate(fh):
                    if i < 20:
                        continue
                    mag_data.append(line[:-2]) #ignore new line character
        else:
            
            with open(f, 'r') as fh:
                for i,line in enumerate(fh):
                    if i < 18:
                        continue
                    vel_data.append(line[:-2]) #ignore new line character
                    
    

    
    cols = ['year', 'month', 'day', 'hhmm','mjd', 'sec', 'mag_dqi', 'Bx', 'By', 'Bz','Bt', 'lat', 'lon']
    df = pd.DataFrame(mag_data)[0].str.split(expand=True).set_axis(cols, axis = 1)
    df = df[['year','month','day','hhmm','mag_dqi','Bx', 'By', 'Bz','Bt']]
    
    cols = ['year', 'month', 'day', 'hhmm','mjd', 'sec', 'vel_dqi', 'Np', 'vel', 'T_ion']
    vel_df = pd.DataFrame(vel_data)[0].str.split(expand=True).set_axis(cols, axis = 1)
    vel_df = vel_df[['year','month','day','hhmm','vel_dqi','vel', 'Np']]
    
    

    df = pd.merge(df, vel_df, on = ['year', 'month', 'day', 'hhmm'])
    df = df.astype(float)
    mask = ((df.vel_dqi == 9) 
              |(df.mag_dqi == 9) 
              |(df.vel_dqi == 9)
              |(df.vel <= -9999)
              |(df.Bz <= -999))

    cols = ['Bx', 'By', 'Bz', 'Bt', 'vel', 'Np']
    df.loc[mask, cols] = np.nan
    df['clock_ang'] = np.arctan2(df.By, df.Bz) #radians
    
    
    def build_datetime(df):
        hh = df['hhmm'] // 100
        mm = df['hhmm'] % 100
        return pd.to_datetime(
            dict(year=df.year, month=df.month, day=df.day,
                 hour=hh, minute=mm),
            errors='coerce'
        )
    
    df['time'] = build_datetime(df)
    time_index = pd.date_range(df.time.min(), df.time.max(), freq='1min')
    time_df = pd.DataFrame({'time': time_index})
    
    df = time_df.merge(df, on='time', how='left')
    df = df.drop_duplicates(subset='time', keep='first')
    df.to_csv(out)
    

    

    
    
#%% GUVI SPECIFIC DOWNLOAD FUNCTIONS
    
def guvi_to_images(og_data_fp = r'E:\guvi_aurora', out = r'E:\ml_aurora\guvi_images'):
    
    ''' Uses full GUVI .nc files to gather images of the energy flux of the northern and
    southern hemisphere. This step requires downloading all of the guvi aurora data into a SINGLE 
    directory.'''

    os.makedirs(out, exist_ok = True)
    files = glob.glob(os.path.join(og_data_fp, '*.ncdf'))
    op_mlat = np.arange(40,90, .5)
    op_mlt =  np.arange(0,24,.25)
    #each scan is ~24 minutes from start to stop
    for file in tqdm(files):
        
        
        ds = nc.Dataset(file, 'r')

        year = ds.variables['Year'][:][0]
        month = ds.variables['Month'][:][0]
        day = ds.variables['Day'][:][0]
        
        orbits = ds.variables['Orbit Number'][:]
        sorted_idx = orbits.argsort()
        #catches fill value
        if ((orbits.mask == True).any()) | (type(orbits) != np.ma.MaskedArray) | (year == 0):
            print('bad sample')
            continue
        




        north_fl   = ds.variables['Magnetic North Flux'][:][sorted_idx[:,None]].squeeze()
        north_ut   = ds.variables['Magnetic North UT second'][:][sorted_idx[:,None]].squeeze() #seconds since day start
        north_mlat = ds.variables['Magnetic North latitude'][:][sorted_idx[:,None]].squeeze()
        north_mlt  = ds.variables['Magnetic North Local Time'][:][sorted_idx[:,None]].squeeze()
        
        
        south_fl   = ds.variables['Magnetic South Flux'][:][sorted_idx[:,None]].squeeze()
        south_ut   = ds.variables['Magnetic South UT second'][:][sorted_idx[:,None]].squeeze() #seconds since day start
        south_mlat = ds.variables['Magnetic South latitude'][:][sorted_idx[:,None]].squeeze()
        south_mlt  = ds.variables['Magnetic South Local Time'][:][sorted_idx[:,None]].squeeze()


        north_mask = np.ones_like(north_ut)
        south_mask = np.ones_like(south_ut)
        
        north_mask[(north_ut == 0)] = np.nan
        south_mask[(south_ut == 0)] = np.nan
        

        north_mlon = (15.0 * north_mlt) % 360.0
        south_mlon = (15.0 * south_mlt) % 360.0
        
        

        
        
        north_fl *= north_mask
        north_ut *= north_mask
        north_mlat *= north_mask
        north_mlt *= north_mask
        
        south_fl *= south_mask
        south_ut *= south_mask
        south_mlat *= south_mask
        south_mlt *= south_mask
        
        base_date = datetime(year, month, day)

        north_datetime = [base_date + timedelta(seconds = int(np.nanmedian(t))) if ~np.isnan(np.nanmedian(t)) else np.nan for t in north_ut]
        south_datetime = [base_date + timedelta(seconds = int(np.nanmedian(t))) if ~np.isnan(np.nanmedian(t)) else np.nan for t in south_ut]
        
        nm = np.nanmedian(north_ut, axis = -1)
        sm = np.nanmedian(south_ut, axis = -1)
        
        if (nm[0] + nm[1] > 1.4e5) | (sm[0] + sm[1] > 1.4e5): # this check might eliminate <20 days, but its worth it for accurate pairing
            continue

        
        
        #this also assumes that the orbit number is consistent
        if not pd.isna(north_datetime[0]) and not pd.isna(north_datetime[1]):
            if (north_datetime[0] - north_datetime[1]) > timedelta(seconds=6000):
                north_datetime[0] -= timedelta(days=1)  # consistent day of year
        
        if not pd.isna(south_datetime[0]) and not pd.isna(south_datetime[1]):
            if (south_datetime[0] - south_datetime[1]) > timedelta(seconds=6000):
                south_datetime[0] -= timedelta(days=1)  # consistent day of year
            
        resampled_flux = np.full((len(north_datetime), len(op_mlat), len(op_mlt)), np.nan)
        for i in range(len(north_datetime)):
            if pd.isna(north_datetime[i]): #skip bad times
                continue
            
            title = 'north_'+north_datetime[i].strftime('%Y%m%d_%H%M%S')
            flux = north_fl[i]
            mlat = north_mlat[i]
            mlt = north_mlt[i]
            
            #resample to a fixed grid
            mlat_idx = np.argmin(np.abs(mlat[:,None] - op_mlat[None,:]), axis = 1) * north_mask[i]
            mlt_idx = np.argmin(np.abs(mlt[:,None] - op_mlt[None,:]), axis = 1) * north_mask[i]
            df = pd.DataFrame(data = np.stack([flux, mlat_idx, mlt_idx]).T, columns = ['flux', 'mlat', 'mlt'])
            mm = df.groupby(by = ['mlat','mlt'])['flux'].median() #maybe it should be mean but idk yet
            for (ml, mt), fl in mm.items():
                resampled_flux[i,int(ml), int(mt)] = fl

            np.save(os.path.join(out, f'{title}.npy'), resampled_flux[i])
            
        resampled_flux = np.full((len(south_datetime), len(op_mlat), len(op_mlt)), np.nan)
        for i in range(len(south_datetime)):
            if pd.isna(south_datetime[i]): #skip bad times
                continue
            
            title = 'south_'+south_datetime[i].strftime('%Y%m%d_%H%M%S')
            flux = south_fl[i]
            mlat = south_mlat[i]
            mlt = south_mlt[i]
            
            #resample to a fixed grid
            mlat_idx = np.argmin(np.abs(mlat[:,None] - -op_mlat[None,:]), axis = 1) * south_mask[i]
            mlt_idx = np.argmin(np.abs(mlt[:,None] - op_mlt[None,:]), axis = 1) * south_mask[i]
            df = pd.DataFrame(data = np.stack([flux, mlat_idx, mlt_idx]).T, columns = ['flux', 'mlat', 'mlt'])
            mm = df.groupby(by = ['mlat','mlt'])['flux'].median() #maybe it should be mean but idk yet
            for (ml, mt), fl in mm.items():
                resampled_flux[i,int(ml), int(mt)] = fl

            np.save(os.path.join(out, f'{title}.npy'), resampled_flux[i])
            
def guvi_input_data(guvi_images_fp = r'E:\ml_aurora\guvi_paired_data\images', 
                    guvi_inputs_fp = r'E:\ml_aurora\guvi_paired_data\inputs',
                    historical = False, sw_only = True):

    '''Creates the input data labels for the GUVI images that have been gathered.
    This should be called directly after guvi_to_images.'''
    

    os.makedirs(guvi_inputs_fp, exist_ok= True)
    #SOLAR WIND DATAFRAME
    #magnetopause UT is the time it takes for the solar wind at time 'time' to reach the earth's magnetopause
    df = pd.read_csv(r'E:\ml_aurora\organized_solar_wind.csv')
    df['time'] = pd.to_datetime(df['time'])
    df['magnetopause_ut'] = pd.to_timedelta(1.5e6 / df['vel'].rolling(window = 15, min_periods = 1).mean(), unit='s') + df.time
    mask = df['magnetopause_ut'].isna()
    df.loc[mask,'magnetopause_ut'] = pd.to_timedelta( 1.5e6/df.vel.mean(), unit = 's') + df.time #just imputes with average solar wind
    df = df.drop(columns = 'Unnamed: 0')



    #FILEPATH DATAFRAME

    data = []
    for ff in glob.glob(os.path.join(guvi_images_fp, '*.npy')):
        f = ff.split('\\')[-1]
        yr = int(f[6:10])
        mo = int(f[10:12])
        dy = int(f[12:14])
        hr = int(f[15:17])
        mi = int(f[17:19])
        se = int(f[19:21])
        ut = datetime(yr, mo, dy, hr, mi, se)
        
        data.append([ff, ut])
        
    if not historical:
        fdf = pd.DataFrame(data = data, columns = ['filepaths', 'magnetopause_ut'])
        fdf['magnetopause_ut'] = pd.to_datetime(fdf['magnetopause_ut'])
        fdf['scan_ut'] = fdf.magnetopause_ut
    
        #MERGE FILEPATH & SOLAR WIND DATAFRAMES
        mm = pd.merge_asof(fdf.sort_values('magnetopause_ut'), 
                           df.sort_values('magnetopause_ut'), 
                           on = 'magnetopause_ut',
                           direction = 'backward',
                           tolerance = timedelta(minutes = 2)
                           )
        
    else:
        fdf = pd.DataFrame(data = data, columns = ['filepaths', 'time'])
        fdf['time'] = pd.to_datetime(fdf['time'])
        fdf['scan_ut'] = fdf.time
    
        #MERGE FILEPATH & SOLAR WIND DATAFRAMES
        mm = pd.merge_asof(fdf.sort_values('time'), 
                           df.sort_values('time'), 
                           on = 'time',
                           direction = 'backward',
                           tolerance = timedelta(minutes = 2)
                           )
        mm['time'] = mm.time.dt.floor('min')
    df = pd.merge(df, mm[['time', 'filepaths', 'scan_ut']],  on ='time', how = 'left')



    #GATHER SPACE WEATHER INDICES
    st = '2000-01-01'
    et = '2025-12-01'

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
    out = pd.merge_asof(out, df, on = 'time', direction = 'backward')
    out = pd.merge_asof(out, sa_df, on = 'time',direction = 'backward')
    out = pd.merge_asof(out, hp_df, on = 'time', direction = 'backward')
    out = pd.merge_asof(out, ej_df, on = 'time',direction = 'backward')

    out = out.drop(columns = ['hhmm', 'clock_ang', 'hhmm',
                              'scan_ut',
                              'DATE', 'ap30',
                              'time (yyyy-MM-dd HH:mm:ss.SSS)', 
                              'year', 'month', 'day'])
    out['doy'] = out.time.dt.day_of_year

    idxs = list(out[~pd.isnull(out.filepaths)].index)
    hr = 4
    sw_hr = 72
    #i can calculate input data statistics here pretty quickly...
    cols = ['Bx', 'By', 'Bz', 'vel', 'Np', 'F10.7_OBS', 'hp30', 'au (nT)', 'al (nT)', 'doy']
    stats = out[cols].describe()
    print('Input data statistics... these are already in the config yaml file')
    print(stats.loc['mean'])
    print(stats.loc['std'])


    # Create paired input data based on guvi files available
    current_files = set(os.listdir(guvi_inputs_fp))
    def process_index(idx):
        """Process a single index and save the data if needed."""
        output_row = out['filepaths'].iloc[idx]
        fn = os.path.join(guvi_inputs_fp, output_row.split('\\')[-1])
        filename = fn.split('\\')[-1]
        
        if filename in current_files:
            return f'{fn} already downloaded...'
        
        if historical:
            indices = out[cols[5:]].iloc[idx-60*hr:idx]
            sw = out[cols[:5]].iloc[idx - 60*(hr+1): idx - 60]
            np.save(fn, np.hstack((sw.to_numpy(), indices.to_numpy())))
        if sw_only:
            input_data = out[cols].iloc[idx-60*sw_hr:idx]
            np.save(fn, input_data.to_numpy())
        else:
            input_data = out[cols].iloc[idx-60*hr:idx]
            np.save(fn, input_data.to_numpy())
        return f'{fn} saved'

    # Parallelize with ThreadPoolExecutor should be about x4 increase than a regular loop
    max_workers = os.cpu_count()  # Adjust based on your needs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_index, idx): idx for idx in idxs}
        
        for future in tqdm(as_completed(futures), total=len(idxs)):
            try:
                result = future.result()
                if "already downloaded" in result:
                    print(result)
            except Exception as e:
                idx = futures[future]
                print(f'Error processing index {idx}: {e}')
