# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:51:12 2026

@author: JDawg
"""

import glob
import os
import pandas as pd
from tqdm import tqdm
import numpy as np


fp = r'E:\ml_aurora\guvi_paired_data\images'


data = []

for f in tqdm(glob.glob(os.path.join(fp, '*.npy'))):
    
    d = np.load(f)
    d = d[~np.isnan(d)]
    data.extend(d)
    

    
#%%
import seaborn as sns
import matplotlib.pyplot as plt
    
df = pd.DataFrame(data)
df = df[df< df.quantile(.9999)]


mm = df.sample(frac = 1e-2)
plt.figure()
sns.histplot(mm)
plt.title('original distribution')
plt.show()

plt.figure()
sns.histplot(mm, log_scale = True)
plt.title('log transform of original distribution')
plt.show()


#%%
mask = df!= 0
kk = df[mask]

std = kk.std()
mean = kk.mean()
p9999 = kk.max()

log_std = np.std(np.log(kk))
log_mean = np.mean(np.log(kk))
log_p9999 = np.max(np.log(kk))



print(std)
print(mean)
print(p9999)

print(log_std)
print(log_mean)
print(log_p9999)
