import numpy as np
from pandas import read_csv,concat
from src.dataset import OpportunityDS
import matplotlib.pyplot as plt

df = concat([read_csv('data/Opportunity/train/S3-Drill.dat',delimiter=' +',engine='python',header=None), read_csv('data/Opportunity/train/S4-Drill.dat',delimiter=' +',engine='python',header=None)],ignore_index=True)
ds = OpportunityDS(df)

all_y = []
for i in range(len(ds)):
    x,y,_ = ds[i]
    all_y += y
    
n,_,_=plt.hist(all_y,[-0.5,0.5,1.5,2.5,3.5,4.5])
plt.show()

plt.bar(list(range(len(n))),n/(n.sum()))
plt.show()