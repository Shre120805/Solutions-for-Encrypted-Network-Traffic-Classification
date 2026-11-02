import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data'))
import pandas as pd,numpy as np,glob,time,sys
import importlib.util as u
print({m:bool(u.find_spec(m)) for m in ['sklearn','tensorflow','shap','lime']})
fs=glob.glob('**/*.csv',recursive=True)
for f in fs:
    d=pd.read_csv(f);print(f,d.shape,dict(d.iloc[:,-1].value_counts()))
f=[x for x in fs if '15s' in x][0]
d=pd.read_csv(f);X=d.drop(columns='class1')
print('dups',d.duplicated().sum(),'feature dups',X.duplicated().sum(),'-1 cells',(X==-1).sum().sum(),'neg cells',(X<0).sum().sum())
print(X.describe().T[['min','50%','mean','max']].to_string())
for a,b in [('min_fiat','max_fiat'),('min_biat','max_biat'),('min_fiat','mean_fiat'),('total_fiat','max_fiat'),('total_fiat','min_fiat'),('max_fiat','min_fiat'),('min_flowiat','max_flowiat'),('min_flowiat','mean_flowiat'),('mean_flowiat','max_flowiat'),('min_active','max_active'),('min_idle','max_idle')]:
    print(a,'<=',b,round((X[a]<=X[b]).mean(),4))
print('dur vs min_fiat <=',round((X.min_fiat<=X.duration).mean(),4))
print(d.head(3).T.to_string())
