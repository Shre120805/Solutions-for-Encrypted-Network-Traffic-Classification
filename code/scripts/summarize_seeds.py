import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','results','rerun'))
import pandas as pd
d=pd.read_csv('seeds.csv');n=d.select_dtypes('number').drop(columns='seed')
m=n.groupby(d.model).mean().round(2).T;s=n.groupby(d.model).std().round(2).T
print(m.to_string());print(s.to_string())
