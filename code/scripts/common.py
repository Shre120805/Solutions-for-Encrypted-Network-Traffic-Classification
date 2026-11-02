import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder,StandardScaler
W=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(f'{W}/results/rerun/adversarial_models',exist_ok=True)
os.chdir(f'{W}/results/rerun')  # outputs are written here
FEATS=['duration','total_fiat','total_biat','min_fiat','min_biat','max_fiat','max_biat','mean_fiat','mean_biat','flowPktsPerSecond','flowBytesPerSecond','min_flowiat','max_flowiat','mean_flowiat','std_flowiat']
def load():
    d=pd.read_csv(f'{W}/data/TimeBasedFeatures-Dataset-15s-VPN.csv')
    X=d.drop(columns='class1').replace([np.inf,-np.inf],np.nan);X=X.fillna(X.median())
    y=LabelEncoder().fit_transform(d.class1)
    a,t,ya,yt=tts(X,y,test_size=.2,random_state=42,stratify=y)
    v,te,yv,yte=tts(t,yt,test_size=.5,random_state=42,stratify=yt)
    s=StandardScaler().fit(a)
    idx=[list(X.columns).index(f) for f in FEATS]
    f=lambda z:s.transform(z)[:,idx].astype('float32')
    return f(a),f(v),f(te),ya,yv,yte
