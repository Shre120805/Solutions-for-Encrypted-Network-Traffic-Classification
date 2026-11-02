import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import keras,joblib
from sklearn.metrics import accuracy_score as A,roc_auc_score as U,confusion_matrix as C,classification_report as R
a,v,t,ya,yv,yt=load()
for n in ['cnn_model','improved_cnn']:
    m=keras.saving.load_model(f'{W}/results/original_run/{n}.keras',compile=False)
    p=m.predict(t[...,None],verbose=0).ravel();h=(p>=.5).astype(int)
    print(n,round(A(yt,h)*100,2),round(U(yt,p),4),C(yt,h).tolist(),m.count_params())
    print(R(yt,h,digits=4)); m.summary()
try:
    r=joblib.load(f'{W}/results/original_run/random_forest.pkl');p=r.predict_proba(t)[:,1];h=(p>=.5).astype(int)
    print('rf',A(yt,r.predict(t))*100,U(yt,p),C(yt,r.predict(t)).tolist());print(R(yt,r.predict(t),digits=4))
except Exception as e: print('RF load fail',repr(e)[:300])
