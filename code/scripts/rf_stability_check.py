import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','data'))
import pandas as pd,numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_selection import SelectKBest,mutual_info_classif as mi
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score as acc,roc_auc_score as auc,confusion_matrix as cm
d=pd.read_csv('TimeBasedFeatures-Dataset-15s-VPN.csv')
def run(d,seed,keepk=True):
    X=d.drop(columns='class1');y=LabelEncoder().fit_transform(d.class1)
    a,t,ya,yt=tts(X,y,test_size=.2,random_state=seed,stratify=y)
    v,te,yv,yte=tts(t,yt,test_size=.5,random_state=seed,stratify=yt)
    s=StandardScaler().fit(a);k=SelectKBest(mi,k=15).fit(s.transform(a),ya)
    f=lambda z:k.transform(s.transform(z))
    m=RF(200,max_depth=20,min_samples_leaf=2,class_weight='balanced',random_state=42,n_jobs=-1).fit(f(a),ya)
    p=m.predict(f(te));pr=m.predict_proba(f(te))[:,1]
    tr=set(map(tuple,a.values));seen=np.array([tuple(r) in tr for r in te.values])
    return acc(yte,p),auc(yte,pr),cm(yte,p).tolist(),int(seen.sum()),acc(yte[~seen],p[~seen]),list(X.columns[k.get_support()])
r=run(d,42);print('seed42',r)
R=[run(d,s)[0] for s in range(5)];print('5 splits',np.round(R,4),np.mean(R).round(4),np.std(R).round(4))
D=d.drop_duplicates();R=[run(D,s)[0] for s in range(5)];print('dedup',len(D),np.round(R,4),np.mean(R).round(4),np.std(R).round(4))
