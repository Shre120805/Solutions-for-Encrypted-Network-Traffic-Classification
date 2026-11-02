import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import keras,joblib,warnings;warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score as A
a,v,t,ya,yv,yt=load();T=t[...,None]
O=keras.saving.load_model(f'{W}/results/original_run/cnn_model.keras',compile=False);I=keras.saving.load_model(f'{W}/results/original_run/improved_cnn.keras',compile=False);RF=joblib.load(f'{W}/results/original_run/random_forest.pkl')
R=[];g=np.random.default_rng(42)
for e in [.01,.05,.1,.15,.2,.3]:
    r=[]
    for _ in range(5):
        n=e*g.choice([-1.,1.],size=t.shape).astype('float32')
        r.append([A(yt,RF.predict(t+n))*100,A(yt,O.predict((t+n)[...,None],verbose=0,batch_size=512).ravel()>=.5)*100,A(yt,I.predict((t+n)[...,None],verbose=0,batch_size=512).ravel()>=.5)*100])
    R.append([e]+list(np.mean(r,0).round(2)))
c=pd.DataFrame(R,columns=['eps','RF','Original','Improved']);c.to_csv('random_noise.csv',index=False);print(c.to_string())
d=pd.read_csv(f'{W}/data/TimeBasedFeatures-Dataset-15s-VPN.csv')
tr,_=tts(d,test_size=.2,random_state=42,stratify=d.class1)
q=pd.DataFrame({'std':tr[FEATS].std(),'median_abs':tr[FEATS].abs().median(),'iqr':tr[FEATS].quantile(.75)-tr[FEATS].quantile(.25)})
q['eps.01_over_median']=.01*q['std']/q.median_abs;q['iqr_in_std']=q.iqr/q['std'];print(q.round(3).to_string())
print('frac test within .01 of feature mode etc: median of |z|',np.round(np.median(np.abs(t),axis=0),3))
