import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import keras,joblib,json,warnings,shap,tensorflow as tf
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score as A,roc_auc_score as U,confusion_matrix as C
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint as PC
from lime.lime_tabular import LimeTabularExplainer as L
a,v,t,ya,yv,yt=load();T=t[...,None]
O=keras.saving.load_model(f'{W}/results/original_run/cnn_model.keras',compile=False);I=keras.saving.load_model(f'{W}/results/original_run/improved_cnn.keras',compile=False)
RF=joblib.load(f'{W}/results/original_run/random_forest.pkl')
bce=keras.losses.BinaryCrossentropy()
def grad(m,x,y):
    x=tf.constant(x);y=tf.constant(y,dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(x);l=bce(y,tf.reshape(m(x,training=False),[-1]))
    return g.gradient(l,x).numpy()
def fgsm(m,x,y,e): return x+e*np.sign(grad(m,x,y))
def pgd(m,x,y,e,n=10):
    z=x.copy();s=2.5*e/n
    for _ in range(n): z=np.clip(z+s*np.sign(grad(m,z,y)),x-e,x+e)
    return z
pr=lambda m,x:m.predict(x,verbose=0,batch_size=512).ravel()
E=[0,.01,.05,.1,.15,.2,.3];R=[]
for e in E:
    for n,m in [('Original',O),('Improved',I)]:
        xa=T if e==0 else fgsm(m,T,yt,e);xp=T if e==0 else pgd(m,T,yt,e)
        p=pr(m,xa)>=.5;q=pr(m,xp)>=.5
        R.append(dict(eps=e,model=n,fgsm=A(yt,p)*100,pgd=A(yt,q)*100,fgsm_vpn_share=p.mean()*100,
            rf_transfer=A(yt,RF.predict(xa[...,0]))*100,zero_grad=float((grad(m,T,yt)==0).mean()*100)))
pd.DataFrame(R).round(2).to_csv('robustness_saved.csv',index=False);print(pd.DataFrame(R).round(2).to_string())
# stats
ph={k:(pr(m,T)>=.5).astype(int) for k,m in [('Original',O),('Improved',I)]};ph['RF']=RF.predict(t)
S={}
for k,h in ph.items():
    c=(h==yt).sum();S[k]=dict(acc=c/len(yt)*100,ci=[x*100 for x in PC(c,len(yt),method='wilson')],cm=C(yt,h).tolist())
for x,y_ in [('RF','Improved'),('RF','Original'),('Improved','Original')]:
    ax=ph[x]==yt;bx=ph[y_]==yt;tab=[[(ax&bx).sum(),(ax&~bx).sum()],[(~ax&bx).sum(),(~ax&~bx).sum()]]
    r=mcnemar(tab,exact=False,correction=True);S[f'{x}_vs_{y_}']=dict(tab=np.array(tab).tolist(),chi2=float(r.statistic),p=float(r.pvalue))
json.dump(S,open('stats.json','w'),indent=1,default=float);print(json.dumps(S,default=float))
# SHAP
np.random.seed(42);bg=a[np.random.choice(len(a),100,replace=False)]
sx=t[:300]
rs=shap.TreeExplainer(RF).shap_values(sx);rs=rs[:,:,1] if rs.ndim==3 else rs[1]
cs=[]
for n,m in [('Original',O),('Improved',I)]:
    e=shap.GradientExplainer(m,bg[...,None]);v_=np.array(e.shap_values(sx[...,None],nsamples=200,rseed=42)).squeeze();cs.append(v_);print(n,v_.shape)
np.savez('shap.npz',rf=rs,orig=cs[0],imp=cs[1],x=sx)
tb=pd.DataFrame(dict(feature=FEATS,RF=np.abs(rs).mean(0),Original=np.abs(cs[0]).mean(0),Improved=np.abs(cs[1]).mean(0)))
for c in ['RF','Original','Improved']: tb[c+'_rank']=tb[c].rank(ascending=False).astype(int)
tb.sort_values('RF',ascending=False).round(5).to_csv('shap_importance.csv',index=False);print(tb.sort_values('RF',ascending=False).round(4).to_string())
print('spearman',tb[['RF','Original','Improved']].corr('spearman').round(3).to_string())
# LIME
ex=L(a,feature_names=FEATS,class_names=['Non-VPN','VPN'],mode='classification',discretize_continuous=True,random_state=42)
fr=lambda z:RF.predict_proba(z)
def fc(m): return lambda z:(lambda p:np.c_[1-p,p])(m.predict(z[...,None],verbose=0,batch_size=4096).ravel())
W_={'RF':fr,'Improved':fc(I)};LA={}
for k,f in W_.items():
    arr=np.zeros((100,15))
    for i in range(100):
        mp=dict(ex.explain_instance(t[i],f,num_features=15,num_samples=2000,labels=(1,)).as_map()[1])
        arr[i]=[mp[j] for j in range(15)]
    LA[k]=arr
np.savez('lime.npz',**LA)
lt=pd.DataFrame(dict(feature=FEATS,RF=np.abs(LA['RF']).mean(0),Improved=np.abs(LA['Improved']).mean(0)));lt.round(5).to_csv('lime_importance.csv',index=False)
print(lt.sort_values('RF',ascending=False).round(4).to_string())
for k,sc in [('RF','RF'),('Improved','Improved')]:
    print(k,'lime vs shap spearman',round(pd.Series(lt[k]).corr(tb[sc],method='spearman'),3),'top5 overlap',len(set(lt.nlargest(5,k).feature)&set(tb.nlargest(5,sc).feature)))
# examples
wrong=[i for i in range(len(yt)) if ph['Improved'][i]!=yt[i] and ph['RF'][i]==yt[i]][0]
out={}
for i in [0,wrong]:
    for k,f in W_.items():
        e=ex.explain_instance(t[i],f,num_features=8,num_samples=5000,labels=(1,))
        out[f'{i}_{k}']=dict(true=int(yt[i]),pred=int(f(t[i:i+1]).argmax()),p_vpn=float(f(t[i:i+1])[0,1]),w=e.as_list(label=1))
json.dump(out,open('lime_examples.json','w'),indent=1);print(json.dumps(out)[:1800])
print('feature std (train scaled=1), raw check done')
