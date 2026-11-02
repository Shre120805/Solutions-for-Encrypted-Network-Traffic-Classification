import sys,os,time,json;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import keras,tensorflow as tf
from keras import layers as Y
from keras.regularizers import l2
from sklearn.metrics import accuracy_score as A,roc_auc_score as U
from sklearn.ensemble import RandomForestClassifier as RFC
a,v,t,ya,yv,yt=load();Ta,Tv,Tt=a[...,None],v[...,None],t[...,None]
E=[0,.01,.05,.1,.15,.2,.3]
def orig():
    return keras.Sequential([keras.Input((15,1)),Y.Conv1D(64,3,activation='relu'),Y.BatchNormalization(),Y.MaxPooling1D(2),Y.Conv1D(128,3,activation='relu'),Y.BatchNormalization(),Y.Flatten(),Y.Dense(64,activation='relu'),Y.Dropout(.3),Y.Dense(1,activation='sigmoid')])
def imp():
    r=lambda:l2(.001)
    return keras.Sequential([keras.Input((15,1)),Y.Conv1D(32,3,padding='same',activation='relu',kernel_regularizer=r()),Y.BatchNormalization(),Y.Conv1D(64,3,padding='same',activation='relu',kernel_regularizer=r()),Y.BatchNormalization(),Y.GlobalMaxPooling1D(),Y.Dense(64,activation='relu'),Y.Dropout(.4),Y.Dense(32,activation='relu'),Y.Dropout(.3),Y.Dense(1,activation='sigmoid')])
bce=keras.losses.BinaryCrossentropy()
def fgsm(m,x,y,e):
    x=tf.constant(x);y=tf.constant(y,dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(x);l=bce(y,tf.reshape(m(x,training=False),[-1]))
    return (x+e*tf.sign(g.gradient(l,x))).numpy()
def pgd(m,x,y,e,n=10):
    z=x.copy();s=2.5*e/n
    for _ in range(n): z=np.clip(fgsm(m,z,y,s),x-e,x+e)
    return z
pr=lambda m,x:m.predict(x,verbose=0,batch_size=512).ravel()
def ev(m):
    d=dict(clean=A(yt,pr(m,Tt)>=.5)*100,auc=U(yt,pr(m,Tt)))
    for e in E[1:]:
        d[f'f{e}']=A(yt,pr(m,fgsm(m,Tt,yt,e))>=.5)*100
    for e in (.05,.1,.2): d[f'p{e}']=A(yt,pr(m,pgd(m,Tt,yt,e))>=.5)*100
    return d
def std(mk,ep,pat,rlr,s):
    keras.utils.set_random_seed(s);m=mk();m.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    cb=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=pat,restore_best_weights=True)]
    if rlr:cb.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-5))
    st=time.time();h=m.fit(Ta,ya,validation_data=(Tv,yv),epochs=ep,batch_size=64,callbacks=cb,verbose=0);return m,time.time()-st,len(h.history['loss'])
def at(s,et,ep=40):
    keras.utils.set_random_seed(s);m=imp();opt=keras.optimizers.Adam();ds=tf.data.Dataset.from_tensor_slices((Ta,ya.astype('float32'))).shuffle(20000,seed=s).batch(64)
    @tf.function
    def step(x,y):
        with tf.GradientTape() as g:
            g.watch(x);l=bce(y,tf.reshape(m(x,training=False),[-1]))
        xa=x+et*tf.sign(g.gradient(l,x))
        with tf.GradientTape() as g:
            lc=bce(y,tf.reshape(m(x,training=True),[-1]));la=bce(y,tf.reshape(m(xa,training=True),[-1]))
            L=.5*lc+.5*la+tf.add_n(m.losses)
        opt.apply(g.gradient(L,m.trainable_variables),m.trainable_variables)
    best,bw,w,wl=1e9,None,0,0;st=time.time();n=0
    for n in range(ep):
        for x,y in ds: step(x,y)
        vv=.5*float(bce(yv,pr(m,Tv)))+.5*float(bce(yv,pr(m,fgsm(m,Tv,yv,et))))
        if vv<best-0: best,bw,w,wl=vv,m.get_weights(),0,0
        else:
            w+=1;wl+=1
            if wl>=3: opt.learning_rate.assign(max(float(opt.learning_rate)*.5,1e-5));wl=0
            if w>=7:break
    m.set_weights(bw);return m,time.time()-st,n+1
rows=[];SE=[1,2,3,4,5]
def rec(n,s,m,tm,ep,extra={}):
    d=dict(model=n,seed=s,time=tm,epochs=ep,**extra,**ev(m));d['val_clean']=A(yv,pr(m,Tv)>=.5)*100
    for e in (.05,.1): d[f'val_f{e}']=A(yv,pr(m,fgsm(m,Tv,yv,e))>=.5)*100
    rows.append(d);pd.DataFrame(rows).round(4).to_csv('seeds.csv',index=False);print(n,s,round(tm),ep,round(d['clean'],2),round(d['f0.1'],2),flush=True)
for s in SE:
    st=time.time();r=RFC(200,max_depth=20,min_samples_leaf=2,class_weight='balanced',random_state=s,n_jobs=-1).fit(a,ya);tm=time.time()-st
    rows.append(dict(model='RF',seed=s,time=tm,clean=A(yt,r.predict(t))*100,auc=U(yt,r.predict_proba(t)[:,1])))
    m,tm,ep=std(orig,30,5,False,s);rec('Original',s,m,tm,ep)
    m,tm,ep=std(imp,40,7,True,s);rec('Improved',s,m,tm,ep)
    for et in (.05,.1):
        m,tm,ep=at(s,et);rec(f'AT{et}',s,m,tm,ep);m.save(f'adversarial_models/at_{et}_{s}.keras')
print('DONE')
