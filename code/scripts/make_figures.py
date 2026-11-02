import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
import json,keras,joblib,warnings;warnings.filterwarnings('ignore')
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch as B
from sklearn.metrics import roc_curve,auc
F=f'{W}/results/figures/'
INK='#0b0b0b';SEC='#52514e';GRID='#e6e5e0'
BL,OR,AQ,VI='#2a78d6','#eb6834','#1baf7a','#6250d6'
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.edgecolor':SEC,'axes.linewidth':.6,'axes.labelcolor':INK,'xtick.color':SEC,'ytick.color':SEC,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':.5,'axes.axisbelow':True,'legend.frameon':False,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':.02,'legend.fontsize':8})
# fig 1 pipeline
fig,ax=plt.subplots(figsize=(3.45,3.7));ax.axis('off');ax.grid(False);ax.set_xlim(0,10);ax.set_ylim(-.1,14.1)
def box(x,y,w,h,t,bold=False,fc='#f4f7fc',ec=BL):
    ax.add_patch(B((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.12',fc=fc,ec=ec,lw=.8));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8,color=INK,fontweight='bold' if bold else 'normal',linespacing=1.1)
def ar(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color=SEC,lw=.8,mutation_scale=7))
box(.05,12.6,9.9,1.4,'ISCX VPN-nonVPN, 15-second time-based flows\n18,758 flows, 23 features, VPN or Non-VPN',True)
box(.05,10.3,9.9,1.75,'Stratified 80/10/10 split, standardization fitted on\ntraining data, 15 features selected by mutual information')
ar((5,12.6),(5,12.07))
xs=[.05,3.4,6.75];L_=['Solution 1\nRandom\nForest','Solution 2\nStandard\n1D-CNN','Solution 3\nImproved\n1D-CNN, plus\nadv. training']
for x,t,c in zip(xs,L_,[AQ,OR,BL]):
    box(x,7.4,3.2,2.3,t,True,fc='#ffffff',ec=c);ar((5,10.3),(x+1.6,9.72))
box(.05,5,4.85,1.85,'Evaluation: accuracy,\nprecision, recall, F1,\nROC-AUC, McNemar test')
box(5.1,5,4.85,1.85,'Explanation: SHAP\n(global) and LIME\n(per flow)')
ar((1.65,7.4),(2.5,6.87));ar((8.35,7.4),(7.5,6.87))
box(.05,2.2,9.9,2.2,'Attacks: FGSM and PGD on CNNs, transfer attack\non Random Forest, random-noise control,\nfive training seeds')
ar((2.5,5),(2.5,4.42));ar((7.5,5),(7.5,4.42))
box(.05,0,9.9,1.65,'Three-way comparison of accuracy,\nexplanations and robustness',True);ar((5,2.2),(5,1.67))
fig.savefig(F+'pipeline.png');plt.close()
# ROC
a,v,t,ya,yv,yt=load();T=t[...,None]
O=keras.saving.load_model(f'{W}/results/original_run/cnn_model.keras',compile=False);I=keras.saving.load_model(f'{W}/results/original_run/improved_cnn.keras',compile=False);RF=joblib.load(f'{W}/results/original_run/random_forest.pkl')
fig,ax=plt.subplots(figsize=(3.3,2.2))
for n,p,c,ls in [('Random Forest',RF.predict_proba(t)[:,1],AQ,'-'),('Standard 1D-CNN',O.predict(T,verbose=0).ravel(),OR,'-'),('Improved 1D-CNN',I.predict(T,verbose=0).ravel(),BL,'--')]:
    x,y,_=roc_curve(yt,p);ax.plot(x,y,c=c,lw=1.5,ls=ls,label=f'{n} (AUC = {auc(x,y):.4f})')
ax.plot([0,1],[0,1],c=SEC,lw=.7,ls=':',label='Chance');ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate');ax.legend(loc='lower right',borderpad=0,handlelength=1.6);ax.set_xlim(0,1);ax.set_ylim(0,1.02)
fig.savefig(F+'roc.png');plt.close()
# robustness
d=pd.read_csv(f'{W}/results/rerun/seeds.csv');E=[0,.01,.05,.1,.15,.2,.3]
fig,axs=plt.subplots(2,1,figsize=(3.3,3.2),sharey=True)
S=[('Original','Standard 1D-CNN',OR,'o'),('Improved','Improved 1D-CNN',BL,'s'),('AT0.05','Adv. trained, ε=0.05',AQ,'^'),('AT0.1','Adv. trained, ε=0.10',VI,'D')]
for k,n,c,mk in S:
    g=d[d.model==k];cols=['clean']+[f'f{e}' for e in E[1:]]
    m,s=g[cols].mean().values,g[cols].std().values
    axs[0].plot(E,m,c=c,marker=mk,ms=3.5,lw=1.3,label=n);axs[0].fill_between(E,m-s,m+s,color=c,alpha=.12,lw=0)
    ep=[0,.05,.1,.2];cols=['clean','p0.05','p0.1','p0.2'];m,s=g[cols].mean().values,g[cols].std().values
    axs[1].plot(ep,m,c=c,marker=mk,ms=3.5,lw=1.3,label=n);axs[1].fill_between(ep,m-s,m+s,color=c,alpha=.12,lw=0)
axs[0].set_title('(a) FGSM',fontsize=8,loc='left',color=INK);axs[1].set_title('(b) PGD, 10 steps',fontsize=8,loc='left',color=INK)
for x in axs: x.set_ylabel('Test accuracy (%)');x.set_ylim(0,90);x.axhline(52.2,c=SEC,lw=.6,ls=':')
axs[1].text(.2,54,'always predict VPN',ha='right',fontsize=8,color=SEC)
axs[1].set_xlabel('Perturbation strength ε (standardized feature units)');fig.tight_layout(rect=(0,0,1,.89));h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center',bbox_to_anchor=(.53,1.0));fig.savefig(F+'robust.png');plt.close()
# SHAP
s=pd.read_csv(f'{W}/results/rerun/shap_importance.csv')
fig,axs=plt.subplots(1,2,figsize=(3.45,2.85),sharey=True);o=s.sort_values('RF');y=np.arange(15)
axs[0].barh(y,o.RF,color=AQ,height=.72);axs[1].barh(y,o.Improved,color=BL,height=.72);axs[0].set_yticks(y,o.feature,fontsize=8,color=INK)
axs[0].set_title('(a) Random Forest',fontsize=8,loc='left',color=INK);axs[1].set_title('(b) Improved 1D-CNN',fontsize=8,loc='left',color=INK)
for x in axs: x.grid(axis='y',visible=False);x.tick_params(axis='x',labelsize=8)
fig.supxlabel('Mean absolute SHAP value',fontsize=8,x=.62,y=.02);fig.tight_layout(w_pad=.6);fig.savefig(F+'shap.png');plt.close()
# LIME
L=json.load(open(f'{W}/results/rerun/lime_examples.json'));import re
fig,axs=plt.subplots(2,1,figsize=(3.45,3.1))
for x,k,tt in zip(axs,['0_RF','0_Improved'],['(a) Random Forest: Non-VPN (correct), P(VPN) = 0.09','(b) Improved 1D-CNN: VPN (wrong), P(VPN) = 0.57']):
    w=L[k]['w'][::-1];y=np.arange(len(w));v=[i[1] for i in w]
    n=[re.search(r'[A-Za-z_]{4,}',i[0]).group() for i in w]
    x.barh(y,v,color=[BL if q>0 else OR for q in v],height=.7);x.set_yticks(y,n,fontsize=8,color=INK);x.axvline(0,c=SEC,lw=.6);x.set_title(tt,fontsize=8,loc='right',color=INK);x.grid(axis='y',visible=False);x.tick_params(axis='x',labelsize=8)
axs[1].set_xlabel('LIME weight (blue: toward VPN, orange: toward Non-VPN)',fontsize=8)
fig.tight_layout();fig.savefig(F+'lime.png');plt.close()
