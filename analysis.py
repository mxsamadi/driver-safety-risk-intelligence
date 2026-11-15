from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_score,recall_score
import matplotlib.pyplot as plt
import json,zipfile
p=Path(__file__).resolve().parent;(p/'data').mkdir(exist_ok=True);(p/'outputs').mkdir(exist_ok=True);(p/'sql').mkdir(exist_ok=True)
rng=np.random.default_rng(20261115);n=6500
miles=np.clip(rng.lognormal(10.1,.48,n),3000,110000).round().astype(int)
experience=rng.integers(0,31,n);viol=rng.poisson(.35+ .000006*miles + .3*(experience<3));prior=rng.binomial(1,np.clip(.05+.06*viol+.05*(experience<3),0,.7));training=rng.binomial(1,.62,n);region=rng.choice(['Northeast','South','Midwest','West'],n);fleet=rng.choice(['Small','Medium','Large'],n,p=[.3,.4,.3]);compliance=rng.poisson(.2+.14*viol+.1*prior);logit=-3.7+.42*viol+.85*prior+.22*compliance+.000013*miles-.018*experience-.27*training+.18*(region=='South')
prob=1/(1+np.exp(-logit));outcome=rng.binomial(1,prob)
df=pd.DataFrame({'driver_id':[f'D{i:05d}' for i in range(n)],'annual_miles':miles,'years_experience':experience,'prior_violations':viol,'prior_incident':prior,'compliance_events':compliance,'completed_training':training,'region':region,'fleet_size':fleet,'future_12m_incident':outcome})
df.to_csv(p/'data'/'synthetic_driver_data.csv',index=False)
features=[c for c in df if c not in ['driver_id','future_12m_incident']];cat=['region','fleet_size'];num=[c for c in features if c not in cat]
train,temp=train_test_split(df,test_size=.4,random_state=42,stratify=df.future_12m_incident);valid,test=train_test_split(temp,test_size=.5,random_state=42,stratify=temp.future_12m_incident)
prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
models={'Logistic regression':Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000))]),'Gradient boosting':Pipeline([('prep',prep),('model',HistGradientBoostingClassifier(max_iter=110,max_leaf_nodes=10,l2_regularization=5,random_state=42))])}
val_scores={};fitted={}
for name,model in models.items():
 model.fit(train[features],train.future_12m_incident);fitted[name]=model;v=model.predict_proba(valid[features])[:,1];val_scores[name]=average_precision_score(valid.future_12m_incident,v)
selected=max(val_scores,key=val_scores.get);model=fitted[selected];pred=model.predict_proba(test[features])[:,1];y=test.future_12m_incident.to_numpy();threshold=np.quantile(pred,.8);flag=pred>=threshold
metrics={'data_type':'synthetic','n_records':n,'event_rate':round(float(df.future_12m_incident.mean()),4),'selected_model':selected,'selection_criterion':'validation average precision','test_roc_auc':round(float(roc_auc_score(y,pred)),3),'test_pr_auc':round(float(average_precision_score(y,pred)),3),'test_brier':round(float(brier_score_loss(y,pred)),3),'top_20pct_recall':round(float(y[flag].sum()/y.sum()),3),'top_20pct_precision':round(float(y[flag].mean()),3),'test_n':len(y)}
(p/'outputs'/'metrics.json').write_text(json.dumps(metrics,indent=2))
fig,ax=plt.subplots(figsize=(7,4));bins=pd.qcut(pred,10,labels=False,duplicates='drop');agg=pd.DataFrame({'bin':bins,'p':pred,'y':y}).groupby('bin').agg(predicted=('p','mean'),observed=('y','mean'));ax.plot([0,1],[0,1],'--',label='Perfect calibration');ax.plot(agg.predicted,agg.observed,'o-',label='Selected model');ax.set(xlabel='Mean predicted risk',ylabel='Observed incident rate',title='Calibration by risk decile (synthetic holdout)',xlim=(0,max(.25,agg.predicted.max()*1.1)),ylim=(0,max(.25,agg.observed.max()*1.1)));ax.legend();fig.tight_layout();fig.savefig(p/'outputs'/'calibration.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));rates=df.groupby('prior_violations').future_12m_incident.agg(['mean','count']);rates=rates[rates['count']>=30];ax.bar(rates.index.astype(str),rates['mean']);ax.set(xlabel='Prior violations',ylabel='Future 12-month incident rate',title='Illustrative association in synthetic data');fig.tight_layout();fig.savefig(p/'outputs'/'risk_by_violations.png',dpi=160);plt.close(fig)
