# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:34:02 2019
@author: admin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('data.csv',encoding='gbk')
col = ['custid',# 'trade_no',  不要'Unnamed: 0'  'bank_card_no',
       'low_volume_percent', 'middle_volume_percent',
       'take_amount_in_later_12_month_highest',
       'trans_amount_increase_rate_lately', 'trans_activity_month',
       'trans_activity_day', 'transd_mcc', 'trans_days_interval_filter',
       'trans_days_interval', 'regional_mobility', #'student_feature',
       'repayment_capability', 'is_high_user', 'number_of_trans_from_2011',
       'first_transaction_time', 'historical_trans_amount',
       'historical_trans_day', 'rank_trad_1_month', 'trans_amount_3_month',
       'avg_consume_less_12_valid_month', 'abs',
       'top_trans_count_last_1_month', 'avg_price_last_12_month',
       #'reg_preference_for_trad',#'avg_price_top_last_12_valid_month',
       'trans_top_time_last_1_month', 'trans_top_time_last_6_month',
       'consume_top_time_last_1_month', 'consume_top_time_last_6_month',
       'cross_consume_count_last_1_month',
       'trans_fail_top_count_enum_last_1_month',
       'trans_fail_top_count_enum_last_6_month',
       'trans_fail_top_count_enum_last_12_month',
       'consume_mini_time_last_1_month',
       'max_cumulative_consume_later_1_month',
       'max_consume_count_later_6_month',
       'railway_consume_count_last_12_month',
       'pawns_auctions_trusts_consume_last_1_month',
       'pawns_auctions_trusts_consume_last_6_month',
       'jewelry_consume_count_last_6_month', #'source',
       'first_transaction_day', 'trans_day_last_12_month', #'id_name',
       'apply_score', 'apply_credibility', 'query_org_count',
       'query_finance_count', 'query_cash_count', 'query_sum_count',
       'latest_one_month_apply',#'latest_query_time',
       'latest_three_month_apply', 'latest_six_month_apply', 'loans_score',
       'loans_credibility_behavior', 'loans_count', 'loans_settle_count',
       'loans_overdue_count', 'loans_org_count_behavior',
       'consfin_org_count_behavior', 'loans_cash_count',
       'latest_one_month_loan', 'latest_three_month_loan',
       'latest_six_month_loan', 'history_suc_fee', 'history_fail_fee',
       'latest_one_month_suc', 'latest_one_month_fail', 'loans_long_time',
       'loans_credit_limit', 'loans_credibility_limit',#'loans_latest_time',
       'loans_org_count_current', 'loans_product_count', 'loans_max_limit',
       'loans_avg_limit', 'consfin_credit_limit', 'consfin_credibility',
       'consfin_org_count_current', 'consfin_product_count',
       'consfin_max_limit', 'consfin_avg_limit', 'latest_query_day',
       'loans_latest_day','status']
dataset = dataset[col]
dataset = dataset.dropna()
#dataset = dataset[dataset['']]

dataX = dataset[dataset.columns[:-1]]
dataY = dataset['status']

trainX,testX,trainY,testY = train_test_split(dataX,dataY,test_size=0.3,random_state = 1)

#for col in dataset.columns:#画出散点图
#    plt.xlabel(col)
#    plt.scatter(range(len(dataset[col])),dataset[col])
#    plt.show()
#
#
#def cal_WOE(dataset,col,targe):
#    subdata=df(dataset.groupby(col)[col].count())
#    suby=df(dataset.groupby(col)[targe].sum())
#    data=df(pd.merge(subdata,suby,how="left",left_index=True,right_index=True))
#    b_total=data[targe].sum()
#    total=data[col].sum()
#    g_total=total-b_total
#    data["bad"]=data.apply(lambda x:round(x[targe]/b_total,3),axis=1)
#    data["good"]=data.apply(lambda x:round((x[col]-x[targe])/g_total,3),axis=1)
#    data["WOE"]=data.apply(lambda x:np.log(x.bad/x.good),axis=1)
#    return data.loc[:,["bad","good","WOE"]]
# 
# 
#def cal_IV(dataset):
#    dataset["IV"]=dataset.apply(lambda x:(x.bad-x.good)*x.WOE,axis=1)
#    IV=sum(dataset["IV"])
#    return IV
#
##---------------IV值筛选特征
#data_IV = pd.DataFrame()
#fea_iv = []
#
#col_list = [col for col in trainX]
#
#for col in col_list:
#        col_WOE = cal_WOE(dataset, col, 'status')
#        col_IV = cal_IV(col_WOE)
#        if col_IV > 0.1:
#            data_IV[col] = [col_IV]
#            fea_iv.append(col)
#print(fea_iv)
#
#univ = [col for col in trainX.columns if col not in fea_iv]
#print(univ)
#-----------------------------
#---------随机森林筛选特征

#rfc = RandomForestClassifier()
#rfc.fit(trainX, trainY)
#rfc_impc = pd.Series(rfc.feature_importances_, index=trainX.columns).sort_values(ascending=False)
#fea_gini = rfc_impc[:20].index.tolist()
#print('rf select', fea_gini)
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(trainX, trainY)
y_pre_xgb = xgb.predict(testX)
print(classification_report(y_pre_xgb, testY))


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(trainX, trainY)
y_pre_dtc = dtc.predict(testX)
print(classification_report(y_pre_dtc, testY))

from sklearn.svm import SVC
# 'linear', 'poly', 'rbf', 'sigmoid'
svc = SVC()
svc.fit(trainX, trainY)
y_pre_svc = svc.predict(testX)
print(classification_report(y_pre_svc, testY))



from sklearn.model_selection import GridSearchCV


param_grid = {'max_depth':np.arange(1, 20, 1)}
rfc = RandomForestClassifier(n_estimators=111,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=5)
GS.fit(trainX,trainY)
print(GS.best_score_)

param_grid = {'max_features':np.arange(5,20,1)} 
rfc = RandomForestClassifier(n_estimators=111,random_state=90)
GS = GridSearchCV(rfc,param_grid,cv=5)
GS.fit(trainX,trainY)
print(GS.best_score_)

parameters = {'min_samples_split':[2,5,10,15,100]  #默认为2
                ,'min_samples_leaf':[1,2,5,10]}
rfc = RandomForestClassifier(n_estimators=111,random_state=25)
GS = GridSearchCV(rfc, parameters, cv=5)
GS.fit(trainX,trainY)
#print("时间:{}".format(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f")))
print(GS.best_score_)

#clf = RandomForestClassifier(n_estimators=111,random_state=90).fit(Xtrain,Ytrain)
#Ytest_pred2 = clf.predict(Xtest)
#acc2 = metrics.accuracy_score(Ytest,Ytest_pred2)
#precision2 = metrics.precision_score(Ytest,Ytest_pred2)
#recall2 = metrics.recall_score(Ytest,Ytest_pred2)
#f1_2 = metrics.f1_score(Ytest,Ytest_pred2)
#auc2= metrics.roc_auc_score(Ytest,Ytest_pred2)
#print('准确率:{:.4f},精确率:{:.4f},召回率:{:.4f},f1-score:{:.4f},auc:{:.4f}'.format(acc2,precision2,recall2,f1_2,auc2))
