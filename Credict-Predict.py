# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:34:02 2019

@author: admin
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
       'reg_preference_for_trad',#'avg_price_top_last_12_valid_month',
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
       'jewelry_consume_count_last_6_month', 'source',
       'first_transaction_day', 'trans_day_last_12_month', #'id_name',
       'apply_score', 'apply_credibility', 'query_org_count',
       'query_finance_count', 'query_cash_count', 'query_sum_count',
       'latest_query_time', 'latest_one_month_apply',
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

for col in dataset.columns:#画出散点图
    plt.xlabel(col)
    plt.scatter(range(len(dataset[col])),dataset[col])
    plt.show()




