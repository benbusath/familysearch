# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:09:25 2019

@author: Ben Busath
"""

"""
We want to see how much each feature contributes to the xgboost model. We will 
do this using cross validation.
"""

import sys
import pandas as pd
import numpy as np
import turbodbc
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from time import time

sys.path.append("R:/JoePriceResearch/record_linking/projects/deep_learning/ml-record-linking/build/lib.win-amd64-3.7")
from splycer.record_set import RecordDataFrame
from splycer.pairs_set import PairsCOO
from splycer.feature_engineer import FeatureEngineer
from splycer.xgboost_match import XGBoostMatch


col_names=np.array(["marstat", "race", "rel", "mbp", "fbp", "first_sdxn", "last_sdxn", "bp", "county",
        "immigration", "birth_year", 'res_lat_lon', 'bp_lat_lon',
        'first_vec', 'last_vec',
        "first_jaro", "last_jaro",
        "first_trigram_comm", "last_trigram_comm",
        "first_bigram_comm", "last_bigram_comm",
        "first_ngram_comm", "last_ngram_comm"
        ])

def create_feature_engineer(cols_to_drop=np.ones(10)):
    indices_to_remove=np.array(np.where(np.logical_not(cols_to_drop))).tolist()[0]
    
    print("creating feature engineer")
    fe = FeatureEngineer()
    cols = ["marstat", "race", "rel", "mbp", "fbp", "first_sdxn", "last_sdxn", "bp", "county",
            "immigration", "birth_year", ["res_lat", "res_lon"], ["bp_lat", "bp_lon"],
            [f"first_vec{i}" for i in range(2, 202)], [f"last_vec{i}" for i in range(2,202)],
            "first", "last",
            "first", "last",
            "first", "last",
            "first", "last"
           ]
    

    #similarity functions
    col_comps = ["exact match"] * 9
    col_comps.extend(["abs dist"] * 2)
    col_comps.extend(["geo dist"] * 2)
    col_comps.extend(["euclidean dist"] * 2)
    col_comps.extend(["jw"] * 2)
    col_comps.extend(["trigram"] * 2)
    col_comps.extend(["bigram"] * 2)
    col_comps.extend(['ngram'] * 2)
    #extra arguments
    col_args = list({} for i in range(5))
    col_args.extend([{"comm_weight": "d", "comm_col": "first_comm"}, {"comm_weight": "d", "comm_col": "last_comm"},
                     {"comm_weight": "d", "comm_col": "bp_comm"}])
    col_args.extend(list({} for i in range(7)))
    col_args.extend([{"comm_weight": "d", "comm_col": "first_comm"}, {"comm_weight": "d", "comm_col": "last_comm"}] * 4)

    for idx in sorted(indices_to_remove, reverse=True):
        del cols[idx]
        del col_args[idx]
        del col_comps[idx]
    #print(str(len(cols))+' '+str(len(col_comps))+' '+str(len(col_args)))
    assert len(cols) == len(col_comps) == len(col_args)
    for i, j, k in zip(cols, col_comps, col_args):
        fe.add_comparison(i,j,k)
    return fe

if __name__=="__main__":
    #load training data
    print("loading data")
    conn = turbodbc.connect(dsn="rec_db")
    data = pd.read_sql("select * from Price.dbo.training_data_1910_1920", conn)
    
     #Build record sets
    print("creating record set 1")
    cols1910 = [s for s in data.columns if "1910" in s]
    data1910 = data[cols1910].drop_duplicates()
    data1910.columns = [s.replace("_1910", "") for s in data1910.columns]
    record_set1 = RecordDataFrame(1910, data1910.set_index("index", drop=True))
    del data1910
        
    print("creating record set 2")
    cols1920 = [s for s in data.columns if "1920" in s and s != "true_index_1920"]
    data1920 = data[cols1920].drop_duplicates()
    data1920.columns = [s.replace("_1920", "") for s in data1920.columns]
    record_set2 = RecordDataFrame(1920, data1920.set_index("index", drop=True))
    del data1920
    
    #Build pairs sets
    print("creating pairs set")
    uids1 = data["index_1910"]
    uids2 = data["index_1920"]
    is_match = (data["index_1920"] == data["true_index_1920"])
    pairs_set = PairsCOO(1910, 1920, uids1, uids2, is_match)
    del data
    
    #create feature engineer
    fe=create_feature_engineer()
    
    #training initial model:
    model = XGBClassifier(n_estimators=2000, n_jobs=8,importance_type='gain')
    print("creating xgb match object")
    xgb_match = XGBoostMatch(record_set1, record_set2, pairs_set, fe, model)
    print("training")
    xgb_match.train()   

    time_list=[]
    dropped_list=[]
    feature_list=[]
    precision_list=[]
    recall_list=[]
    
    
    thresholds = np.sort(xgb_match.model.feature_importances_)
    try:
        for thresh in thresholds:
            # select features using threshold
            selection = SelectFromModel(xgb_match.model, threshold=thresh, prefit=True)
            feature_idx=selection.get_support()
            cols_dropped=col_names[np.logical_not(feature_idx)]
            cols_to_run=col_names[feature_idx]
            
            
            print('columns dropped: '+ str(cols_dropped))
            print('new feature set: '+ str(cols_to_run))
            # train model
            fe=create_feature_engineer(cols_to_drop=feature_idx)
            new_model = XGBClassifier(n_estimators=2000, n_jobs=8,importance_type='gain')
            print("creating new xgb match object")
            new_xgb_match = XGBoostMatch(record_set1, record_set2, pairs_set, fe, new_model)
            print("training")
            
            tic = time()
            new_xgb_match.train()
            toc = time()
            
            time_list.append(toc-tic)
            dropped_list.append(cols_dropped)
            feature_list.append(cols_to_run)
            precision_list.append(new_xgb_match.test_precision)
            recall_list.append(new_xgb_match.test_recall)
    except: pass

    results=pd.DataFrame({'features':feature_list,'dropped':dropped_list,\
                  'precision':precision_list,'recall':recall_list,\
                  'threshold':thresholds,'train_time':time_list})
        
    results.to_csv('feature_selection_results.csv',index=False)

