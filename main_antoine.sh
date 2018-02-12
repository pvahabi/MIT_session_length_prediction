#!/bin/bash


#name_dataset="enriched_sample_anonymized"
#name_dataset="main_anonymized"
name_dataset="last_fm"


DIRLOAD="../helper_data_bis"
DIROUT="../results_bis"


############################ BASIC FEATURES ############################ 
kind_features="basic"

which_model='sequential'
do_processing='False'
run_models='True'
#python main.py $kind_features $name_dataset $DIRLOAD $DIROUT/$name_dataset/$kind_features $which_model 0 $do_processing $run_models





############################ ADVANCED FEATURES ############################ 
kind_features='advanced'


##### 1/ Sequential ##### 
which_model='sequential'
do_processing='False'
run_models='True'
python main.py $kind_features $name_dataset $DIRLOAD $DIROUT/$name_dataset/$kind_features/$which_model $which_model 0 $do_processing $run_models


##### 2/ EB-Ridge ##### 
which_model='EB-penalization'

#for range_lambda in 2 3 1 0
for range_lambda in 3
do
python main.py $kind_features $name_dataset $DIRLOAD $DIROUT/$name_dataset/$kind_features/$which_model $which_model $range_lambda
done



##### 3/ EB-xgboost ##### 
which_model='EB-xgboost'

#for range_lambda in 3 2 1 0
for range_lambda in 1
do
python main.py $kind_features $name_dataset $DIRLOAD $DIROUT/$name_dataset/$kind_features/$which_model $which_model $range_lambda
done
















############################ XGBOOST ADVANCED FEATURES ############################ 
kind_features='xgboost_advanced'

##### 1/ Sequential ##### 
which_model='sequential'
do_processing='True'
run_models='True'
load_best_tree='True'
#python main.py $kind_features $name_dataset $DIRLOAD $DIROUT/$name_dataset/$kind_features/$which_model $which_model 0 $do_processing $run_models $load_best_tree



##### 2/ EB-Ridge ##### 
which_model='EB-penalization'
#for range_lambda in 2 3 1 0
#for range_lambda in 2
#do
#python main.py $kind_features $name_dataset $DIRLOAD $DIROUT/$name_dataset/$kind_features/$which_model $which_model $range_lambda
#done







