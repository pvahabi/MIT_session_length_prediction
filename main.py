import sys
from prediction import *
import numpy as np


def main():
	kind_features = sys.argv[1]
	name_dataset  = sys.argv[2]
	load_file     = sys.argv[3]
	output_file   = sys.argv[4]
	which_model   = sys.argv[5]
	range_llambda = int(sys.argv[6])

	do_processing  = (sys.argv[7]=='True') if len(sys.argv) > 7 else False
	run_models     = (sys.argv[8]=='True') if len(sys.argv) > 8 else True
	load_best_tree = (sys.argv[9]=='True') if len(sys.argv) > 9 else False


	if which_model == 'sequential':
		prediction_sequential(kind_features, name_dataset, load_file, output_file, do_processing, run_models, load_best_tree)


	elif which_model == 'EB-penalization':
		all_range_lambda = [ np.arange(6,9,.5), np.arange(9,12,.5), np.arange(12,15,.5), np.arange(15,20,1), ] #if name_dataset != 'last_fm' else [ np.arange(10,13,.5), np.arange(13,16,.5), np.arange(16,19,.5), np.arange(100,150,10)] 
		list_llambda     = all_range_lambda[range_llambda]
		output_file     += '/'+str(list_llambda[0])+'_'+str(list_llambda[-1])

		prediction_EB_penalization(kind_features, name_dataset, load_file, output_file, list_llambda)


	elif which_model == 'EB-xgboost':
		all_range_lambda = [ np.arange(1,3,1), np.arange(3,5,1), np.arange(5,7,1), np.arange(7,9,1)]
		list_llambda     = all_range_lambda[range_llambda]
		output_file     += '/'+str(list_llambda[0])+'_'+str(list_llambda[-1])

		prediction_EB_xgboost(kind_features, name_dataset, load_file, output_file, list_llambda)


if __name__ == '__main__':
	main()