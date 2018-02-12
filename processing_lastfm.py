import datetime
import math
import numpy  as np
import pandas as pd

THRESHOLD_MINUTS=30



def process_last_fm():

##### STORE GENRE AND YEAR OF USER  #####
	f = open('../Dataset/lastfm-dataset-1K/userid-profile.tsv','r')
	dict_sex = {'m':1,'f':0}

	aux=-1
	dict_user = {}
	all_reg = []
	for line in f:
		aux+=1
		line = line.split('\t')
		user_id, sex, registration = line[0], line[1], line[-1]
		registration = registration.split(' ')[-1]
		
		if len(registration)>1 and len(sex)>0 and aux>0:
			sex = dict_sex[sex] 
			registration = int(registration) 
			dict_user[user_id] = (sex, registration)




##### FEATURES FOR SESSIONS  #####
	f = open('../Dataset/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv','r')

	## features created
	columns = ['listener_id','genre', 'year', 'start_time', 'end_time', 'morning/afternoon', 'session_duration', 'log_session_duration', 
			   'absence', 'log_absence', 'past_session', 'log_past_session', 'averaged_past_session', 'log_averaged_past_session']


	aux=-1
	old_listener_id = 'user_000001'
	n_listener = 0
	old_time = datetime.datetime.utcnow()
	end_session  = old_time
	line_session = []
	all_sessions = []
	all_sessions_user = []




	for line in f:
		if True:
		#if aux < 1e6:
			aux+=1
			line = line.split('\t')


	####### A BIT STRONG ??
			if line[0] in dict_user:

				user_cov = dict_user[line[0]] ## user i

			### Test if new user
				if old_listener_id != line[0]:

				### Create absence time, past session time, averaged past session time for previous user
					if len(all_sessions_user)>0:
						all_sessions_user = all_sessions_user[::-1] ##chronological order
						
						sum_past_session = 0
						sum_log_past_session  = 0
						all_sessions_user[0] += [0 for _ in range(6)]
						
						for i in range(1, len(all_sessions_user)):
							absence_time      = (all_sessions_user[i][3]-all_sessions_user[i-1][4]).total_seconds()

							past_session_time     = all_sessions_user[i-1][5]
							sum_past_session     += past_session_time

							log_past_session_time = all_sessions_user[i-1][6]
							sum_log_past_session += log_past_session_time
							
							all_sessions_user[i] += [absence_time, math.log(absence_time),
													 past_session_time, log_past_session_time,
													 sum_past_session/i, sum_log_past_session/i]
						all_sessions += all_sessions_user
						
				
				### New user
					n_listener += 1
					print 'USER '+str(n_listener)+' N_session '+str(len(all_sessions))
					old_listener_id   = line[0]
					all_sessions_user = []

					
				
				time = line[1].split('T')[0]+' '+line[1].split('T')[1][:-1]
				time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
				
			
			### Test if new sessions
				if (old_time-time).total_seconds() > 60*THRESHOLD_MINUTS:


				## Create session_length, morning + new_session
					session_time = (end_session-old_time).total_seconds()
					if session_time>0:
						line_session.append(old_time)
						line_session.append(end_session)
						line_session.append(int(old_time.hour<13))
						line_session.append(session_time)
						line_session.append(math.log(session_time))
						all_sessions_user.append(line_session)
					#print line_session, end_session

					line_session  = []
					end_session   = time
					
					line_session.append(old_listener_id)
					line_session.append(user_cov[0])
					line_session.append(user_cov[1])
				old_time = time

			
		else:
			break
			
	df_all_sessions = pd.DataFrame(np.array(all_sessions), columns=columns)
	df_all_sessions = df_all_sessions.sort_values(['start_time'], ascending=True)
	df_all_sessions.index = range(df_all_sessions.shape[0])

	print 'Dataset processed shape:', df_all_sessions.shape 
	df_all_sessions.to_csv('../Dataset/last_fm.csv')
	return df_all_sessions












def process_last_fm_absence():

##### STORE GENRE AND YEAR OF USER  #####
	f = open('../Dataset/lastfm-dataset-1K/userid-profile.tsv','r')
	dict_sex = {'m':1,'f':0}

	aux=-1
	dict_user = {}
	all_reg = []
	for line in f:
	    aux+=1
	    line = line.split('\t')
	    user_id, sex, registration = line[0], line[1], line[-1]
	    registration = registration.split(' ')[-1]

	    if len(registration)>1 and len(sex)>0 and aux>0:
	        sex = dict_sex[sex] 
	        registration = int(registration) 
	        dict_user[user_id] = (sex, registration)




##### FEATURES FOR SESSIONS  #####
	f = open('../Dataset/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv','r')

	## features created
	columns = ['listener_id','genre', 'year', 'start_time', 'end_time', 'morning/afternoon', 'session_duration', 'log_session_duration', 
	           'absence', 'log_absence', 'past_absence', 'log_past_absence', 'averaged_past_absence', 'averaged_log_past_absence']


	aux=-1
	old_listener_id = 'user_000001'
	n_listener = 0
	old_time = datetime.datetime.utcnow()
	end_session  = old_time
	line_session = []
	all_sessions = []
	all_sessions_user = []




	for line in f:
	    
	    if True:
	    #if aux < 1e6:
	        aux+=1
	        line = line.split('\t')

	        if line[0] in dict_user:

	            user_cov = dict_user[line[0]] ## user i

	        ### Test if new user
	            if old_listener_id != line[0]:
	      
	        

	########## CHANGE HERE: at least 2 sessions ##########

	            ### Create absence time, past session time, averaged past session time for previous user
	                if len(all_sessions_user)>2:
	                    all_sessions_user = all_sessions_user[::-1] ##chronological order
	                    
	

	                    ### new
	                    sum_past_absence       = 0
	                    sum_log_past_absence   = 0
	                    all_sessions_user[0]  += [0 for _ in range(6)]
	                    all_sessions_user[-1] += [0 for _ in range(6)]
	                    ### end new

	                    for i in range(1, len(all_sessions_user)-1):
	                    
	                        ### new
	                        absence_time = (all_sessions_user[i+1][3]-all_sessions_user[i][4]).total_seconds() - 1800
	                        
	                        past_absence_time      = (all_sessions_user[i][3]-all_sessions_user[i-1][4]).total_seconds() - 1800
	                        sum_past_absence      += past_absence_time
	                        
	                        log_past_absence_time  = math.log(absence_time)
	                        sum_log_past_absence  += log_past_absence_time
	                        ### end new
	                        

	                        all_sessions_user[i] += [absence_time, math.log(absence_time),
	                                                 past_absence_time, log_past_absence_time,
	                                                 sum_past_absence/i, sum_log_past_absence/i]
	                    all_sessions += all_sessions_user[1:-1]
	                    #print np.min([ len(session_user) for session_user in all_sessions_user])
	########## END CHANGE HERE ##########
	                    

	            ### New user
	                n_listener += 1
	                print 'USER '+str(n_listener)+' N_session '+str(len(all_sessions))
	                old_listener_id   = line[0]
	                all_sessions_user = []



	            time = line[1].split('T')[0]+' '+line[1].split('T')[1][:-1]
	            time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")


	        ### Test if new sessions
	            if (old_time-time).total_seconds() > 60*THRESHOLD_MINUTS:


	            ## Create session_length, morning + new_session
	                session_time = (end_session-old_time).total_seconds()
	                if session_time>0:
	                    line_session.append(old_time)
	                    line_session.append(end_session)
	                    line_session.append(int(old_time.hour<13))
	                    line_session.append(session_time)
	                    line_session.append(math.log(session_time))
	                    all_sessions_user.append(line_session)
	                #print line_session, end_session

	                line_session  = []
	                end_session   = time

	                line_session.append(old_listener_id)
	                line_session.append(user_cov[0])
	                line_session.append(user_cov[1])
	            old_time = time


	    else:
	        break

	df_all_sessions = pd.DataFrame(np.array(all_sessions), columns=columns)
	df_all_sessions = df_all_sessions.sort_values(['start_time'], ascending=True)
	df_all_sessions.index = range(df_all_sessions.shape[0])

	print 'Dataset processed shape:', df_all_sessions.shape 
	df_all_sessions.to_csv('../Dataset/last_fm_absence.csv')
	return df_all_sessions













def split_train_val_test(data, is_absence=''):

### Split for a given ratio
	RATIO_TRAIN = .8
	RATIO_VAL   = .1


	N = data.shape[0]

	data_train      = data[:int(RATIO_TRAIN*N)]
	data_val        = data[int(RATIO_TRAIN*N):int( (RATIO_TRAIN+RATIO_VAL)*N) ]
	data_test       = data[int((RATIO_TRAIN+RATIO_VAL)*N): ]
	data_val.index  = range(data_val.shape[0])
	data_test.index = range(data_test.shape[0])

	#print data_val.start_time.head()
	#print data_test.start_time.head()
	
	print 'Train shape: ', data_train.shape
	print 'Val shape: ',   data_val.shape
	print 'Test shape : ', data_test.shape

	print data_train.columns


### Only keep users in test which appear in train
	listeners_id_train = pd.unique(data_train['listener_id'])
	listeners_id_val   = pd.unique(data_val[  'listener_id'])
	listeners_id_test  = pd.unique(data_test[ 'listener_id'])

	for idx in set(list(listeners_id_val) + list(listeners_id_test)):
	    if idx not in listeners_id_train:
	    	print 'Listener '+str(idx)+' not in train' 
	        data_user_test = data_test[data_test['listener_id'] == idx]
	        data_test      = data_test.drop(data_user_test.index)
	        
	        data_user_val  = data_val[data_val['listener_id'] == idx]
	        data_val       = data_val.drop(data_user_val.index)

	        #print data_user_test.shape, data_user_val.shape



### Drop times
	#print data_test.shape, data_train.shape
	#listeners_id_val   = pd.unique(data_val[  'listener_id'])
	#listeners_id_test  = pd.unique(data_test[ 'listener_id'])
	#print len(listeners_id_val), len(listeners_id_test)

	data_train = data_train.drop(['start_time', 'end_time'], axis=1)
	data_val   = data_val.drop([  'start_time', 'end_time'], axis=1)
	data_test  = data_test.drop([ 'start_time', 'end_time'], axis=1)

### Save
	if is_absence=='absence':
		data_train.to_csv('../Dataset/last_fm_absence_train'+'.csv', index=False)
		data_val.to_csv(  '../Dataset/last_fm_absence_val'+'.csv',   index=False)
		data_test.to_csv( '../Dataset/last_fm_absence_test'+'.csv',  index=False)
	else:
		data_train.to_csv('../Dataset/last_fm_train'+'.csv', index=False)
		data_val.to_csv(  '../Dataset/last_fm_val'+'.csv',   index=False)
		data_test.to_csv( '../Dataset/last_fm_test'+'.csv',  index=False)


#data  = process_last_fm()
data = pd.read_csv('../Dataset/last_fm.csv')
split_train_val_test(data)

#data  = process_last_fm_absence()
#split_train_val_test(data, is_absence='absence')



		 