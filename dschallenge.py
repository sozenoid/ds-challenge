def from_network_params_return_accuracy(reg1, reg2, lr):
	import time
	from keras.models import Sequential
	from keras.layers import Dense
	from tensorflow.keras import regularizers
	from tensorflow import keras

	import matplotlib.pyplot as plt
	import pandas as pd
	import numpy as np 
	import eli5
	from eli5.sklearn import PermutationImportance
	from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
	'''
	This function builds a predictive model to predict whether a vessel will be involved in an incident in the next 12 months. 
	The predictor is a neural network with a logistic regression acting as a benchmark 
	PRE: Takes as argument 
		reg1: The Lasso regularisatin parameter to train a neural network 
		reg2: The Ridge regularization paramter to train a neural network
		lr  : The learning rate of the neural network 
		
	POST: 	The method will pre process the input data ./nav_inc_data.pkl, dropping null fields and train a neural network and a logistic regression on this data. 
		The feature detection uses the Mean Decrease Accuracy algorithm. 
	'''
	marker = "#"*10
	# import and do a sanity check of the data
	print(marker, "Loading and inspecting the data...")
	ds = pd.read_pickle("./nav_inc_data.pkl")
	# print(ds.head()) 
	# print(ds.info())
	# print(sum(ds["nav_incident"])) 

	    
	# Normalization
	print(marker,"Removal of undesired fields")
	columns_to_drop = ['enc_flag', 'enc_sreg', 'enc_cport', 'Management Risk', 'Owner Risk', 'Previous Years Number of Claims', 
	 					'Previous Years Number of Claims of Interest',  'safety_deficiencies_oDay'] 
	print(marker,"Dropping the following fields: ", columns_to_drop)
	ds_normalized = ds.drop(columns=columns_to_drop).copy(True) # removing the columns that don't contain useful information and those with no nonzero entries

	print(marker,"Data normalization, zero centering and division by the standard deviation of each field")
	ds_normalized.iloc[:,0:-1] = ds_normalized.iloc[:,0:-1].apply(lambda x: (x-x.mean())/ x.std(), axis=0) # so we normalize them to have zero mean and std deviation of 1
	print(ds_normalized.info())
	
	# Model structure definition
	def regression_model(): 
		model = Sequential()
		model.add(Dense(12, input_dim=(ds_normalized.shape[1]-1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=reg1, l2=reg2)))
		model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=reg1, l2=reg2)))
		model.add(Dense(1, activation='sigmoid'))

		opt = keras.optimizers.Adam(learning_rate=lr)

		model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
		return model 
	
	# Training
	# split in training, optimization and validation datasets
	split = 0.2
	print(marker,"Splitting the data in training ({}%) and validation ({}%) datasets".format(100*(1-split), 100*split))
	ds_normalized = ds_normalized.sample(frac=1).reset_index(drop=True)
	n_train, n_eval = int(np.floor(len(ds_normalized)*(1-split))), int(np.floor(len(ds_normalized)*split))

	X_train, y_train  = ds_normalized.iloc[:n_train, 0:-1], ds_normalized.iloc[:n_train,   -1]
	X_eval, y_eval = ds_normalized.iloc[n_train:n_train+n_eval, 0:-1], ds_normalized.iloc[n_train:n_train+n_eval,   -1]
	X_all, y_all =  ds_normalized.iloc[:,0:-1], ds_normalized.iloc[:,-1]
	
	# Training step
	nepochs = 100
	print(marker, "Training a neural network classifier, this may take some time ({} epochs)".format(nepochs))
	estimator = KerasRegressor(build_fn=regression_model, validation_data=(X_eval, y_eval), batch_size=10, epochs=nepochs, verbose=0)
	t1 = time.time() 
	history = estimator.fit(X_train, y_train)
	t2 = time.time() 
	print(marker, "The neural network training took {0:.2f} seconds ({1} epochs)".format(t2-t1, nepochs))
	
	# Performance
	print(marker, "Accuracy of the neural network classifier on the validation data:  {0:2.2f}%".format(100*history.history['val_accuracy'][-1]))
	# print(history.history.keys())
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
	
	
	from sklearn.linear_model import LogisticRegression
	clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
	print(marker, "Sanity check: accuracy of a logistic regression binary classifier: {0:2.2f}%".format(100*clf.score(X_eval, y_eval)))



	# Identification of relevant features
	print(marker, "Starting feature importance analysis using the permutation importance algorithm, this may take some time")
	t3 = time.time()
	perm = PermutationImportance(estimator, random_state=1).fit(X_all,y_all, verbose=0)
	t4 = time.time()
	print(marker, "Permutation importance algorithm took {0:2.2f} seconds".format(t4-t3))
	# eli5.show_weights(perm, feature_names = X_eval.columns.tolist())
	print(marker, "Most important feature is 'Current Year Number of Claims'")
	print(eli5.format_as_text(eli5.explain_weights(perm, feature_names = X_train.columns.tolist())))


def param_optimiser():
	"""
	This function runs a scan of relevant parameters to train the neural network of the function 'from_network_params_return_accuracy'
	"""
	reg1 = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
	reg1 = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
	lr =   [1e-4, 3e-4]

	for r1 in reg1:
		for r2 in reg2:
			for l in lr:
				acc = from_network_params_return_accuracy(r1, r2, l)
				with open("accuracy_file_resume", "a") as a:
					a.write("{}\t{}\t{}\t{}\n".format(acc, r1, r2, l))
if __name__=="__main__":
	# param_optimiser()
	from_network_params_return_accuracy(3e-4, 3e-5, 0.0001)
