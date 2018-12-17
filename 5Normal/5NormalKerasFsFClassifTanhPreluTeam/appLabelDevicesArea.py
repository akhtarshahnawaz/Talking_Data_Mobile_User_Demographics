import pandas as pd
import numpy as np
import os
np.random.seed(0)

import xgboost as xgb
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2,SelectFromModel
from sklearn.metrics import log_loss
import pickle
from sklearn.cross_validation import KFold,StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

#------------------------------------------------- Functions ---------------------------------------------------

def batch_generator(X, y, batch_size, shuffle):
	#chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
	number_of_batches = np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	if shuffle:
		np.random.shuffle(sample_index)
	while True:
		batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
		X_batch = X[batch_index,:].toarray()
		y_batch = y[batch_index]
		counter += 1
		yield X_batch, y_batch
		if (counter == number_of_batches):
			if shuffle:
				np.random.shuffle(sample_index)
			counter = 0

def batch_generatorp(X, batch_size, shuffle):
	number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
	counter = 0
	sample_index = np.arange(X.shape[0])
	while True:
		batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
		X_batch = X[batch_index, :].toarray()
		counter += 1
		yield X_batch
		if (counter == number_of_batches):
			counter = 0

#------------------------------------------------ Functions ---------------------------------------------------

##################
#   Read Data
##################
appLabelDeviceTrain = pickle.load(open("../generated/appLabelDeviceTrain.p","r"))
areaTrain = pickle.load(open("../generated/areaTrain.p","r"))
train = hstack((appLabelDeviceTrain,areaTrain), format='csr')

appLabelDeviceTest = pickle.load(open("../generated/appLabelDeviceTest.p","r"))
areaTest = pickle.load(open("../generated/areaTest.p","r"))
test = hstack((appLabelDeviceTest,areaTest), format='csr')

target = pickle.load(open("../generated/group.p","r"))
device_id = pickle.load(open("../generated/device_id.p","r"))

trainDevices = pd.read_csv("../../../data/gender_age_train.csv", usecols =["device_id"])
indexes = pd.read_csv("../generated/raddarIndices.csv")
indexes = pd.merge(trainDevices, indexes, how="left", on="device_id", left_index=True).reset_index().drop(["index"], axis=1)
######################
#   Feature Selection
######################
fs = SelectPercentile(f_classif, percentile=23).fit(train, target)
train = fs.transform(train)
test = fs.transform(test)

##################
#   Pre Processing
##################
targetEncoder = LabelEncoder()
target = targetEncoder.fit_transform(target)
skfTarget = target.copy()
target = np_utils.to_categorical(target)

##################
#  Build Model
##################
def modelBuilder():
	model = Sequential()
	model.add(Dense(200, input_dim=train.shape[1], init='normal', activation ="tanh"))
	model.add(Dropout(0.4))
	model.add(Dense(70, input_dim=150, init='normal'))
	model.add(PReLU())
	model.add(Dropout(0.2))
	model.add(Dense(12, init='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])  #logloss
	return model
 
def runKeras(x_train,x_test,y_train,y_test,test):
	clf = modelBuilder()

	tmpModel = "tmpWeightappLabelDeviceArea.hdf5"
	early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose =1)
	checkpointer = ModelCheckpoint(filepath= tmpModel , monitor='val_loss', verbose=1, save_best_only=True)

	fit= clf.fit_generator(generator=batch_generator(x_train, y_train, 100, True),nb_epoch=100,samples_per_epoch=(train.shape[0]/100)*100,validation_data=(x_test.todense(), y_test), verbose=2,callbacks=[early_stopping,checkpointer])

	clf = load_model(tmpModel)

	cv_y_test = clf.predict_generator(generator=batch_generatorp(x_test, 400, False), val_samples=x_test.shape[0])
	real_y_test = clf.predict_generator(generator=batch_generatorp(test, 400, False), val_samples=test.shape[0])
	# model.save_weights('weights/model6Weights2.h5')
	os.unlink(tmpModel)
	return cv_y_test,real_y_test

##################
#  Run Model
##################
print "Training"
folds = 5
trainStack = np.zeros((train.shape[0], 12))
testStack = 0
indices = np.zeros(train.shape[0])
losses = []
print test.shape
# kf = StratifiedKFold(skfTarget,n_folds =folds, shuffle =True, random_state = 0)
for i in range(0,5):
	train_index = indexes[indexes.idx != i].index.values
	test_index = indexes[indexes.idx == i].index.values

	print "Predicting Fold: ",i,
	x_train , y_train = train[train_index], target[train_index]
	x_test, y_test = train[test_index], target[test_index]

	cv_y_test, real_y_test = runKeras(x_train,x_test,y_train,y_test,test)
	trainStack[test_index] = cv_y_test
	indices[test_index]=i
	
	losses.append(log_loss(y_test, trainStack[test_index]))
	print losses[-1]

	testStack += real_y_test

CVLogLoss = log_loss(target, trainStack)
print "Overall LogLoss on Whole Train CV", CVLogLoss

testStack /= folds

trainStack =pd.DataFrame(trainStack, columns =targetEncoder.classes_)
testStack = pd.DataFrame(testStack, columns =targetEncoder.classes_)

testStack["device_id"] = device_id
pd.DataFrame({"device_id":trainDevices.device_id.values, "idx":indices.astype(int)}).to_csv("indices/myIndices.csv", index=False)

trainStack.to_csv("stacks/train/appLabelDevicesArea.csv", index=False)
testStack.to_csv("stacks/test/appLabelDevicesArea.csv", index=False)

#Generating Report for this Model
f = open("reports/appLabelDevicesArea.txt","wb")
for i,loss in enumerate(losses):
	f.write("Log Loss for round "+str(i)+": "+str(loss)+"\n")
f.write("Overall CV Loss: "+str(CVLogLoss))
f.close()
