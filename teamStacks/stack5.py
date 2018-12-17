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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
##################
#   Read Data
##################

train1 = pd.read_csv("../binary/Leaked5NormalKerasFsChi2PreluPreluTeam/stacks/train/appLabelDevicesArea.csv")
train2 = pd.read_csv("../binary/Leaked5NormalKerasFsChi2TanhPreluTeam/stacks/train/appLabelDevicesArea.csv")
train3 = pd.read_csv("../binary/Leaked5NormalKerasFsFClassifPreluPreluTeam/stacks/train/appLabelDevicesArea.csv")
train4 = pd.read_csv("../binary/Leaked5NormalKerasFsFClassifTanhPreluTeam/stacks/train/appLabelDevicesArea.csv")
train5 = pd.read_csv("../binary/Leaked5NormalKerasFullPreluPreluTeam/stacks/train/appLabelDevicesArea.csv")
train6 = pd.read_csv("../binary/Leaked5NormalKerasFullTanhPreluTeam/stacks/train/appLabelDevicesArea.csv")

dfs = [train1, train2, train3, train4, train5,train6]
for df in dfs:
	cols = [str(i)+"_"+str(col) for i,col in enumerate(df.columns.values.tolist())]
	df.columns = cols

train = pd.concat(dfs, axis=1).reset_index().drop(["index"],axis=1)
train["idx"] = train.index/float(train.shape[0])
train = train.as_matrix()

test1 = pd.read_csv("../binary/Leaked5NormalKerasFsChi2PreluPreluTeam/stacks/test/appLabelDevicesArea.csv").drop(["device_id"], axis=1)
test2 = pd.read_csv("../binary/Leaked5NormalKerasFsChi2TanhPreluTeam/stacks/test/appLabelDevicesArea.csv").drop(["device_id"], axis=1)
test3 = pd.read_csv("../binary/Leaked5NormalKerasFsFClassifPreluPreluTeam/stacks/test/appLabelDevicesArea.csv").drop(["device_id"], axis=1)
test4 = pd.read_csv("../binary/Leaked5NormalKerasFsFClassifTanhPreluTeam/stacks/test/appLabelDevicesArea.csv").drop(["device_id"], axis=1)
test5 = pd.read_csv("../binary/Leaked5NormalKerasFullPreluPreluTeam/stacks/test/appLabelDevicesArea.csv").drop(["device_id"], axis=1)
test6 = pd.read_csv("../binary/Leaked5NormalKerasFullTanhPreluTeam/stacks/test/appLabelDevicesArea.csv").drop(["device_id"], axis=1)

dfs = [test1, test2, test3, test4, test5,test6]
for df in dfs:
	cols = [str(i)+"_"+str(col) for i,col in enumerate(df.columns.values.tolist())]
	df.columns = cols

test = pd.concat(dfs, axis=1).reset_index().drop(["index"],axis=1)
test["idx"] = test.index/float(test.shape[0])
test = test.as_matrix()

target = pd.read_csv("../../data/gender_age_train.csv").group
device_id = pd.read_csv("../../data/gender_age_test.csv").device_id

trainDevices = pd.read_csv("../../data/gender_age_train.csv", usecols =["device_id"])
indexes = pd.read_csv("metafeatures/raddarIndices.csv")
indexes = pd.merge(trainDevices, indexes, how="left", on="device_id", left_index=True).reset_index().drop(["index"], axis=1)
 

##################
#   Pre Processing
##################
targetEncoder = LabelEncoder()
target = targetEncoder.fit_transform(target)

##################
#  Build Model
##################

def xgbClassifier(x_train,x_test,y_train,y_test,test):
	dtrain = xgb.DMatrix(x_train, y_train)
	dvalid = xgb.DMatrix(x_test, y_test)

	params = {
	    "objective": "multi:softprob",
	    "num_class": 12,
	    "booster": "gbtree",
	    "eval_metric": "mlogloss",
	    "eta": 0.02,
	    "max_depth": 6,
	    "colsample_bytree": 0.8,
	    "subsample":0.9,
	    "silent": 1,
	    "seed":0
	}
	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	clf = xgb.train(params, dtrain, 2000, evals=watchlist, early_stopping_rounds=30, verbose_eval=True)
	cv_y_test = clf.predict(xgb.DMatrix(x_test), ntree_limit=clf.best_iteration)
	real_y_test = clf.predict(xgb.DMatrix(test), ntree_limit=clf.best_iteration)
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

	cv_y_test, real_y_test = xgbClassifier(x_train,x_test,y_train,y_test,test)
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

trainStack.to_csv("stacks/train/stack5.csv", index=False)
testStack.to_csv("stacks/test/stack5.csv", index=False)

#Generating Report for this Model
f = open("reports/stack5.txt","wb")
for i,loss in enumerate(losses):
	f.write("Log Loss for round "+str(i)+": "+str(loss)+"\n")
f.write("Overall CV Loss: "+str(CVLogLoss))
f.write("Overall CV STD: "+str(np.std(losses)))
f.close()
