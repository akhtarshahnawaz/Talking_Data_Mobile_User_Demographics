import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pickle

##################
# Loading Dataset
##################

print("# Loading Data")
test = pd.read_csv("../../../data/gender_age_test.csv")   # device_id
train = pd.read_csv("../../../data/gender_age_train.csv")  # device_id, gender, age, group
print("# Data Loaded")


# Group Labels
pickle.dump(train["group"],open("../generated/group.p","wb"))
pickle.dump(train["age"],open("../generated/age.p","wb"))
pickle.dump(train["gender"],open("../generated/gender.p","wb"))
pickle.dump(test["device_id"],open("../generated/device_id.p","wb"))

