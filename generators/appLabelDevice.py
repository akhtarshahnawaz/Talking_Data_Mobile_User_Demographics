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
labelCats = pd.read_csv("../../../data/label_categories.csv") # label_id, category
appLabels = pd.read_csv("../../../data/app_labels.csv")    # app_id, label_id
appEvents = pd.read_csv("../../../data/app_events.csv",usecols =["event_id","app_id"]) # 1GB event_id, app_id, is_installed (remove-- is constant), is_active
events = pd.read_csv("../../../data/events.csv",usecols=["event_id","device_id"])   # 200MB event_id, device_id, timestamp, longitude, latitude
devices = pd.read_csv("../../../data/phone_brand_device_model.csv")    # device_id, phone_brand, device_model
test = pd.read_csv("../../../data/gender_age_test.csv")   # device_id
train = pd.read_csv("../../../data/gender_age_train.csv")  # device_id, gender, age, group
print("# Data Loaded")

##################
#   Preprocessing
##################
print("# Pre Processing")
devices.drop_duplicates('device_id', keep='first', inplace=True)
devices["device_model"] = devices["phone_brand"]+devices["device_model"]

##################
#   App Labels
##################
print("# Read App Labels")

labels = appLabels.groupby("app_id")["label_id"].apply(lambda x: " ".join("Label:"+str(s) for s in x))
del appLabels

##################
#   App Events
##################
print("# Read App Events")

appEvents["labels"] = appEvents["app_id"].map(labels)
labels = appEvents.groupby("event_id")["labels"].apply(lambda x: " ".join(str(s) for s in x))
applications = appEvents.groupby("event_id")["app_id"].apply(lambda x: " ".join("App:"+str(s) for s in x))
del appEvents

##################
#     Events
##################
print("# Read Events")
events["labels"] = events["event_id"].map(labels)
events["applications"] = events["event_id"].map(applications)
labels = events.groupby("device_id")["labels"].apply(lambda x: " ".join(str(s) for s in x))
applications = events.groupby("device_id")["applications"].apply(lambda x: " ".join(str(s) for s in x))
del events

##################
#  Train and Test
##################
print("# Generate Train and Test")

train["labels"] = train["device_id"].map(labels)
train["applications"] = train["device_id"].map(applications)
train = pd.merge(train, devices, how='left', on='device_id', left_index=True)

test["labels"] = test["device_id"].map(labels)
test["applications"] = test["device_id"].map(applications)
test = pd.merge(test, devices, how='left',on='device_id', left_index=True)
del devices
##################
#   Vectorizer
##################
print("# Vectorizing Train and Test")

def vectorize(train, test, columns, fillMissing):
    data = pd.concat((train, test), axis=0, ignore_index=True)
    data = data[columns].astype(np.str).apply(lambda x: " ".join(s for s in x), axis=1).fillna(fillMissing)
    split_len = len(train)

    # TF-IDF Feature
    vectorizer = CountVectorizer(min_df=1, binary=True)
    data = vectorizer.fit_transform(data)

    train = data[:split_len, :]
    test = data[split_len:, :]
    return train, test


# Group Labels
train, test = vectorize(train,test,["phone_brand", "device_model", "labels","applications"],"missing")
pickle.dump(train,open("../generated/appLabelDeviceTrain.p","wb"))
pickle.dump(test,open("../generated/appLabelDeviceTest.p","wb"))

