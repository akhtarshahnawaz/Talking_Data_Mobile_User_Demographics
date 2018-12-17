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
from sklearn.cluster import KMeans


##################
# Loading Dataset
##################

print("# Loading Data")
labelCats = pd.read_csv("../../../data/label_categories.csv") # label_id, category
appLabels = pd.read_csv("../../../data/app_labels.csv")    # app_id, label_id
appEvents = pd.read_csv("../../../data/app_events.csv",usecols =["event_id","app_id"]) # 1GB event_id, app_id, is_installed (remove-- is constant), is_active
events = pd.read_csv("../../../data/events.csv")   # 200MB event_id, device_id, timestamp, longitude, latitude
devices = pd.read_csv("../../../data/phone_brand_device_model.csv")    # device_id, phone_brand, device_model
test = pd.read_csv("../../../data/gender_age_test.csv")   # device_id
train = pd.read_csv("../../../data/gender_age_train.csv")  # device_id, gender, age, group
print("# Data Loaded")

##################
#   Preprocessing
##################
events["area"] = KMeans(n_clusters = 120, n_init = 50, random_state = 0).fit_predict(events[["longitude","latitude"]])
events["area"] = "Area:"+events["area"].astype(str)

##################
#     Events
##################
print("# Read Events")

area = events.groupby("device_id")["area"].apply(lambda x: " ".join(str(s) for s in x))

del events
##################
#  Train and Test
##################
print("# Generate Train and Test")

train["area"] = train["device_id"].map(area)
test["area"] = test["device_id"].map(area)

##################
#   Vectorizer
##################
print("# Vectorizing Train and Test")

def vectorize(train, test, columns, fillMissing):
    data = pd.concat((train, test), axis=0, ignore_index=True)
    data = data[columns].astype(np.str).apply(lambda x: " ".join(s for s in x), axis=1).fillna(fillMissing)
    split_len = len(train)

    # TF-IDF Feature
    vectorizer = CountVectorizer(min_df=1, binary =True)
    data = vectorizer.fit_transform(data)

    train = data[:split_len, :]
    test = data[split_len:, :]
    return train, test


# Group Labels
train, test = vectorize(train,test,["area"],"missing")

pickle.dump(train,open("../generated/areaTrain.p","wb"))
pickle.dump(test,open("../generated/areaTest.p","wb"))

