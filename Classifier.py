import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import Dataset_preprocessing


# we have to open and read the saved pickled data

data_dict = pickle.load(open('./data.pickle','rb'))
data = np.asanyarray(data_dict['data'])
labels = np.asanyarray(data_dict['labels'])

#The features (input variables) and labels (output targets) 
# are now in a form ready to be used for training the machine learning model.

#now train and test split

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size = .25,shuffle = True,stratify=labels)

print(x_train.shape,x_test.shape)