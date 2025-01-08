import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# we have to open and read the saved pickled data
data_dict = pickle.load(open('F:/Project/New folder/Sign Language Detection Using Machine Learning/data.pickle', 'rb'))



data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
#now train and test split

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size = .2,shuffle = True,stratify=labels)

print(x_train.shape,x_test.shape)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

import os
print("Saving pickle file in:", os.getcwd())
