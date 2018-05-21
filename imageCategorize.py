# base
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pylab as plt
# model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

aveList = []
L = 1
print('##image_shape_org-------------------------------->>\n')
#for i in range(1,41):
for i in range(1,41):
#    if i%10>=4:
#        continue
    fName = './data/a (' + str(i) + ').jpg'
    a = np.array(Image.open(fName))
    print(a.shape)
    buf = [0,0,0,0]
    for j in range(0,a.shape[0]):
        for k in range(0,a.shape[1]):
            buf[0] += a[j][k][0]
            buf[1] += a[j][k][1]
            buf[2] += a[j][k][2]

    ave = a.shape[0] * a.shape[1]

    buf[0] /= ave
    buf[1] /= ave
    buf[2] /= ave
    buf[3] = L
    
    buf[0] = int(buf[0])
    buf[1] = int(buf[1])
    buf[2] = int(buf[2])
    
    if ((i % 10) == 0):
        L += 1
    aveList.append(buf)

print('##image_list-------------------------------->>\n')
print(aveList)

# Set dataframe
data = pd.DataFrame(aveList, columns=['avg_r', 'avg_g', 'avg_b', 'catagory'])

X = data.iloc[:, 0:3]
y = data.iloc[:, -1]

print('##head-------------------------------->>\n')
print(X.head())
print(y.head())

# Holdout
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=1)

# set pipelines for two different algorithms
pipe_knn_5 = Pipeline([('scl',StandardScaler()),('est',KNeighborsClassifier())])
pipe_logistic = Pipeline([('scl',StandardScaler()),('est',LogisticRegression(random_state=1))])

# optimize the parameters of each algorithms
pipe_knn_5.fit(X_train,y_train.as_matrix().ravel())
pipe_logistic.fit(X_train,y_train.as_matrix().ravel())

accuracy_score(y_train, pipe_knn_5.predict(X_train))
print(f1_score(y_train, pipe_knn_5.predict(X_train), average=None))
accuracy_score(y_train, pipe_logistic.predict(X_train))
print(f1_score(y_train, pipe_logistic.predict(X_train), average=None))
