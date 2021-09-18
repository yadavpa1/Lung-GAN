"""
Created on Mon Feb 26 20:47:21 2021

@author: Lung-GANs
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
import numpy as np
from sklearn.metrics import classification_report

num = 75
style_label_file = './style_names.txt'
target_names = list(np.loadtxt(style_label_file, str, delimiter='\n'))

X_train=np.load('features%d_train.npy'%num)
y_train=np.load('label%d_train.npy'%num)
X_test=np.load('features%d_test.npy'%num)
y_test=np.load('label%d_test.npy'%num)

estimators = [
              ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svm', make_pipeline(StandardScaler(),
                                    LinearSVC(random_state=42)))
]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
model = clf.fit(X_train, y_train)

print("Predicting..")
y_pred = model.predict(X_test)

print("Calculating score..")
score = model.score(X_test, y_test)

print("Accuracy: %.3f" %score)
print("Classification Report")
print(classification_report(y_test, y_pred, target_names=target_names))