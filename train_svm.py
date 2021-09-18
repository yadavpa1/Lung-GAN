# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:47:21 2021

@author: Lung-GANs
"""

from time import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import classification_report

from sklearn import svm
#
import numpy as np
acc = []
nums = [75]
style_label_file = './style_names.txt'
target_names = list(np.loadtxt(style_label_file, str, delimiter='\n'))
for num in nums:
    X_train=np.load('features/features%d_train.npy'%num)
    y_train=np.load('features/label%d_train.npy'%num)
    X_test=np.load('features/features%d_test.npy'%num)
    y_test=np.load('features/label%d_test.npy'%num)

    print("Fitting the classifier to the training set")
    t0 = time()
    C = 1000.0  # SVM regularization parameter
    clf = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    
    print("Predicting...")
    t0 = time()
    y_pred = clf.predict(X_test)
    
    print "Accuracy: %.3f" %(accuracy_score(y_test, y_pred))
    acc.append(accuracy_score(y_test, y_pred))

    print "Classification Report"
    print classification_report(y_test, y_pred, target_names=target_names)

print acc
