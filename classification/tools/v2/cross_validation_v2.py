from random import shuffle
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np

# Performance Measures

def accuracy(tp, fp, tn, fn):
  return 100*(tp+tn)/(tp+fp+tn+fn)

def precision(tp, fp):
  return 100*tp/(tp+fp)

def recall(tp, fn):
  return 100*tp/(tp+fn)

"""
Perform 3-fold cross-validation for training SVM classifier for vehicle detection.
features: numpy array containing the hog features of size N x F where N is the number of samples
          and F the number of HoG features.
labels: (numpy array containing the labels associated with each feature vector.

C: parameter used for SVM training.
classifier_type: svm rbf or random forest classifier used
Return accuracies, recalls, precisions and inference execution time for each fold.
"""
def cross_validation_v2(features, labels, classifier_type='svm', C=10, bagging=False, n_estimators=10, n_estimators_random_forest = 500, criterion = 'entropy', max_depth = None, min_samples_split = 2):
  # 3-Fold Cross Validation
  K=3

  # Performance measures
  tp, fp, tn, fn = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)

  accuracies, recalls, precisions, inference_execution_times =  np.zeros(K),  np.zeros(K),  np.zeros(K),  np.zeros(K)

  N = features.shape[0]
  inds = list(range(N))
  shuffle(inds)

  interval = int(N/K)
  
  for k in range(K):
    # val dataset
    x_val, y_val = features[inds[k*interval:(k+1)*interval]], labels[inds[k*interval:(k+1)*interval]]

    # train dataset
    if k == 0:
        x_train, y_train = features[inds[(k+1)*interval:]], labels[inds[(k+1)*interval:]]
    elif k == K-1:
        x_train, y_train = features[inds[:k*interval]], labels[inds[:k*interval]]
    else:
        x_train1, y_train1 = features[inds[:k*interval]], labels[inds[:k*interval]]
        x_train2, y_train2 = features[inds[(k+1)*interval:]], labels[inds[(k+1)*interval:]]
        x_train = np.concatenate((x_train1, x_train2), axis=0)
        y_train = np.concatenate((y_train1, y_train2), axis=0)

    # SVM
    if classifier_type == 'svm':
      # Bagging classifier
      if bagging:
        clf = BaggingClassifier(base_estimator=svm.SVC(gamma='scale', C=C), n_estimators=n_estimators, random_state=None)

      # SVM Binary Classifier
      else:
        clf = svm.SVC(gamma='scale', C=C)
    
    # Random Forest
    else:
      # Bagging classifier
      if bagging:
        clf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=n_estimators_random_forest, criterion=criterion), n_estimators=n_estimators, random_state=None)

      # Random Forest Classifier
      else:
        clf = RandomForestClassifier(n_estimators=n_estimators_random_forest, criterion=criterion)
          
    # Train
    clf.fit(x_train, y_train)

    # Predict
    start = time.time()
    y_pred = clf.predict(x_val)
    end = time.time()
    inference_execution_times[k] = (end-start)/x_val.shape[0]

    # Performance measurements
    tp[k] = np.sum(np.logical_and(y_pred == 1, y_val == 1))
    fp[k] = np.sum(np.logical_and(y_pred == 1, y_val == 0))
    tn[k] = np.sum(np.logical_and(y_pred == 0, y_val == 0))
    fn[k] = np.sum(np.logical_and(y_pred == 0, y_val == 1))

    accuracies[k] = accuracy(tp[k], fp[k], tn[k], fn[k])
    recalls[k] = recall(tp[k], fn[k])
    precisions[k] = precision(tp[k], fp[k])

  return accuracies, recalls, precisions, inference_execution_times