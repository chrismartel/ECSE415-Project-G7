from random import shuffle
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import gc


# Performance Measures
def accuracy(tp, fp, tn, fn):
  return 100*(tp+tn)/(tp+fp+tn+fn)

def precision(tp, fp):
  return 100*tp/(tp+fp)

def recall(tp, fn):
  return 100*tp/(tp+fn)

"""
Perform 3-fold cross-validation for training SVM classifier for vehicle detection.

Params
------
fold_features: dicitonary of features used for K-fold cross-validation
fold_labels: dicitionary of labels used for K-fold cross-validation

other_features: additional features used for training
other_labels: additional labels used for training

I: number of iterations of cross-validation for each fold

classifier_type: possible values 'linear', 'rbf', 'rf'

C: regularization parameter for SVMs.

bagging: indicate if use bagging or not
n_estimators: # of estimator sin bagging

Return
------
Return accuracies, recalls, precisions and inference execution time for each fold.
"""
def cross_validation(fold_features, fold_labels, other_features=None, other_labels=None,classifier_type='linear', K=3, C=10, gamma='scale', bagging=False, n_estimators=10, n_estimators_random_forest = 500, criterion = 'entropy', max_depth = None, min_samples_split = 2):
  fold_ids = list(fold_features.keys())

  # Performance measures
  tp, fp, tn, fn = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
  accuracies, recalls, precisions, inference_execution_times =  np.zeros(K),  np.zeros(K),  np.zeros(K),  np.zeros(K)

  if other_features is not None:
    number_of_other_features = other_features.shape[0]
    other_inds = list(range(number_of_other_features))
    shuffle(other_inds)

  
  for k in range(K): # number of folds

    # val set
    x_val, y_val = fold_features[fold_ids[k]], sequence_labels[fold_ids[k]]

    # train set
    x_train, y_train = None, None
    train_ids = list()
    if k == 0:
      train_ids += fold_ids[k+1:]
    elif k == K-1:
      train_ids += fold_ids[:k]
    else:
      train_ids += fold_ids[:k]
      train_ids += fold_ids[k+1:]

    for train_id in train_ids:
      if x_train is None:
        x_train, y_train = fold_features[train_id], fold_labels[train_id]
      else:
        x_train, y_train = np.concatenate((x_train, fold_features[train_id]),axis=0), np.concatenate((y_train, fold_labels[train_id]),axis=0)

    # Add other set
    if other_features is not None:
      x_train, y_train = np.concatenate((x_train, other_features),axis=0), np.concatenate((y_train, other_labels),axis=0)

    gc.collect()

    # RBF SVM
    if classifier_type == 'rbf':
      # Bagging classifier
      if bagging:
        clf = BaggingClassifier(base_estimator=svm.SVC(gamma=gamma, C=C), n_estimators=n_estimators, random_state=None)
      # Binary Classifier
      else:
        clf = svm.SVC(gamma=gamma, C=C)
    
    # LINEAR SVM
    elif classifier_type == 'linear':
      # Bagging classifier
      if bagging:
        clf = BaggingClassifier(base_estimator=svm.LinearSVC(C=C), n_estimators=n_estimators, random_state=None)
      # Binary Classifier
      else:
        clf = svm.LinearSVC(C=C)

    # Random Forest
    elif classifier_type == 'rf':
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
  return (accuracies, recalls, precisions, inference_execution_times)