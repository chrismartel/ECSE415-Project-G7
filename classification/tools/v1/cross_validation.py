from random import shuffle
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
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
features: dictionary where keys are the image sequences ids and values are numpy arrays of features.
labels: dictionary where keys are the image sequences ids and value are numpy arrays of labels (1 for vehicle, 0 for non-vehcile)
K: number of folds. Must be a value between 1 and number of sequences.
C: parameter used for SVM training.
Return accuracies, recalls, precisions and inference execution time for each fold.
"""
def cross_validation(features, labels, C=10, bagging=False, n_estimators=10):
  # 3-Fold Cross Validation
  K=3

  # Performance measures
  tp, fp, tn, fn = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)

  accuracies, recalls, precisions, inference_execution_times =  np.zeros(K),  np.zeros(K),  np.zeros(K),  np.zeros(K)

  for k in range(K):
    # val dataset
    x_val, y_val = features[k], labels[k]

    # train dataset
    train_range = list(range(0,k)) + list(range(k+1,K))
    x_train, y_train = None, None
    for i in train_range:
      if x_train is None:
        x_train = features[i]
      else:
        x_train = np.concatenate((x_train, features[i]), axis=0)

      if y_train is None:
        y_train = labels[i]
      else:
        y_train = np.concatenate((y_train, labels[i]), axis=0)

    # Bagging classifier
    if bagging:
      clf = BaggingClassifier(base_estimator=svm.SVC(gamma='scale', C=C), n_estimators=n_estimators, random_state=None)

    # SVM Binary Classifier
    else:
      clf = svm.SVC(gamma='scale', C=C)
          
    # Train
    clf.fit(x_train, y_train)

    # Predict
    start = time.time()
    y_pred = clf.predict(x_val)
    end = time.time()
    inference_execution_times[k] = (end-start)

    # Performance measurements
    tp[k] = np.sum(np.logical_and(y_pred == 1, y_val == 1))
    fp[k] = np.sum(np.logical_and(y_pred == 1, y_val == 0))
    tn[k] = np.sum(np.logical_and(y_pred == 0, y_val == 0))
    fn[k] = np.sum(np.logical_and(y_pred == 0, y_val == 1))

    accuracies[k] = accuracy(tp[k], fp[k], tn[k], fn[k])
    recalls[k] = recall(tp[k], fn[k])
    precisions[k] = precision(tp[k], fp[k])

  return accuracies, recalls, precisions, inference_execution_times