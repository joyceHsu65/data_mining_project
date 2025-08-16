#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:49:30 2025

@author: 409383712 徐胤瑄
"""
import pandas as pd
parking = pd.read_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/preprocessing3_taipei_paring_lot_availble.csv")
parking.drop(["id"], axis=1, inplace=True)
parking.info()

X_DT = pd.read_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/X_DT.csv")
X_SVM = pd.read_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/X_SVM.csv")
X_RF = pd.read_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/X_RF.csv")
X_KNN = pd.read_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/X_KNN.csv")

y = parking["type2"]

from sklearn.model_selection import train_test_split
X_train_DT, X_test_DT, y_train_DT, y_test_DT = train_test_split(X_DT, y, test_size=0.2, random_state=20250610)
X_train_SVM, X_test_SVM, y_train_SVM, y_test_SVM = train_test_split(X_SVM, y, test_size=0.2, random_state=20250610)
X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(X_RF, y, test_size=0.2, random_state=20250610)
X_train_KNN, X_test_KNN, y_train_KNN, y_test_KNN = train_test_split(X_KNN, y, test_size=0.2, random_state=20250610)

#===================================================


# DT
from sklearn.tree import DecisionTreeClassifier
clf1=DecisionTreeClassifier(criterion="entropy", min_samples_split=0.05, min_samples_leaf=3, random_state=20250610)
clf1.fit(X_train_DT, y_train_DT)
print("全變數的entropy「訓練」正確率=", clf1.score(X_train_DT, y_train_DT))
print("全變數的entropy挑選變數重要性=", clf1.feature_importances_)
print("全變數的entropy「測試」正確率=", clf1.score(X_test_DT, y_test_DT))
feature=pd.DataFrame([X_train_DT.columns,clf1.feature_importances_]).T
    # ChargingStation, farecar_weekday, type, serviceTime, parking_fare_classification
X_train_DT_new=pd.DataFrame([X_train_DT["ChargingStation"], X_train_DT["farecar_weekday"], X_train_DT["type"], X_train_DT["serviceTime"], X_train_DT["parking_fare_classification"]]).T
X_test_DT_new=pd.DataFrame([X_test_DT["ChargingStation"], X_test_DT["farecar_weekday"], X_test_DT["type"], X_test_DT["serviceTime"], X_test_DT["parking_fare_classification"]]).T
clf1.fit(X_train_DT_new, y_train_DT)
print("模型建構top5的entropy「訓練」正確率=", clf1.score(X_train_DT_new, y_train_DT))
print("模型建構top5的entropy「測試」正確率=", clf1.score(X_test_DT_new, y_test_DT))


# SVC
from sklearn.svm import SVC
svc = SVC(C=1, gamma=0.1, kernel="rbf", class_weight="balanced", probability=True)
svc.fit(X_train_SVM, y_train_SVM)
print("SVC的訓練正確率 =", svc.score(X_train_SVM, y_train_SVM))
print("SVC的測試正確率 =", svc.score(X_test_SVM, y_test_SVM))

# RF
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=20250610)
RF.fit(X_train_RF, y_train_RF)
print("RF的訓練正確率 =", RF.score(X_train_RF, y_train_RF))
print("RF的測試正確率 =", RF.score(X_test_RF, y_test_RF))

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_KNN, y_train_KNN)
print("KNN的訓練正確率 =", knn.score(X_train_KNN, y_train_KNN))
print("KNN的測試正確率 =", knn.score(X_test_KNN, y_test_KNN))

# Voting
from sklearn.ensemble import VotingClassifier
vot1 = VotingClassifier(estimators=[("RF", RF), ("KNN", knn), ("SVC", svc)], voting="soft", n_jobs=-1)
vot1.fit(X_train_KNN, y_train_KNN)
print("Voting soft 的訓練正確率 =", vot1.score(X_train_KNN, y_train_KNN))
print("Voting soft 的測試正確率 =", vot1.score(X_test_KNN, y_test_KNN))

vot2 = VotingClassifier(estimators=[("RF", RF), ("KNN", knn), ("SVC", svc)], voting="hard", n_jobs=-1)
vot2.fit(X_train_KNN, y_train_KNN)
print("Voting hard 的訓練正確率 =", vot2.score(X_train_KNN, y_train_KNN))
print("Voting hard 的測試正確率 =", vot2.score(X_test_KNN, y_test_KNN))

