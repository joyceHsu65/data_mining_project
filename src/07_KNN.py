#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:30:36 2025

@author: 409383712 徐胤瑄
"""
import pandas as pd
parking = pd.read_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/preprocessing3_taipei_paring_lot_availble.csv")
parking.drop(["id"], axis=1, inplace=True)
parking.info()


# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ohe = OneHotEncoder(sparse_output=False)
area = ohe.fit_transform(parking[["area"]])
area = pd.DataFrame(area)
area.columns = ohe.categories_[0]

parking_fare_classification = ohe.fit_transform(parking[["parking_fare_classification"]])
parking_fare_classification = pd.DataFrame(parking_fare_classification)
parking_fare_classification.columns = ohe.categories_[0]

# 標準化數值欄位
numeric_features = [
    'totalcar', 'totalmotor', 'totalbike', 'Pregnancy_First', 'Handicap_First',
    'totallargemotor', 'ChargingStation',
    'farecar_weekday', 'farecar_night', 'farecar_haliday',
    'farecar_month', 'faremotor_day/hour', 'faremotor_month', 'largemotor_month'
]

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(parking[numeric_features])
numeric_scaled = pd.DataFrame(numeric_scaled, columns=[col + "_std" for col in numeric_features])

# 取出其餘已處理好的數值欄位
other_features = pd.DataFrame([parking["type"], parking["serviceTime"]]).T

# 合併所有欄位
X = pd.concat([area, other_features, numeric_scaled, parking_fare_classification], axis=1)
y = parking["type2"]

# 切分訓練與測試集 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20250610)

X.to_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/X_KNN.csv", index=False, encoding="utf_8_sig")
#===================================================
# KNN: 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
acc = []
for i in range(1, 1361): 
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred)) 

bestK = 0
for i in range(0,1360):
    if acc[i] == max(acc): 
        bestK = i + 1

print("最佳的 K = ",bestK, "資料集的測試正確率 = ", max(acc))

knn_best = KNeighborsClassifier(n_neighbors=bestK)
knn_best.fit(X_train, y_train)
train_pred = knn_best.predict(X_train)
print("使用最佳 K =",bestK, "的訓練資料正確率 = ", accuracy_score(y_train, train_pred))