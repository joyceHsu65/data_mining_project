#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 18:58:29 2025

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

#===================================================
# kmeans
from sklearn.cluster import KMeans
SSE = []
# 找出最合適群數
for i in range(10):
    kmeans = KMeans(n_clusters=i+1, init="k-means++", random_state=20250610)
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)#SSE是群內誤差
print("SSE 1~10 群 =", SSE)

import matplotlib.pyplot as plt
plt.plot(range(1,11), SSE, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")

# 分四群(期末固定分四群)
kmeans1 = KMeans(n_clusters=4, init="k-means++", random_state=20250610)
kmeans1.fit(X)
print("SSE 4群 =", kmeans1.inertia_)
print("每一群每個變數的質心 =", kmeans1.cluster_centers_)

# 分五群（從陡坡圖選下降速度何時開始趨緩）
kmeans2 = KMeans(n_clusters=5, init="k-means++", random_state=20250610)
kmeans2.fit(X)
print("SSE 5群 =", kmeans2.inertia_)
print("每一群每個變數的質心 =", kmeans2.cluster_centers_)

# 分五群的輪廓結果
centroid = pd.DataFrame(kmeans2.cluster_centers_, columns=X.columns)
X_pred = kmeans2.predict(X)
print(pd.crosstab(y, X_pred))
print("用分群來預測分類的正確率為 ＝",(1115+6+143+2+63)/1701) # 正確率為 ＝ 0.7813

# 結果
if kmeans2.inertia_ < kmeans1.inertia_:
    print("K=5 的 SSE 較小，符合 elbow point，是較佳的群數")
else:
    print("K=4 的 SSE 較小，可考慮使用 K=4")

