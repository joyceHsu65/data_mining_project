# src/06_RF.py
"""
Random Forest Classification for type2
======================================

目的：
- 使用隨機森林預測停車場經營型態。
- 比較不同樹數與深度的模型效能，輸出最佳參數組合。

輸入：
- data/processed/preprocessing3_taipei_paring_lot_availble.csv

輸出：
- results/06_X_RF.csv      # Voting 使用的特徵集
- 終端輸出：各參數組合的訓練與測試準確率。

主要步驟：
1) OneHotEncoder 處理 area 欄位
2) 建立 RandomForestClassifier
3) 測試 n_estimators ∈ {100, 200}, max_depth ∈ {5, 8}
4) 比較準確率並印出結果
5) 儲存特徵集供投票模型使用

建議執行：
- python src/06_RF.py
"""

import pandas as pd
parking = pd.read_csv("data/processed/preprocessing3_taipei_paring_lot_availble.csv")
parking.drop(["id"], axis=1, inplace=True)
parking.info()


# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ohe = OneHotEncoder(sparse_output=False)
area = ohe.fit_transform(parking[["area"]])
area = pd.DataFrame(area)
area.columns = ohe.categories_[0]

# labelEncoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
parking_fare_classification = le.fit_transform(parking["parking_fare_classification"])

# X, y
X = pd.DataFrame([parking["type"], parking["totalcar"], parking["totalmotor"], parking["totalbike"], parking["Pregnancy_First"], parking["Handicap_First"], parking["totallargemotor"], parking["ChargingStation"], parking["serviceTime"], parking["farecar_weekday"], parking["farecar_night"], parking["farecar_haliday"], parking["farecar_month"], parking["faremotor_day/hour"], parking["faremotor_month"], parking["largemotor_month"], parking_fare_classification]).T
X.columns = ["type", "totalcar", "totalmotor", "totalbike", "Pregnancy_First", "Handicap_First", "totallargemotor", "ChargingStation", "serviceTime", "farecar_weekday", "farecar_night", "farecar_haliday", "farecar_month", "faremotor_day_hour", "faremotor_month", "largemotor_month", "parking_fare_classification"]

X_new = pd.concat([X, area], axis=1)

y = parking["type2"]

# 切分訓練與測試集 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=20250610)

X_new.to_csv("results/06_X_RF.csv", index=False, encoding="utf_8_sig")
#=========================================================

# random forest
from sklearn.ensemble import RandomForestClassifier
# 參數組合
n_estimators_list = [100, 200]
max_depth_list = [5, 8]

# 執行模型並列印結果
for n in n_estimators_list:
    for d in max_depth_list:
        RF = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=20250610)
        RF.fit(X_train, y_train)
        
        print(f"隨機森林參數：n_estimators={n}, max_depth={d}")
        print("訓練資料正確率 =", RF.score(X_train, y_train))
        print("測試資料正確率 =", RF.score(X_test, y_test))
        print("-" * 40)