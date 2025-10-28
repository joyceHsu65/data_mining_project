# src/05_svm.py
"""
Support Vector Machine (SVM) for type2 分類
===========================================

目的：
- 使用 SVM（Linear 與 RBF kernel）預測停車場經營型態。
- 比較不同 C 值對準確率的影響。

輸入：
- data/processed/preprocessing3_taipei_paring_lot_availble.csv

輸出：
- results/05_X_SVM.csv      # Voting 使用的特徵集
- 終端輸出：各 C 值之訓練與測試準確率。

主要步驟：
1) OneHotEncoder（area, parking_fare_classification）
2) StandardScaler 標準化數值特徵
3) 使用 LinearSVC 與 SVC (RBF kernel, gamma=0.1)
4) 迴圈測試 C ∈ {0.01, 0.1, 1, 10}
5) 輸出最佳模型的訓練與測試結果

建議執行：
- python src/05_svm.py
"""

import pandas as pd
parking = pd.read_csv("data/processed/preprocessing3_taipei_paring_lot_availble.csv")

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

X.to_csv("results/05_X_SVM.csv", index=False, encoding="utf_8_sig")
#=========================================================
# 建模 SVC
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# 固定參數 C
C_values = [0.01, 0.1, 1, 10]
fixed_gamma = 0.1

# LinearSVC 部分
for c in C_values:
    model_linear = LinearSVC(C=c, dual=False, class_weight="balanced")
    model_linear.fit(X_train, y_train)
    
    # 訓練集
    train_acc = model_linear.score(X_train, y_train)
    y_train_pred = model_linear.predict(X_train)
    train_err = (y_train != y_train_pred).sum()
    
    # 測試集
    test_acc = model_linear.score(X_test, y_test)
    
    print(f"[LinearSVC] C={c:<5} → 訓練正確率={train_acc:.3f}, 分錯={train_err}, 測試正確率={test_acc:.3f}")


# SVC (RBF 核心) 多組 gamma
for c in C_values:
    model_rbf = SVC(C=c, gamma=fixed_gamma, kernel="rbf", class_weight="balanced", probability=True)
    model_rbf.fit(X_train, y_train)

    # 訓練集
    train_acc = model_rbf.score(X_train, y_train)
    y_train_pred = model_rbf.predict(X_train)
    train_err = (y_train != y_train_pred).sum()

    # 測試集
    test_acc = model_rbf.score(X_test, y_test)
    
    print(f"[SVC-RBF]  C={c:<5} → 訓練正確率={train_acc:.3f}, 分錯={train_err}, 測試正確率={test_acc:.3f}")

