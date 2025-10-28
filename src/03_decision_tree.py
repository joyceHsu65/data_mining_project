#!/usr/bin/env python3
"""
Decision Tree Classification (CART) for Operator Type (type2)
=============================================================

目的：
- 以決策樹分類停車場經營型態（停管處 vs 非停管處），
- 強調可解釋性與規則萃取。

輸入：
- data/processed/preprocessing3_taipei_paring_lot_availble.csv

輸出（預期）：
- results/03_X_DT.csv                     # Voting 使用的特徵集
- rresults/03_CART_param_grid.png         # 模型參數挑選
- results/03_dt_tree_structure.gv         # 決策樹結構
- 終端輸出：訓練／測試準確率、特徵重要度、交叉驗證結果。

主要步驟：
1) 資料切割：train/test = 80/20
2) 類別編碼：LabelEncoder
3) 等頻分箱（Equal Frequency）處理偏態數值
4) 使用 CART 演算法建立多組模型（Gini / Entropy）
5) 比較各模型表現並繪製決策樹結構
6) 進行交叉驗證並輸出結果

建議執行：
- python src/03_decision_tree.py
"""

import pandas as pd
parking = pd.read_csv("data/processed/preprocessing3_taipei_paring_lot_availble.csv")
parking.info()
parking.drop(["id"], axis=1, inplace=True)


# 離散化 - 等分位數
import numpy as np
print(parking["largemotor_month"].value_counts(dropna=False))
print(np.min(parking["largemotor_month"])) #-1.0
print(np.max(parking["largemotor_month"])) #9000.0
# 停車位
parking["totalcar"] = pd.qcut(parking["totalcar"], q = 3, labels = ["Less", "Medium", "Many"])
parking["totalmotor"] = parking["totalmotor"].apply(
    lambda x: "None" if x == 0 else "Have"
)
parking["totalbike"] = parking["totalbike"].apply(
    lambda x: "None" if x == 0 else "Have"
)
parking["Pregnancy_First"] = parking["Pregnancy_First"].apply(
    lambda x: "None" if x == 0 else "Have"
)
parking["Handicap_First"] = parking["Handicap_First"].apply(
    lambda x: "None" if x == 0 else "Have"
)
parking["totallargemotor"] = parking["totallargemotor"].apply(
    lambda x: "None" if x == 0 else "Have"
)
parking["ChargingStation"] = parking["ChargingStation"].apply(
    lambda x: "None" if x == 0 else "Have"
)
# 費用
parking["farecar_weekday"] = pd.qcut(parking["farecar_weekday"], q = 3, labels = ["Low", "Medium", "High"])
parking["farecar_night"] = parking["farecar_night"].apply(
    lambda x: "Not Provided" if x == -1 else "Provided"
)
parking["farecar_haliday"] = parking["farecar_haliday"].apply(
    lambda x: "Not Provided" if x == -1 else "Provided"
)
parking["farecar_month"] = pd.qcut(parking["farecar_month"], q = 3, labels = ["Low", "Medium", "High"])
parking["faremotor_day/hour"] = parking["faremotor_day/hour"].apply(
    lambda x: "Not Provided" if x == -1 else "Provided"
)
parking["faremotor_month"] = parking["faremotor_month"].apply(
    lambda x: "Not Provided" if x == -1 else "Provided"
)
parking["largemotor_month"] = parking["largemotor_month"].apply(
    lambda x: "Not Provided" if x == -1 else "Provided"
)

#=====================================
# 是否重新編碼? 用什麼方式編碼及理由?

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
area = le.fit_transform(parking[["area"]])
totalcar = le.fit_transform(parking[["totalcar"]])
totalmotor = le.fit_transform(parking[["totalmotor"]])
totalbike = le.fit_transform(parking[["totalbike"]])
Pregnancy_First = le.fit_transform(parking[["Pregnancy_First"]])
Handicap_First = le.fit_transform(parking[["Handicap_First"]])
totallargemotor = le.fit_transform(parking[["totallargemotor"]])
ChargingStation = le.fit_transform(parking[["ChargingStation"]])

farecar_weekday = le.fit_transform(parking[["farecar_weekday"]])
farecar_night = le.fit_transform(parking[["farecar_night"]])
farecar_haliday = le.fit_transform(parking[["farecar_haliday"]])
farecar_month = le.fit_transform(parking[["farecar_month"]])
faremotor_day_hour = le.fit_transform(parking[["faremotor_day/hour"]])
faremotor_month = le.fit_transform(parking[["faremotor_month"]])
largemotor_month = le.fit_transform(parking[["largemotor_month"]])
parking_fare_classification = le.fit_transform(parking[["parking_fare_classification"]])
print(le.classes_)  # 印出所有原始類別（按字典順序排序）


X = pd.DataFrame([area, parking["type"], totalcar, totalmotor, totalbike, Pregnancy_First, Handicap_First, totallargemotor, ChargingStation, parking["serviceTime"], farecar_weekday, farecar_night, farecar_haliday, farecar_month, faremotor_day_hour, faremotor_month, largemotor_month, parking_fare_classification]).T
X.columns = ["area", "type", "totalcar", "totalmotor", "totalbike", "Pregnancy_First", "Handicap_First", "totallargemotor", "ChargingStation", "serviceTime", "farecar_weekday", "farecar_night", "farecar_haliday", "farecar_month", "faremotor_day_hour", "faremotor_month", "largemotor_month", "parking_fare_classification"]
y = parking["type2"]

X.to_csv("results/03_X_DT.csv", index=False, encoding="utf_8_sig")

#=====================================
# 切割80%建模20%測試正確率(請展示程式碼)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.20, random_state=20250610) 

# 選擇C&M
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

C_list = np.round(np.arange(0.05, 1.05, 0.05), 2)  # min_samples_split
M_list = range(2, 11)                              # min_samples_leaf

results = pd.DataFrame(index=C_list, columns=M_list)

for C in C_list:
    for M in M_list:
        clf = DecisionTreeClassifier(criterion="gini", 
                                     min_samples_split=C, 
                                     min_samples_leaf=M, 
                                     random_state=20250610)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.loc[C, M] = round(acc, 4)
        
best_C, best_M = results.stack().astype(float).idxmax()
best_acc = results.stack().astype(float).max()
print(f"最佳參數：C={best_C}, M={best_M}, 準確率={best_acc}")

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.heatmap(results.astype(float), annot=True, cmap="Blues", fmt=".4f")
plt.title("CART Model C vs M Accuracy")
plt.xlabel("M (min_samples_leaf)")
plt.ylabel("C (min_samples_split)")
plt.show()

#=====================================
# 變數挑選方式?與理由?全變數和部分變數結果比較(使用最優化的方式C,M,樹的深度,在統一的random_state,展示程式碼)
#chi-square
from sklearn.feature_selection import SelectKBest, chi2
sk=SelectKBest(chi2, k=5) 
sk.fit(X_train, y_train)
print("The feature selection by chi2=", sk.get_feature_names_out())
X_train_cki=sk.transform(X_train)
X_test_cki=sk.transform(X_test)

clf1=DecisionTreeClassifier(criterion="gini", min_samples_split=0.05, min_samples_leaf=3, random_state=20250610)
clf1.fit(X_train_cki, y_train)
print("卡方分配法gini挑選的變數「訓練」正確率=", clf1.score(X_train_cki, y_train))
print("卡方分配法gini挑選的變數重要性=", clf1.feature_importances_)
print("卡方分配法gini挑選的變數「測試」正確率=", clf1.score(X_test_cki, y_test))


clf2=DecisionTreeClassifier(criterion="entropy", min_samples_split=0.05, min_samples_leaf=3, random_state=20250610)
clf2.fit(X_train_cki, y_train)
print("卡方分配法entropy挑選的變數「訓練」正確率=", clf2.score(X_train_cki, y_train))
print("卡方分配法entropy挑選的變數重要性=", clf2.feature_importances_)
print("卡方分配法entropy挑選的變數「測試」正確率=", clf2.score(X_test_cki, y_test))

# modeled Feature Inportance
clf3=DecisionTreeClassifier(criterion="gini", min_samples_split=0.05, min_samples_leaf=3, random_state=20250610)
clf3.fit(X_train, y_train)
print("全變數的gini「訓練」正確率=", clf3.score(X_train, y_train))
print("全變數的gini挑選變數重要性=", clf3.feature_importances_)
print("全變數的gini「測試」正確率=", clf3.score(X_test, y_test))
feature1=pd.DataFrame([X.columns,clf3.feature_importances_]).T
    # ChargingStation, serviceTime, farecar_weekday, type, farecar_month
X_train_new3=pd.DataFrame([X_train["ChargingStation"], X_train["serviceTime"], X_train["farecar_weekday"], X_train["type"], X_train["farecar_month"]]).T
X_test_new3=pd.DataFrame([X_test["ChargingStation"], X_test["serviceTime"], X_test["farecar_weekday"], X_test["type"], X_test["farecar_month"]]).T
clf3.fit(X_train_new3,y_train)
print("模型建構top5的gini「訓練」正確率=", clf3.score(X_train_new3,y_train))
print("模型建構top5的gini「測試」正確率=", clf3.score(X_test_new3,y_test))

clf4=DecisionTreeClassifier(criterion="entropy", min_samples_split=0.05, min_samples_leaf=3, random_state=20250610)
clf4.fit(X_train, y_train)
print("全變數的entropy「訓練」正確率=", clf4.score(X_train, y_train))
print("全變數的entropy挑選變數重要性=", clf4.feature_importances_)
print("全變數的entropy「測試」正確率=", clf4.score(X_test, y_test))
feature2=pd.DataFrame([X.columns,clf4.feature_importances_]).T
    # ChargingStation, farecar_weekday, type, serviceTime, parking_fare_classification
X_train_new4=pd.DataFrame([X_train["ChargingStation"], X_train["farecar_weekday"], X_train["type"], X_train["serviceTime"], X_train["parking_fare_classification"]]).T
X_test_new4=pd.DataFrame([X_test["ChargingStation"], X_test["farecar_weekday"], X_test["type"], X_test["serviceTime"], X_test["parking_fare_classification"]]).T
clf4.fit(X_train_new4, y_train)
print("模型建構top5的entropy「訓練」正確率=", clf4.score(X_train_new4, y_train))
print("模型建構top5的entropy「測試」正確率=", clf4.score(X_test_new4, y_test))

#=====================================
# 建模(C,M,樹的深度,最佳參數)和測試正確率(共8結果)&建議用哪一棵樹？提供 size 和 葉子數？理由？
# 選擇「模型建構top5的entropy」
print("Leaves of the tree", clf4.get_n_leaves())
print("Depth of the tree", clf4.get_depth())

#=====================================================
# 各類別決策樹法則（每個類別至少一條，必要時 Resample）
from sklearn.tree import export_text
tree_rules = export_text(clf4, feature_names=list(X_train_new4.columns))
print(tree_rules)

from sklearn import tree
import graphviz
class_names = [str(cls) for cls in clf4.classes_]
tree_data = tree.export_graphviz(clf4, out_file=None, feature_names=X_train_new4.columns, class_names=class_names, filled=True, proportion=True, rounded=True, special_characters=True)
graph = graphviz.Source(tree_data)
graph.format="png"
graph.render("03_dt_tree_structure.gv", view=True)

#=====================================================
#交叉驗證
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf4, X, y, cv=5, scoring="accuracy")
print("最佳模型交叉驗證cv=5的模型正確率=", score.mean())
    
