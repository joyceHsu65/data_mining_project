#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:04:24 2025

@author: 409383712 徐胤瑄
"""
import pandas as pd
parking = pd.read_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/raw1_taipei_paring_lot_Info.csv")
#parking.info()

# 刪除parking無需使用之欄位
parking.drop(["name", "payex", "summary", "address", "tel", "tw97x", "tw97y", "totalbus", "Taxi_OneHR_Free", "AED_Equipment", "CellSignal_Enhancement", "Accessibility_Elevator", "Phone_Charge", "Child_Pickup_Area", "FareInfo", "EntranceCoord", "ptype"], axis=1, inplace=True)
parking.info()
#parking.to_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/preprocessing1_taipei_paring_lot_Info.csv", index=False, encoding="utf_8_sig")

# 處理fare
fare = pd.read_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/preprocessing2_fare.csv")
fare.drop(["Unnamed: 0"], axis=1, inplace=True)
fare.columns = ["farecar_weekday", "farecar_night", "farecar_haliday", "farecar_month", "faremotor_day/hour", "faremotor_month", "largemotor_month"]
fare.info()
fare["farecar_weekday"] = (
    fare["farecar_weekday"]
    .astype(str)  # 先保證是字串
    .str.replace(",", "", regex=False)  
    .str.extract(r"(\d+\.?\d*)")[0]     
    .astype(float)                      
)

# 合併
parking_new = pd.concat([parking, fare], axis=1)
parking_new.info()


# 處理目標變數type2
print(parking_new["type2"].value_counts(dropna=False))
mapping = {
    "民營停車場": 2,
    "本處委外停車場": 1,
    "本處自營停車場": 1,
    "市屬機關學校委外": 1,
    "市屬機關學校自營": 1
}
parking_new["type2"] = parking_new["type2"].map(mapping)

# 處理serviceTime
print(parking_new["serviceTime"].value_counts(dropna=False))
parking_new["serviceTime"] = parking_new["serviceTime"].apply(
    lambda x: 1 if x == "24小時" else 0
)


# 處理遺失值資料
## 刪除沒有目標變數的資料
parking_new = parking_new.dropna(subset=["type2"])

## 刪除僅提供給大客車的停車場
mask = parking_new[["farecar_weekday", "farecar_month", "faremotor_day/hour", "faremotor_month"]].isna().all(axis=1)
print(parking_new[mask])  # 顯示符合條件的 row
parking_new_mask = pd.DataFrame(parking_new[mask])
parking_new = parking_new[~mask]

## 將不同類型停車場的做分類
#建立分類函數
def classify_parking_type(row):
    # 條件一：僅提供月租
    if pd.isna(row["farecar_weekday"]) and pd.isna(row["farecar_night"]) and pd.isna(row["farecar_haliday"]) and pd.isna(row["faremotor_day/hour"]):
        return "monthly_rent"
    # 條件二：僅提供小客車或重型機車
    elif pd.isna(row["faremotor_day/hour"]) and pd.isna(row["faremotor_month"]):
        return "car_and_largemotor"
    # 條件三：僅提供機車
    elif pd.isna(row["farecar_weekday"]) and pd.isna(row["farecar_night"]) and pd.isna(row["farecar_haliday"]) and pd.isna(row["farecar_month"]) and pd.isna(row["largemotor_month"]):
        return "motor"
    # 其他為混合式
    else:
        return "mixed_parking_lot"
#套用分類函數
parking_new["parking_fare_classification"] = parking_new.apply(classify_parking_type, axis=1)
print(parking_new["parking_fare_classification"].value_counts(dropna=False))

## 停車場價錢處理遺失值
#定義條件對應的規定遺失值欄位（要補 -1）
fixed_fill_rules = {
    "monthly_rent": ["farecar_weekday", "farecar_night", "farecar_haliday", "faremotor_day/hour"],
    "car_and_largemotor": ["faremotor_day/hour", "faremotor_month"],
    "motor": ["farecar_weekday", "farecar_night", "farecar_haliday", "farecar_month", "largemotor_month"],
    "mixed_parking_lot": []
}
#對每一條 row 根據分類補上 -1
parking_new2 = parking_new.copy()
for group, cols_to_fill in fixed_fill_rules.items():
    mask = parking_new2["parking_fare_classification"] == group
    for col in cols_to_fill:
        parking_new2.loc[mask & parking_new2[col].isna(), col] = -1
#parking_new2.info()

#個別處理剩下的遺失值
import numpy as np
parking_new2["farecar_weekday"]=np.where(parking_new2["farecar_weekday"].isnull(), np.nanmedian(parking_new2["farecar_weekday"]), parking_new2["farecar_weekday"])
parking_new2["farecar_night"]=np.where(parking_new2["farecar_night"].isnull(), np.nanmedian(parking_new2["farecar_night"]), parking_new2["farecar_night"])
parking_new2["farecar_haliday"]=np.where(parking_new2["farecar_haliday"].isnull(), np.nanmedian(parking_new2["farecar_haliday"]), parking_new2["farecar_haliday"])
parking_new2["farecar_month"]=np.where(parking_new2["farecar_month"].isnull(), np.nanmedian(parking_new2["farecar_month"]), parking_new2["farecar_month"])
parking_new2["faremotor_day/hour"]=np.where(parking_new2["faremotor_day/hour"].isnull(), np.nanmedian(parking_new2["faremotor_day/hour"]), parking_new2["faremotor_day/hour"])
parking_new2["faremotor_month"]=np.where(parking_new2["faremotor_month"].isnull(), np.nanmedian(parking_new2["faremotor_month"]), parking_new2["faremotor_month"])
parking_new2["largemotor_month"]=np.where(parking_new2["largemotor_month"].isnull(), np.nanmedian(parking_new2["largemotor_month"]), parking_new2["largemotor_month"])

# 輸出
parking_new2.info()
parking_new2.to_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/preprocessing3_taipei_paring_lot_availble.csv", index=False, encoding="utf_8_sig")
