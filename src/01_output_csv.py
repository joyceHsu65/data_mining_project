#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Taipei Parking Lot Data Preprocessing Pipeline
==============================================

使用資料：2025/05/06

目的：
- 將台北市停車場原始資料（raw JSON）轉換為 CSV 格式，
- 以便後續清理與整併。

輸入：
- data/raw/TCMSV_alldesc-2.json

輸出：
- data/raw/raw_taipei_paring_lot_Info.csv（原始資料轉換後的 CSV 格式）

步驟摘要：
1) 載入 JSON 資料
2) 提取停車場資訊並轉換為 DataFrame
3) 輸出為 CSV 格式

作者：徐胤瑄
"""

import json
import pandas as pd

with open("data/raw/TCMSV_alldesc-2.json", "r", encoding="utf-8") as file:
    raw1_json = json.load(file)
data = raw1_json["data"]["park"]  # 這一層是 list of dict
data = pd.DataFrame(data)
data.info()
data.to_csv("data/raw/raw_taipei_paring_lot_Info.csv", index=False, encoding="utf_8_sig")



         