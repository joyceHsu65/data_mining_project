#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 22:17:29 2025

@author: 409383712 徐胤瑄
"""

import json
import pandas as pd

with open("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/TCMSV_alldesc-2.json", "r", encoding="utf-8") as file:
    raw1_json = json.load(file)
data = raw1_json["data"]["park"]  # 這一層是 list of dict
data = pd.DataFrame(data)
data.info()
data.to_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/raw1_taipei_paring_lot_Info.csv", index=False, encoding="utf_8_sig")


'''
with open("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/TCMSV_allavailable-2.json", "r", encoding="utf-8") as file2:
    raw2_json = json.load(file2)
data2 = raw2_json["data"]["park"]  # 這一層才是 list of dict
data2 = pd.DataFrame(data2)
data2.info()
data2.to_csv("/Users/joycehsu/大學/113-2/2資料探勘/data mining code files_報告/raw2_taipei_paring_lot_availble.csv", index=False, encoding="utf_8_sig")
'''

         