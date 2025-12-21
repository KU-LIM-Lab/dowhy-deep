import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cx_Oracle
import os
import copy
import pandas as pd
import json

def queryOracle(query):
    conn = cx_Oracle.connect("커넥션 정보는 추후 제공드리도록 하겠습니다.") 
    cur = conn.cursor()
    cur.execute(query)    
    _result = cur.fetchall()
    header = [row[0] for row in cur.description]
    cur.close()
    conn.close()
    
    return(header, _result)

# 변환 함수 (numpy 타입 -> 파이썬 타입)
def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    return obj




resume_query = "테이블 정보는 추후 제공드리도록 하겠습니다." 

header, result = queryOracle(resume_query)
resume_df = pd.DataFrame(result, columns=header)

group_cols = ["SEEK_CUST_NO","TMPL_SEQNO","RESUME_TITLE","BASIC_RESUME_YN"]
json_cols = [col for col in resume_df.columns if col not in group_cols]

resume_result = []

for keys, group in resume_df.groupby(group_cols):
    group_dict = dict(zip(group_cols, keys))
    group_dict["RESUME_JSON"] = group[json_cols].astype(object).to_dict(orient="records")
    resume_result.append(group_dict)

# JSON 변환
resume_json = json.dumps(resume_result, ensure_ascii=False, indent=2, default=convert_numpy)

print(resume_json)




coverletters_query = "테이블 정보는 추후 제공드리도록 하겠습니다."

header, result = queryOracle(coverletters_query)
coverletters_df = pd.DataFrame(result, columns=header)

group_cols = ["SEEK_CUST_NO","SFID_NO","BASS_SFID_YN"]
json_cols = [col for col in coverletters_df.columns if col not in group_cols]

coverletters_result = []

for keys, group in coverletters_df.groupby(group_cols):
    group_dict = dict(zip(group_cols, keys))
    group_dict["COVERLETTERS_JSON"] = group[json_cols].astype(object).to_dict(orient="records")
    coverletters_result.append(group_dict)

# JSON 변환
coverletters_json = json.dumps(coverletters_result, ensure_ascii=False, indent=2, default=convert_numpy)

print(coverletters_json)




tranings_query = "테이블 정보는 추후 제공드리도록 하겠습니다." 

header, result = queryOracle(tranings_query)
tranings_df = pd.DataFrame(result, columns=header)

group_cols = ["CLOS_YM","JHNT_CTN","JHCR_DE"]
json_cols = [col for col in tranings_df.columns if col not in group_cols]

tranings_result = []

for keys, group in tranings_df.groupby(group_cols):
    group_dict = dict(zip(group_cols, keys))
    group_dict["TRAININGS_JSON"] = group[json_cols].astype(object).to_dict(orient="records")
    tranings_result.append(group_dict)

# JSON 변환
tranings_json = json.dumps(tranings_result, ensure_ascii=False, indent=2, default=str)

print(tranings_json)




licenses_query = "테이블 정보는 추후 제공드리도록 하겠습니다."

header, result = queryOracle(licenses_query)
licenses_df = pd.DataFrame(result, columns=header)

group_cols = ["CLOS_YM","JHNT_CTN"]
json_cols = [col for col in licenses_df.columns if col not in group_cols]

licenses_result = []

for keys, group in licenses_df.groupby(group_cols):
    group_dict = dict(zip(group_cols, keys))
    group_dict["LICENSES_JSON"] = group[json_cols].astype(object).to_dict(orient="records")
    licenses_result.append(group_dict)

# JSON 변환
licenses_json = json.dumps(licenses_result, ensure_ascii=False, indent=2, default=str)

print(licenses_json)





