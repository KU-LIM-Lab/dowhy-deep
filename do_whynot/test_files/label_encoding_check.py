import json

#JSON 파일 경로
json_path='label_encoding_map.json'

#JSON 불러오기
with open(json_path,'r',encoding='utf-8') as f:
    mapping=json.load(f)

# 각 컬럼별 레이블 개수 출력
for col,label_dict in mapping.items():
    print(f"{col}:{len(label_dict)}labels")