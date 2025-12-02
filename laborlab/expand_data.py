"""
데이터를 3개에서 50개로 확장하는 스크립트
- data.csv 확장
- RESUME_JSON.json 확장
- COVERLETTERS_JSON.json 확장
- TRAININGS_JSON.json 확장
- LICENSES_JSON.json 확장
"""

import pandas as pd
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# 랜덤 시드 설정 (재현 가능성을 위해)
random.seed(42)

def expand_csv_data(input_file, output_file, target_count=50):
    """CSV 데이터를 확장합니다."""
    df = pd.read_csv(input_file)
    original_count = len(df)
    
    if original_count >= target_count:
        print(f"이미 {original_count}개 이상의 데이터가 있습니다. 확장하지 않습니다.")
        return
    
    # 기존 데이터를 복사하여 확장
    new_rows = []
    for i in range(original_count, target_count):
        # 기존 데이터 중 하나를 랜덤하게 선택하여 복사
        base_row = df.iloc[i % original_count].copy()
        
        # 고유 ID 생성 (W10001 형식 유지)
        base_row['JHNT_MBN'] = f"W{i+1+10000:05d}"
        base_row['JHNT_CTN'] = f"C{i+1+10000:05d}"
        
        # 날짜 변형 (기존 날짜에 랜덤 일수 추가)
        if pd.notna(base_row.get('JHCR_DE')):
            try:
                base_date = datetime.strptime(str(base_row['JHCR_DE']), '%Y-%m-%d')
                new_date = base_date + timedelta(days=random.randint(-30, 30))
                base_row['JHCR_DE'] = new_date.strftime('%Y-%m-%d')
            except:
                pass
        
        # 숫자 값에 약간의 변형 추가
        numeric_cols = ['HOPE_WAGE_SM_AMT', 'AGE', 'BFR_OCTR_CT', 'CARR_MYCT1', 
                       'SFID_IEM_NUM', 'SFID_LTTR_NUM', 'IPS_IRDS_NMPR', 'JBHT_NMPR', 
                       'RCIT_NMPR', 'NTR_BPLC_PSNT_WAGE_AMT', 'AVG_EMPN_VS_CRQF_CT',
                       'IPS_VS_RCIT_RATE', 'IPS_VS_JBHT_RATE', 'AVG_OTIO_MYAV_RMNT_AMT',
                       'AVG_HOPE_WAGE_SM_AMT', 'CRQF_CT']
        
        for col in numeric_cols:
            if col in base_row and pd.notna(base_row[col]):
                try:
                    val = float(base_row[col])
                    # ±10% 범위 내에서 변형
                    base_row[col] = val * (1 + random.uniform(-0.1, 0.1))
                except:
                    pass
        
        # 범주형 변수 랜덤 선택
        categorical_options = {
            'EMPL_STLE_CD': ['기간의 정함이 없는 근로계약', '기간의 정함이 있는 근로계약', '일용근로', '단시간근로'],
            'DSPT_LABR_YN': ['예', '아니요'],
            'COMM_WAGE_TYCD': ['상용', '일용', '단시간'],
            'SXDS_CD': ['남', '여'],
            'ACCR_STCD': ['졸업', '중퇴', '재학중'],
            'ACCR_CD': ['4년제 대학', '대학원', '고등학교', '전문대학', '2년제 대학'],
            'JHNT_PPOS_CD': ['구직급여', '취업알선', '기타'],
            'JHNT_RQUT_CHNL_SECD': ['온라인', '고용24', '오프라인', '기타'],
            'INFO_OTPB_GRAD_CD': ['예', '아니요'],
            'MDTN_HOPE_GRD_CD': ['필요', '불필요'],
            'IDIF_AOFR_YN': ['예', '아니요'],
            'EMAIL_RCYN': ['예', '아니요'],
            'DRV_PSBL_YN': ['예', '아니요'],
            'SAEIL_CNTC_AGRE_YN': ['예', '아니요'],
            'SHRS_IDIF_AOFR_YN': ['예', '아니요'],
            'SULC_IDIF_AOFR_YN': ['예', '아니요'],
            'IDIF_IQRY_AGRE_YN': ['예', '아니요'],
            'DLY_LABR_HOPE_YN': ['예', '아니요'],
            'RQAG_HOPE_YN': ['예', '아니요'],
            'SHSY_YN': ['예', '아니요'],
            'MDTN_HOPE_YN': ['예', '아니요'],
            'SMS_RCYN': ['예', '아니요'],
            'EMAIL_OTPB_YN': ['예', '아니요'],
            'MPNO_OTPB_YN': ['예', '아니요'],
            'AFIV_RDJT_PSBL_YN': ['예', '아니요'],
            'BFR_OCTR_YN': ['예', '아니요'],
            'UEPS_RECP_YN': ['예', '아니요'],
        }
        
        for col, options in categorical_options.items():
            if col in base_row:
                base_row[col] = random.choice(options)
        
        # ACQ_180_YN 랜덤 설정 (0 또는 1)
        if 'ACQ_180_YN' in base_row:
            base_row['ACQ_180_YN'] = random.choice([0, 1])
        
        new_rows.append(base_row)
    
    # 새 행 추가
    new_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    new_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ CSV 데이터 확장 완료: {original_count}개 → {len(new_df)}개")


def expand_resume_json(input_file, output_file, target_count=50):
    """이력서 JSON 데이터를 확장합니다."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    if original_count >= target_count:
        print(f"이미 {original_count}개 이상의 데이터가 있습니다. 확장하지 않습니다.")
        return
    
    # 기존 데이터를 복사하여 확장
    for i in range(original_count, target_count):
        base_item = json.loads(json.dumps(data[i % original_count]))  # Deep copy
        
        # 고유 ID 생성 (W10001 형식 유지)
        base_item['JHNT_MBN'] = f"W{i+1+10000:05d}"
        base_item['JHNT_CTN'] = f"C{i+1+10000:05d}"
        
        # ITEMS 내부의 일부 값 변형
        if 'RESUMES' in base_item and len(base_item['RESUMES']) > 0:
            resume = base_item['RESUMES'][0]
            if 'ITEMS' in resume:
                for item in resume['ITEMS']:
                    # 날짜 변형
                    for date_key in ['HIST_STDT', 'HIST_ENDT']:
                        if date_key in item and item[date_key]:
                            try:
                                base_date = datetime.strptime(item[date_key], '%Y-%m-%d')
                                new_date = base_date + timedelta(days=random.randint(-365, 365))
                                item[date_key] = new_date.strftime('%Y-%m-%d')
                            except:
                                pass
        
        data.append(base_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 이력서 JSON 데이터 확장 완료: {original_count}개 → {len(data)}개")


def expand_coverletter_json(input_file, output_file, target_count=50):
    """자기소개서 JSON 데이터를 확장합니다."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    if original_count >= target_count:
        print(f"이미 {original_count}개 이상의 데이터가 있습니다. 확장하지 않습니다.")
        return
    
    # 기존 데이터를 복사하여 확장
    for i in range(original_count, target_count):
        base_item = json.loads(json.dumps(data[i % original_count]))  # Deep copy
        
        # 고유 ID 생성 (W10001 형식 유지)
        base_item['JHNT_MBN'] = f"W{i+1+10000:05d}"
        base_item['JHNT_CTN'] = f"C{i+1+10000:05d}"
        
        # SFID_NO 변형
        if 'COVERLETTERS' in base_item:
            for coverletter in base_item['COVERLETTERS']:
                if 'SFID_NO' in coverletter:
                    coverletter['SFID_NO'] = f"{(i+1)*1000000000 + random.randint(1, 999):012d}"
        
        data.append(base_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 자기소개서 JSON 데이터 확장 완료: {original_count}개 → {len(data)}개")


def expand_training_json(input_file, output_file, target_count=50):
    """직업훈련 JSON 데이터를 확장합니다."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    if original_count >= target_count:
        print(f"이미 {original_count}개 이상의 데이터가 있습니다. 확장하지 않습니다.")
        return
    
    # 기존 데이터를 복사하여 확장
    for i in range(original_count, target_count):
        base_item = json.loads(json.dumps(data[i % original_count]))  # Deep copy
        
        # 고유 ID 생성 (W10001 형식 유지)
        base_item['JHNT_MBN'] = f"W{i+1+10000:05d}"
        base_item['JHNT_CTN'] = f"C{i+1+10000:05d}"
        
        # 날짜 변형
        if 'JHCR_DE' in base_item and base_item['JHCR_DE']:
            try:
                base_date = datetime.strptime(base_item['JHCR_DE'], '%Y-%m-%d')
                new_date = base_date + timedelta(days=random.randint(-30, 30))
                base_item['JHCR_DE'] = new_date.strftime('%Y-%m-%d')
            except:
                pass
        
        # TRAININGS 내부의 날짜 변형
        if 'TRAININGS' in base_item:
            for training in base_item['TRAININGS']:
                for date_key in ['TRNG_BGDE', 'TRNG_ENDE']:
                    if date_key in training and training[date_key]:
                        try:
                            base_date = datetime.strptime(training[date_key], '%Y-%m-%d')
                            new_date = base_date + timedelta(days=random.randint(-180, 180))
                            training[date_key] = new_date.strftime('%Y-%m-%d')
                        except:
                            pass
                
                # CRSE_ID 변형
                if 'CRSE_ID' in training:
                    training['CRSE_ID'] = str(random.randint(10000000000000000, 99999999999999999))
        
        data.append(base_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 직업훈련 JSON 데이터 확장 완료: {original_count}개 → {len(data)}개")


def expand_license_json(input_file, output_file, target_count=50):
    """자격증 JSON 데이터를 확장합니다."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    if original_count >= target_count:
        print(f"이미 {original_count}개 이상의 데이터가 있습니다. 확장하지 않습니다.")
        return
    
    # 자격증 옵션
    license_options = [
        {"QULF_ITNM": "전기기능사", "QULF_LCNS_LCFN": "국가기술자격"},
        {"QULF_ITNM": "산업안전기사", "QULF_LCNS_LCFN": "국가기술자격"},
        {"QULF_ITNM": "정보처리기사", "QULF_LCNS_LCFN": "국가기술자격"},
        {"QULF_ITNM": "건설기계정비기능사", "QULF_LCNS_LCFN": "국가기술자격"},
        {"QULF_ITNM": "컴퓨터활용능력 2급", "QULF_LCNS_LCFN": "국가기술자격"},
        {"QULF_ITNM": "가스기능사", "QULF_LCNS_LCFN": "국가기술자격"},
        {"QULF_ITNM": "ADsP(데이터분석준전문가)", "QULF_LCNS_LCFN": "민간자격"},
        {"QULF_ITNM": "SQLD(데이터베이스 개발자)", "QULF_LCNS_LCFN": "민간자격"},
        {"QULF_ITNM": "토익", "QULF_LCNS_LCFN": "민간자격"},
        {"QULF_ITNM": "한국사능력검정시험", "QULF_LCNS_LCFN": "민간자격"},
    ]
    
    # 기존 데이터를 복사하여 확장
    for i in range(original_count, target_count):
        base_item = json.loads(json.dumps(data[i % original_count]))  # Deep copy
        
        # 고유 ID 생성 (W10001 형식 유지)
        base_item['JHNT_MBN'] = f"W{i+1+10000:05d}"
        base_item['JHNT_CTN'] = f"C{i+1+10000:05d}"
        
        # LICENSES 내부의 값 변형
        if 'LICENSES' in base_item:
            for license_item in base_item['LICENSES']:
                # 자격증 정보 랜덤 선택
                license_info = random.choice(license_options)
                license_item['QULF_ITNM'] = license_info['QULF_ITNM']
                license_item['QULF_LCNS_LCFN'] = license_info['QULF_LCNS_LCFN']
                
                # CRQF_CD 변형
                if 'CRQF_CD' in license_item:
                    license_item['CRQF_CD'] = str(random.randint(1000000, 9999999))
                
                # 날짜 변형
                if 'ETL_DT' in license_item and license_item['ETL_DT']:
                    try:
                        # 날짜 부분만 추출
                        date_str = license_item['ETL_DT'].split()[0]
                        base_date = datetime.strptime(date_str, '%Y-%m-%d')
                        new_date = base_date + timedelta(days=random.randint(-365, 365))
                        time_part = license_item['ETL_DT'].split()[1] if ' ' in license_item['ETL_DT'] else '오전 12:00:00'
                        license_item['ETL_DT'] = f"{new_date.strftime('%Y-%m-%d')} {time_part}"
                    except:
                        pass
        
        data.append(base_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 자격증 JSON 데이터 확장 완료: {original_count}개 → {len(data)}개")


def main():
    """메인 함수"""
    data_dir = Path(__file__).parent / "data"
    fixed_data_dir = data_dir / "fixed_data"
    variant_data_dir = data_dir / "variant_data"
    
    target_count = 50
    
    print(f"📊 데이터를 {target_count}개로 확장합니다...\n")
    
    # CSV 데이터 확장
    csv_input = fixed_data_dir / "data.csv"
    csv_output = fixed_data_dir / "data.csv"
    if csv_input.exists():
        expand_csv_data(csv_input, csv_output, target_count)
    else:
        print(f"⚠️ CSV 파일을 찾을 수 없습니다: {csv_input}")
    
    # JSON 데이터 확장
    json_files = [
        ("RESUME_JSON.json", expand_resume_json),
        ("COVERLETTERS_JSON.json", expand_coverletter_json),
        ("TRAININGS_JSON.json", expand_training_json),
        ("LICENSES_JSON.json", expand_license_json),
    ]
    
    for filename, expand_func in json_files:
        json_input = variant_data_dir / filename
        json_output = variant_data_dir / filename
        if json_input.exists():
            expand_func(json_input, json_output, target_count)
        else:
            print(f"⚠️ JSON 파일을 찾을 수 없습니다: {json_input}")
    
    print(f"\n✅ 모든 데이터 확장 완료! (목표: {target_count}개)")


if __name__ == "__main__":
    main()

