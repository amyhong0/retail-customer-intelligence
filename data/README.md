# 데이터 준비

출처는 [dunnhumby 공식 공개 자료](https://www.dunnhumby.com/source-files/)의 **The Complete Journey**입니다. 이번 결과는 공식 ZIP의 CSV를 실제 분석했습니다. 제공 페이지에서 이용 조건을 확인하세요.

```bash
python download_data.py
python run_analysis.py --max-items 1000
```

실제 분석에는 dunnhumby의 공개 자료 `The Complete Journey`를 이용합니다. 제공 페이지에서 이용 조건을 확인하고 내려받은 뒤 다음 파일을 `data/raw/`에 넣으세요.

- `transaction_data.csv`: 필수
- `product.csv`: 필수
- `hh_demographic.csv`: 선택
- `campaign_table.csv`: 선택
- `campaign_desc.csv`: 캠페인 시작일을 연결해 미래 배정 누수 방지

거래 원본 2,595,732행, 상품 92,353행, 인구통계 801행, 캠페인 배정 7,208행을 사용했습니다. 정제 후 거래는 2,581,266행입니다. DAY는 상대 일수이므로 임의의 날짜나 요일을 부여하지 않습니다. 원본 CSV는 제출 ZIP에도 포함하지 않습니다.

`--synthetic`은 원본 없이 코드를 검증하는 인공 거래 생성 옵션입니다. 공식 데이터 분석 결과와 구분해야 합니다.

이후 `python run_analysis.py`를 실행합니다. 용량과 재배포 조건을 고려해 원본 CSV는 Git 추적에서 제외했습니다. 일부 파일의 복수형 이름도 인식합니다. 인구통계 파일은 현재 로딩만 하며 예측 피처에는 사용하지 않습니다.
