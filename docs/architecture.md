# 현재 분석 아키텍처

```text
거래·상품·고객·캠페인 CSV
            ↓
data.py: 로딩·정제 → run_analysis.py
            ├─ recommender.py: 구매 행렬 → 인기 상품 추천·구매 이력 기반 개인화 추천 → 추천 지표
            └─ features.py: 고객 특성 12개 → prediction.py: 다음 28일 구매 예측
                                                        → 예측 지표·특성 중요도·대상 고객
            ↓
outputs/: 지표·고객 특성·학습 모델
            ↓
visualization.py: 저장된 결과로 추천 비교·고객 특성 중요도 그래프 생성
            ↓
site/dist/: 데이터·분석 방법·결과를 설명하는 정적 사이트
```

추천은 구매 행렬을 사용하고 구매 예측은 고객 특성 테이블을 사용합니다. 그래프만 다시 생성할 때는 `python run_analysis.py --plots-only`를 실행합니다. 저장된 CSV를 사용하므로 모델과 분석 지표는 변경하지 않습니다.
