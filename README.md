# kNN 기반 실내 위치 추적 시스템

## 개요 📝

다양한 측위 알고리즘을 사용하여 실내 위치 추적 시스템(IPS)을 구현하는 것을 목표로 합니다. 시스템은 수신 신호 강도 지표(RSSI)를 활용하 위치를 추정합니다.

---

## 주요 특징 💡

- PDR, kNN, wkNN, NN 알고리즘을 이용한 위치 추정
- 실시간 위치 시각화
- 데이터베이스와 측위 알고리즘을 이용한 정확도 높은 위치 추정
- Database 3차원 RSS 패턴 확인

---

## 사용 방법 🛠️

### 데이터 로딩
데이터베이스와 테스트 결과 데이터를 로드합니다.

```python
DB_File = 'DB'
FP_File = 'FP_TestResult_15k'
PDR_File = 'PDR_Data1'
```

### 파라미터 설정
필요한 모든 파라미터를 설정합니다.

```python
BT_Num = 11
RP_Point = len(DB)
PDR_Point = len(PDR)
K = 15
```

### 실행
코드를 실행하면 실내 위치 추적 결과가 2D 그래프로 출력됩니다.

---

## 요구 사항 📋

- Python 3.x
- Pandas
- NumPy
- Matplotlib

---

## 실행 결과 📊

- DB 포인트와 비교하여 계산된 위치가 2D 그래프로 출력됩니다.
- 위치는 설정한 시간에 맞춰 실시간 업데이트가 가능합니다.

---
