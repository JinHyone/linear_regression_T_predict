import sklearn
from sklearn.linear_model import RidgeCV
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import seaborn as sns


matplotlib.use('Qt5Agg')
sklearn.set_config(transform_output="pandas")

ridge = RidgeCV([0.001, 0.01, 0.1, 1, 10, 100])  # ridge 다중 회귀 모델


df = pd.read_csv('2011-2020 기상 데이터.csv', encoding='cp949')  # 데이터 불러오기
df = df.dropna()  # 결측치 제거

df['datetime'] = pd.to_datetime(df['일시'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour


train_data = df[0:78888]  # 2010.01.01-2019.12.31
test_data = df[78889:-1]  # 2020.01.01-2020.12.31

# 훈련 데이터
train_X = train_data[['year', 'month', 'day', 'hour']]
train_Y = train_data[['기온']]

# 테스트 데이터
test_X = test_data[['year', 'month', 'day', 'hour']]
test_Y = test_data[['기온']]


poly = PolynomialFeatures(include_bias=False, degree=5)  # 특성 생성 모델
poly.fit(train_X)

# 특성 데이터
train_poly = poly.transform(train_X)
test_poly = poly.transform(test_X)

ss = StandardScaler()  # 정규화 모델
ss.fit(train_poly)

# 정규화 데이터
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 모델 훈련 및 결정 계수(R -> 유사 정확도) 출력
ridge.fit(train_scaled, train_Y)
print(ridge.score(test_scaled, test_Y))

# 예측 데이터 프레임 생성
ttt = test_data[['datetime', '기온']].copy(deep=True)
ttt['기온'] = ridge.predict(ss.transform(poly.transform(test_data[['year', 'month', 'day', 'hour']])))

# 시각화
plt.figure(figsize=(10, 5))
sns.lineplot(data=test_data, x='datetime', y='기온')
sns.lineplot(data=ttt, x='datetime', y='기온')
plt.show()
