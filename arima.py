# sudo apt install python3-pyaudio -y
# // 스피커
# sudo apt-get install espeak -y
# // 텍스트 -> 음성
# pip3 install speechrecognition

# sudo apt-get install flac

# //날씨 음성을 인식하는 코드

# import speech_recognition as sr
# try:
#     while  True :
#             r = sr.Recognizer()

#             with sr.Microphone() as source:
#                 print("Say something!")
#                 audio = r.listen(source)
#             try:
#                 text = r.recognize_google(audio, language = 'ko-KR')
#                 print('You said : ' + text)

#                 if text in "날씨" :
#                     print("날씨 음성을 인식하였습니다.")
#             except sr.UnknownValueError:
#                 print('Google Speech Recognition could not understand audio')
#             except sr.ReauestError as e:
#                 print("Could not request results from Google Speech Recognition Service; {0}".format(e))
# except KeyboardInterrupt:
#     pass


# //자신의 지역의 날씨정보를 음성으로 출력하는 코드

#
import os
import pandas as pd
from pandas import to_datetime
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
import seaborn as sns
import itertools
import warnings
import datetime
from datetime import datetime

warnings.filterwarnings('ignore')

import itertools
from tqdm import tqdm
# ARIMA 모델 패키지
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# AUTO ARIMA 모델 패키지
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_predict

# 인덱스 추가
# 예측

df = pd.read_csv('/Users/kylee/Downloads/SURFACE_ASOS_108_MI_2022-11_2022-11_2022.csv', encoding='CP949')
df

df['date'] = pd.to_datetime(df['일시'], format='%Y-%m-%d %H', errors='raise')
# df['new_Date'] = pd.to_datetime(df['일시'])
df.drop('일시', axis=1, inplace=True)

# train 데이터와 validation 데이터 나누기
x_train = df[df['date'] < '2022-11-25']
x_valid = df[df['date'] >= '2022-11-25']

# 예측한 열만 필터링하기
nyc = x_train[['date', '기온(°C)']]
nyc_v = x_valid[['date', '기온(°C)']]

# Date 를 dataframe의 인덱스로 만들어주기
nyc.set_index('date', inplace=True)
nyc_v.set_index('date', inplace=True)

nyc_i = nyc.index
nyc.index = pd.DatetimeIndex(nyc.index).to_period('H')
nyc_v.index = pd.DatetimeIndex(nyc_v.index).to_period('H')
# nyc.index = pd.DatetimeIndex(nyc.index).to_period('H')

# 4주간의 값을 예측할 것이므로 예측날짜들을 인덱스로 한 Dataframe 만들기
index_4_days = pd.date_range("2022-11-24 23:00", freq='1H', periods=24 * 4, tz=None)

# 확인해보기
index_4_days

import statsmodels.api as sm

# order에 파라미터 넣어주기 (p,d,q)
model_arima = ARIMA(nyc, order=(1, 0, 2))
# p개의 과거 값들을 이용해 예측
# q개 전값부터 오차값을 이용해 예측
model_arima_fit = model_arima.fit()  # disp<0 : convergence data가 표시 안됨
print(model_arima_fit.summary())
# 예측한 값들을 저장
fcast1 = model_arima_fit.forecast(96).values
fcast1 = pd.Series(fcast1, index=index_4_days)
fcast1 = fcast1.rename("Arima")

fcast1

# 예측값 시각화 하기 -> 잘 안됨
fig, ax = plt.subplots(figsize=(15, 5))
chart = sns.lineplot(x='new_Date', y='기온(°C)', data=nyc)
chart.set_title('forecast_temparature')
fcast1.plot(ax=ax, color='red', legend=True)
nyc_v.plot(ax=ax, color='blue', legend=True)
plt.savefig("forecast.png")

# 최적의 갯수를 직접 찾아보기
import itertools

p = range(0, 3)
d = range(1, 2)
q = range(0, 6)

pdq = list(itertools.product(p, d, q))

aic = []
params = []

with tqdm(total=len(pdq)) as pg:
    for i in pdq:
        pg.update(1)
        try:
            model = SARIMAX(nyc['기온(°C)'], order=(i))
            model_fit = model.fit()
            aic.append(round(model_fit.aic, 2))
            params.append((i))
        except:
            continue

# 최적의 조건으로 시각화
optimal = [(params[i], j) for i, j in enumerate(aic) if j == min(aic)]
optimal[0]
import statsmodels.api as sm

# order에 파라미터 넣어주기 (p,d,q)
model_arima = ARIMA(nyc, order=(1, 1, 4))
# p개의 과거 값들을 이용해 예측
# q개 전값부터 오차값을 이용해 예측
model_arima_fit = model_arima.fit()  # disp<0 : convergence data가 표시 안됨
print(model_arima_fit.summary())

# 예측한 값들을 저장
fcast1 = model_arima_fit.forecast(24 * 4).values
fcast1 = pd.Series(fcast1, index=index_4_days)
fcast1 = fcast1.rename("Arima")

fig, ax = plt.subplots(figsize=(15, 5))
chart = sns.lineplot(x='date', y='기온(°C)', data=nyc)
chart.set_title('forecast_temparature')
fcast1.plot(ax=ax, color='red', legend=True)
nyc_v.plot(ax=ax, color='blue', legend=True)
plt.savefig("forecast2.png")

#####
p = range(0, 3)
d = range(1, 2)
q = range(0, 6)
m = 24
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], m) for x in list(itertools.product(p, d, q))]

aic = []
params = []

with tqdm(total=len(pdq) * len(seasonal_pdq)) as pg:
    for i in pdq:
        for j in seasonal_pdq:
            pg.update(1)
            try:
                model = SARIMAX(nyc['기온(°C)'], order=(i), season_order=(j))
                model_fit = model.fit()
                aic.append(round(model_fit.aic, 2))
                params.append((i, j))
            except:
                continue
optimal = [(params[i], j) for i, j in enumerate(aic) if j == min(aic)]
model_opt = SARIMAX(nyc['기온(°C)'], order=optimal[0][0][0], seasonal_order=optimal[0][0][1])
model_opt_fit = model_opt.fit(disp=0)
model_opt_fit.summary()

# optimal[0][0][0] = (1,1,4)
# optimal[0][0][1] = (0, 1, 0, 24)
# 시각화
# model = SARIMAX(nyc['기온(°C)'], order=optimal[0][0][0], seasonal_order=optimal[0][0][1])
model = SARIMAX(nyc['기온(°C)'], order=(1, 1, 4), seasonal_order=(0, 1, 1, 24))
# plt.show()
# plt.ion()
model_fit = model.fit(disp=0)
model_fit.summary()
model_fit.plot_diagnostics()
pred = model_fit.get_prediction(start=pd.to_datetime('2022-11-25 00:00'), end=pd.to_datetime('2022-11-29 00:00'),
                                dynamic=True)

forecast = model_fit.forecast(steps=1)

# plt.figure(figsize=(20,5))
# plt.plot(nyc_v, label="real")
# plt.plot(forecast, label="predict")
# plt.legend()
# plt.savefig("forecast_auto.png")


fig, ax = plt.subplots(figsize=(15, 5))
chart = sns.lineplot(x='date', y='기온(°C)', data=nyc)
chart.set_title('forecast_temparature')
pred.plot(ax=ax, color='red', legend=True)
nyc_v.plot(ax=ax, color='blue', legend=True)
plt.savefig("forecast_auto.png")

######


# pdq = list(itertools.product(p,d,q))
# for param in pdq:
#     try:
#         model_arima = ARIMA(nyc, order=param)
#         model_arima_fit = model_arima.fit()
#         print(param, model_arima_fit.aic)
#     except:
#         continue


# 최적의 갯수를 자동으로 찾아보기
# Auto-arima 돌리기 계측값이 일별이면 m=1, 월별이면 m=12, 주별이면 m=52
# 계절성 있는 데이터면 seasonal=True로 바꿔줘야함 알아서 d 값을 찾아줌
auto_arima_model = pm.auto_arima(nyc, seasonal=True, m=1)
auto_arima_model.fit(nyc)
# 모델 예측
fcast2 = auto_arima_model.predict(4)
fcast2 = pd.Series(fcast2, index=index_4_days)
fcast2 = fcast2.rename("Auto Arima")

# 예측 시각화
fig, ax = plt.subplots(figsize=(15, 5))
chart = sns.lineplot(x='new_Date', y='기온(°C)', data=nyc)
chart.set_title('forecast_temparature')
fcast2.plot(ax=ax, color='red', legend=True)
nyc_v.plot(ax=ax, color='blue', legend=True)
plt.savefig("forecast_auto.png")

######### 데이터 수집
temp = []
humi = []


def tick1Min():
    url = 'http://www.kma.go.kr/wid/queryDFSRSS.jsp?zone=4139054000'
    response = requests.get(url)

    temp = re.findall(r'<temp>(.+)</temp>', response.text)
    humi = re.findall(r'<reh>(.+)</reh>', response.text)

    # display = str(temp[0]) + "C" + " " + str(humi[0]) + "%"
    df2 = pd.DataFrame({'temp': temp, 'humi': humi})

    return df2
    # label.config(text=display)
    # window.after(60000, tick1Min)


def message(text):
    print("데이터추출중..")











