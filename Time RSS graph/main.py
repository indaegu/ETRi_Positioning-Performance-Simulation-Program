import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# import csv
folderPath = r'C:\project' # 폴더 경로
attributeTable = 'UserTest_3.csv' # 파일 이름: UserTest_2.csv = RSS값 / UserTest_3.csv = 중복 제거값
os.chdir(folderPath)
df = pd.read_csv(attributeTable)
df = df[['time', 'RSS1', 'RSS2', 'RSS3', 'RSS4', 'RSS5', 'RSS6', 'RSS7', 'RSS8', 'RSS9', 'RSS10', 'RSS11']]

# RSS 개별 그래프

for i in range(1, 12):
    plt.figure(figsize=(12.7, 6))
    plt.rcParams.update({'font.size': 18})
    ax = df.set_index('time')[f'RSS{i}'].plot(kind='line', title=f'시간에 따른 RSS{i} 변화', marker='.') # RRS 번호 설정
    ax.set_ylabel("RSS{i}")
    ax.set_xlabel("Time")
    plt.show()


# RSS 종합 그래프
plt.figure(figsize=(12.7, 6))
plt.rcParams.update({'font.size': 18})
for i in range(1, 12):
    ax = df.set_index('time')[f'RSS{i}'].plot(kind='line', title='시간에 따른 RSS 변화 종합', marker='.')  # RRS 번호 설정
ax.set_ylabel("RSS")
ax.set_xlabel("Time")
plt.show()







