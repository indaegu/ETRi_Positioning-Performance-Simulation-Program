import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:\project\DB2.csv') # DB2.csv = x + y / PDR_Data2.csv = Stride + rad
x = data['x']
y = data['y']
for i in range(1, 12): # 반복문을 이용한 변수 선언
    globals()['z'+str(i)] = data[f'RSS{i}']
# 반복문을 이용한 각각의 RSS 출력
"""
for i in range(1, 12):
    fig = plt.figure(figsize=(8, 7))  # 출력창 크기 조절
    ax = fig.add_subplot(projection='3d')  # 그래프 출력 모드 3d로 변경
    ax.view_init(10, 60)  # 그래프가 보이는 각도 지정, 출력 후 마우스로 임의 지정 가능
    ax.scatter(x, y, eval('z' + str(i)), marker='.', s=20)
    ax.set_ylabel("Y")  # 레이블 이름 수정
    ax.set_xlabel("X")
    ax.set_zlabel("RSS")
    plt.show()
"""
# 반복문을 이용한 모든 RSS 출력
fig = plt.figure(figsize=(8, 7)) # 출력창 크기 조절
ax = fig.add_subplot(projection='3d') # 그래프 출력 모드 3d로 변경
ax.view_init(10, 60) # 그래프가 보이는 각도 지정, 출력 후 마우스로 임의 지정 가능
for i in range(1, 12):
    ax.scatter(x, y, eval('z'+str(i)), marker='.', s=20)
ax.set_ylabel("Y") # 레이블 이름 수정
ax.set_xlabel("X")
ax.set_zlabel("RSS")
plt.show()
