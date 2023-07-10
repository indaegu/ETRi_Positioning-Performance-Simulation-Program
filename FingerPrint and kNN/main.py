import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D




''' Data Load ''' #현재는 데이터가 없어서 실행은 불가능
DB_File = 'DB'
FP_File = 'FP_TestResult_15k'
PDR_File = 'PDR_Data1'
Result_ = pd.read_csv(f'C:\project\{FP_File}.csv', header=None)
Result = Result_.to_numpy()
DB_ = pd.read_csv(f'C:\project\{DB_File}.csv', header=None)
DB = DB_.to_numpy()
PDR_ = pd.read_csv(f'C:\project\{PDR_File}.csv', header=None)
PDR = PDR_.to_numpy()

''' Parameters '''
BT_Num = 11
RP_Point = len(DB)
PDR_Point = len(PDR)
K = 15

''' Database Figure Plot '''
X_Value = DB[:, 0]
Y_Value = DB[:, 1]
Z_Value = DB[:, 2:]


''' FP Figure Plot '''
Stride_Value = PDR[:, 0]
Radian_Value = PDR[:, 1]
RSS_Value = PDR[:, 2:]

# PDR + 핑거프린트 #
# RESULT1 = []
# row_sum_matrix = []
# PDR_Location = []
# KalmanGain = 0
# H = np.array([[0, 0], [0, 0]])
# Kalman_Location = []
# SL = 0.62
# Q = 0.1
# R = 1 - Q
# for a in range(PDR_Point-1): # PDR.csv의 행의 개수만큼 반복
#     if a == 0:
#         XL = DB[0, 0] + SL * math.cos(PDR[a, 1])
#         YL = DB[0, 1] + SL * math.sin(PDR[a, 1])
#         PDR_Location.append([XL, YL])
#     XL = XL + SL * math.cos(PDR[a, 1])
#     YL = YL + SL * math.sin(PDR[a, 1])
#     PDR_Location.append([XL, YL])
#     pp = np.array(PDR_Location)
#     for o in range(RP_Point-1): # DB.csv의 행의 개수만큼 반복
#         inner_sum = 0
#         count = 0
#
#         for i in range(2, BT_Num + 2): # PDR과 DB의 11개의 RSS값을 비교하기 위함
#             if PDR[a, i] != -100:  # 측정 되지 않은 RSS값은 연산 하지 않기 위함
#                 inner_sum = inner_sum + math.sqrt((PDR[a, i] - DB[o, i]) ** 2)
#                 count += 1 # 연산 할때만 count 개수를 받아옴
#         row_sum = inner_sum / count # inner_sum을 모두 합하고 연산한 횟수 만큼나 나누어 줘서 i행의 값을 구함
#         row_sum_matrix.append([row_sum, DB[o, 0], DB[o, 1]]) # i행의 값과 DB의 X,Y 좌표를 리스트로 저장
#     row_sum_matrix.sort(key=lambda x: x[0]) # row_sum을 기준으로 리스트를 정렬
#     zz = np.array(row_sum_matrix) # 정렬된 리스트를 배열로 생성
#     SumX = 0
#     SumY = 0
#     sigma = 0
#     for c in range(K):
#         sigma = sigma + (1/zz[c, 0])
#     W = 0
#     for j in range(K):
#         W = (1/zz[j, 0])/sigma
#         SumX += (W * zz[j, 1]) # X좌표를 K개 만큼 더 해줌
#         SumY += (W * zz[j, 2]) # Y좌표를 K개 만큼 더 해줌
#     RESULT1.append([SumX, SumY]) # 평균낸 X,Y좌표값을 리스트에 추가해줌
#     jj = np.array(RESULT1) # 리스트를 배열로 생성해줌
#     Kalman_Location.append([Q*jj[a, 0]+R*XL, Q*jj[a, 1]+R*YL])
#     kk = np.array(Kalman_Location)
#     # print(Kalman_Location)
# plt.figure() # 2차원 좌표평면에 그리기
# for p in range(PDR_Point-1):
#     plt.plot(X_Value, Y_Value, 'k.', label='RP (DB Point)') # DB
#     #print(f'{p}번째 X좌표 = ', jj[p, 0], 'Y좌표 = ', jj[p, 1])
#     plt.plot(pp[p, 0], pp[p, 1], 'ro', label=f'MY PDR Result (K = {K})') # MY
#     plt.plot(Result[p, 0], Result[p, 1], 'bo', label='kNN Result (K = 15)') # 이정호 박사님
#     plt.legend()
#     plt.xlabel('X [m]')
#     plt.ylabel('Y [m]')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.pause(0.01)
#     plt.clf()
# plt.show()


# WkNN 구현 #
# RESULT1 = []
# for a in range(PDR_Point-1): # PDR.csv의 행의 개수만큼 반복
#     row_sum_matrix = []
#     for o in range(RP_Point-1): # DB.csv의 행의 개수만큼 반복
#         inner_sum = 0
#         count = 0
#         for i in range(2, BT_Num + 2): # PDR과 DB의 11개의 RSS값을 비교하기 위함
#             if PDR[a, i] != -100:  # 측정 되지 않은 RSS값은 연산 하지 않기 위함
#                 inner_sum = inner_sum + math.sqrt((PDR[a, i] - DB[o, i]) ** 2)
#                 count += 1 # 연산 할때만 count 개수를 받아옴
#         row_sum = inner_sum / count # inner_sum을 모두 합하고 연산한 횟수 만큼나 나누어 줘서 i행의 값을 구함
#         row_sum_matrix.append([row_sum, DB[o, 0], DB[o, 1]]) # i행의 값과 DB의 X,Y 좌표를 리스트로 저장
#     row_sum_matrix.sort(key=lambda x: x[0]) # row_sum을 기준으로 리스트를 정렬
#     zz = np.array(row_sum_matrix) # 정렬된 리스트를 배열로 생성
#     SumX = 0
#     SumY = 0
#     sigma = 0
#     for c in range(K):
#         sigma = sigma + (1/zz[c, 0])
#     W = 0
#     for j in range(K):
#         W = (1/zz[j, 0])/sigma
#         SumX += (W * zz[j, 1]) # X좌표를 K개 만큼 더 해줌
#         SumY += (W * zz[j, 2]) # Y좌표를 K개 만큼 더 해줌
#     RESULT1.append([SumX, SumY]) # 평균낸 X,Y좌표값을 리스트에 추가해줌
#     jj = np.array(RESULT1) # 리스트를 배열로 생성해줌
#
# plt.figure() # 2차원 좌표평면에 그리기
# for p in range(PDR_Point-1):
#     plt.plot(X_Value, Y_Value, 'k.', label='RP (DB Point)') # DB
#     print(f'{p}번째 X좌표 = ', jj[p, 0], 'Y좌표 = ', jj[p, 1])
#     plt.plot(jj[p, 0], jj[p, 1], 'ro', label=f'MY WkNN Result (K = {K})') # MY
#     plt.plot(Result[p, 0], Result[p, 1], 'bo', label='kNN Result (K = 15)') # 이정호 박사님
#     plt.legend()
#     plt.xlabel('X [m]')
#     plt.ylabel('Y [m]')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.pause(0.01)
#     plt.clf()
# plt.show()

# #kNN 구현 #
RESULT1 = []
for a in range(PDR_Point-1): # PDR.csv의 행의 개수만큼 반복
    row_sum_matrix = []
    for o in range(RP_Point-1): # DB.csv의 행의 개수만큼 반복
        inner_sum = 0
        count = 0
        for i in range(2, BT_Num + 2): # PDR과 DB의 11개의 RSS값을 비교하기 위함
            if PDR[a, i] != -100:  # 측정 되지 않은 RSS값은 연산 하지 않기 위함
                inner_sum = inner_sum + math.sqrt((PDR[a, i] - DB[o, i]) ** 2)
                count += 1 # 연산 할때만 count 개수를 받아옴
        row_sum = inner_sum / count # inner_sum을 모두 합하고 연산한 횟수 만큼나 나누어 줘서 i행의 값을 구함
        row_sum_matrix.append([row_sum, DB[o, 0], DB[o, 1]]) # i행의 값과 DB의 X,Y 좌표를 리스트로 저장
    row_sum_matrix.sort(key=lambda x: x[0]) # row_sum을 기준으로 리스트를 정렬
    zz = np.array(row_sum_matrix) # 정렬된 리스트를 배열로 생성
    SumX = 0
    SumY = 0
    for j in range(K):
        SumX += zz[j, 1] # X좌표를 K개 만큼 더 해줌
        SumY += zz[j, 2] # Y좌표를 K개 만큼 더 해줌
    SumX_ = SumX/K # X좌표를 K개로 나누어 평균 내줌
    SumY_ = SumY/K # Y좌표를 K개로 나누어 평균 내줌
    RESULT1.append([SumX_, SumY_]) # 평균낸 X,Y좌표값을 리스트에 추가해줌
    jj = np.array(RESULT1) # 리스트를 배열로 생성해줌

plt.figure() # 2차원 좌표평면에 그리기
for p in range(PDR_Point-1):
    plt.plot(X_Value, Y_Value, 'k.', label='RP (DB Point)') # DB
    print('X좌표 = ', jj[p, 0], 'Y좌표 = ', jj[p, 1])
    plt.plot(jj[p, 0], jj[p, 1], 'ro', label=f'MY kNN Result (K = {K})') # MY
    plt.plot(Result[p, 0], Result[p, 1], 'bo', label='kNN Result (K = 15)') # 이정호 박사님
    plt.legend()
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.pause(0.01)
    plt.clf()
plt.show()


# # NN 구현#
# plt.figure()
# for a in range(0, PDR_Point-1):
#     row_sum_matrix = []
#     for o in range(0, RP_Point - 1):
#         inner_sum = 0
#         count = 0
#         for i in range(2, BT_Num + 2):
#             if PDR[0, i] != -100:
#                 inner_sum = inner_sum + math.sqrt((PDR[a, i] - DB[o, i]) ** 2)
#                 count = count + 1
#         row_sum = inner_sum / count
#         row_sum_matrix.append([row_sum, DB[a, 0], DB[a, 1]])
#     print(f'PDR {a}행 X좌표 = ', DB[row_sum_matrix.index(min(row_sum_matrix)), 0], f'Y좌표 = ', DB[row_sum_matrix.index(min(row_sum_matrix)), 1])
#     plt.xlabel('X [m]')
#     plt.ylabel('Y [m]')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.plot(X_Value, Y_Value, 'k.', label='RP (DB Point)')  # DB 2차원 경로
#     plt.plot(DB[row_sum_matrix.index(min(row_sum_matrix)), 0], DB[row_sum_matrix.index(min(row_sum_matrix)), 1], 'ro', label='kNN Result')  # NN 2차원 경로
#     plt.pause(0.1)
#     plt.clf()
#     row_sum_matrix = []
# plt.show()


# n = 101
# sum = 0
# for i in range( 2, n ) :
#     inner_sum = 0
#     for j in range( 1, i ) :
#         inner_sum += j
#     inner_sum *= i
#     sum += inner_sum
# sum *= 2
# print( sum )
#
# print(PDR[0, 7])


# Database 2차원 경로 확인 ##
# plt.figure()
# plt.plot(X_Value, Y_Value, 'k.')
# plt.xlabel('X [m]')
# plt.ylabel('Y [m]')
# plt.axis('equal')
# plt.grid(True)
# plt.show()

# # Database 3차원 RSS 패턴 확인 ##
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(BT_Num):
#     str = 'Beacon {0:02d}'.format(i)
#     ax.plot(X_Value, Y_Value, Z_Value[:, i], '.-', label=str)
#     plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
# plt.axis('equal')
# plt.grid(True)
# ax.set_ylabel("Y") # 레이블 이름 수정
# ax.set_xlabel("X")
# ax.set_zlabel("RSS")
# plt.show()

''' 측위 결과 확인 '''
# plt.figure()
# for i in range(Result.shape[0]):
#     plt.plot(X_Value, Y_Value, 'k.', label='RP (DB Point)')
#     plt.plot(Result[i, 0], Result[i, 1], 'ro', label='kNN Result')
#     plt.legend()
#     plt.xlabel('X [m]')
#     plt.ylabel('Y [m]')
#     plt.axis('equal')
#     plt.grid(True)
#     plt.pause(0.01)
#     plt.clf()
# plt.show()
