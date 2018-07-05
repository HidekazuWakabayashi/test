import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot
from scipy.interpolate import interp1d
import math

# サンプル点の個数
N = 300

# ファイルの読み込み
lst = pd.read_csv("sample2.csv").values

# 変換行列
RGB = np.array([[0.4898, 0.3101, 0.2001],
                [0.1769, 0.8124, 0.0107],
                [0.0000, 0.0100, 0.9903]])

# 逆行列
inv_RGB = np.linalg.inv(RGB)

# 列の切り出し
X = lst[:,1]
Y = lst[:,2]
Z = lst[:,3]

# 白色点の計算
X_max = sum(X)
Y_max = sum(Y)
Z_max = sum(Z)
S_max = X_max + Y_max + Z_max
Xp = X_max / S_max
Yp = Y_max / S_max
Zp = Z_max / S_max

# 正規化
S = lst[:,1] + lst[:,2] + lst[:,3]
X = lst[:,1] / S
Y = lst[:,2] / S
Z = lst[:,3] / S

# centerRGB = inv_RGB.dot([Xp,Yp,Zp])
# print Xp


# pyplot.plot(X,Y,'-')

print "---end plot1---"

# 直線の計算
s1 = 0.15958146
t1 = 0.0158926119617
u1 = 8.20516825e-01

s2 = 0.723291748157
t2 = 0.27670825
u2 = 0

x_lin = np.linspace(s1,s2,N)
y_lin = np.linspace(t1,t2,N)
z_lin = np.linspace(u1,u2,N)
# pyplot.plot(x_lin,y_lin,'-')
print "---end plot2---"

lin1=np.array([])
lin2=np.array([])
lin3=np.array([])

# 座標の取得(1)
for i in range(len(X)):
    tmp1 = np.linspace(X[i], Xp, N)
    tmp2 = np.linspace(Y[i], Yp, N)

    lin1 = np.append(lin1,tmp1)
    lin2 = np.append(lin2,tmp2)

# 白色点のパラメータ
Ya = 0.7
Ox = 10
Oy = 10

Y1 = lin2
Y1 = Y1 + Ya*np.exp(-((lin1 - Xp)**2 / 2*(Ox**2)) - ((lin2 - Yp)**2 / 2*(Oy**2)) )
X1 = lin1 / lin2 * Y1
Z1 = ((1-lin1-lin2)/lin2)*Y1

# RGBの取得(1)
col_list = []
for i in range(len(X1)):
    col_list.append(inv_RGB.dot([X1[i],Y1[i],Z1[i]]))

col_list = [[0.0 if elm < 0.0 else 1.0 if elm > 1.0 else elm for elm in v]for v in col_list]

# print col_list

pyplot.scatter(lin1, lin2, c = col_list, marker='x', alpha = 0.5)
print "---end plot3---"

lin1=np.array([])
lin2=np.array([])
lin3=np.array([])

# 座標の取得(2)
for i in range(len(x_lin)):
    tmp1 = np.linspace(x_lin[i], Xp, N)
    tmp2 = np.linspace(y_lin[i], Yp, N)

    lin1 = np.append(lin1,tmp1)
    lin2 = np.append(lin2,tmp2)

Y1 = lin2
Y1 = Y1 + Ya*np.exp(-((lin1 - Xp)**2 / 2*(Ox**2)) - ((lin2 - Yp)**2 / 2*(Oy**2)) )
X1 = lin1 / lin2 * Y1
Z1 = ((1-lin1-lin2)/lin2)*Y1

# RGBの取得(2)
col_list = []
for i in range(len(X1)):
    col_list.append(inv_RGB.dot([X1[i],Y1[i],Z1[i]]))

col_list = [[0.0 if elm < 0.0 else 1.0 if elm > 1.0 else elm for elm in v]for v in col_list]

# 散布図のプロット
pyplot.scatter(lin1, lin2, c = col_list, marker='x', alpha = 0.5)
print "---end plot4---"

# 画像ファイルの出力
pyplot.savefig('figure.png')
pyplot.show()
