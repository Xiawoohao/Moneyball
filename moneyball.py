import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf



# 讀取 CSV 資料（注意使用 r"..." 或自行雙反斜線避免路徑錯誤）
base = pd.read_csv(r"C:/Users/jg991/OneDrive/桌面/上課文件/進階程式 人工智慧/人工智慧期末專題/baseball.csv")

# 顯示原始資料的前5列
print("原始資料預覽：")
print(base.head())

# 篩選出 2002 年以前的資料（1962–2001）
dpyears = base[base.Year < 2002]

# 顯示篩選後的前5列資料
print("\n2002年前（DePodesta 可用）的資料預覽：")
print(dpyears.head())
#結果顯示過濾後剩下 902 筆資料，這就是後續分析所用的訓練資料。

#這段會回傳 2001 年有多少支 MLB 球隊（回傳 30，符合當年實際球隊數量）。
teamcount = base[base.Year == 2001]['Team'].count()
print("2001年有多少球隊:",teamcount)

 #number of observations decreased because of year range
print("查看篩選後每個欄位的有效資料數量")
print(dpyears.count())

# 95勝以上的球隊數量
print("勝場超過或等於95場的球隊數：", base[base.W >= 95].shape[0])

# 其中有進季後賽的比例
high_win_teams = base[base.W >= 95]
playoff_rate = high_win_teams.Playoffs.mean()
print("這些球隊進入季後賽的比例：", playoff_rate)

qualifiedwins = base[base.Year < 2002]
# qualifiedwins
qualifiedwinsnew = qualifiedwins[['Team','Year','W','Playoffs']]
# qualifiedwinsnew = qualifiedwinsnew[qualifiedwinsnew.Playoffs == 1]
print(qualifiedwinsnew)

plt.figure(figsize=(10,9))
ax = sns.scatterplot(x="W", y="Team", hue="Playoffs",data=qualifiedwinsnew)
plt.plot(95, 0, color='r')

#使用 seaborn 和 matplotlib 繪製一張 球隊每年勝場數與是否進入季後賽的散佈圖，並標出 DePodesta 所認為的「進入季後賽門檻：95 勝」。
plt.title("MLB Teams: Wins vs. Playoffs Qualification (Before 2002)")
plt.legend()
plt.show()

runs = base[base.Year < 2002]
runs.info()
runs['RD'] = runs['RS'] - runs['RA']
runs.info()
#畫出得失分差（RD）與勝場數（W）的散點圖與二次多項式回歸曲線。

#圖形顯示勝場數跟得失分差關係非常強，符合直覺：得失分差越高，勝場越多。
sns.lmplot(x ="RD", y ="W", data = runs, order = 2, ci = None)
plt.legend()
plt.show()
#建立簡單線性回歸模型：Wins = Intercept + Slope * RD

#R² 約 0.885，代表模型解釋了約 88.5% 勝場數的變異，非常好。

#預測 RD=133 得到勝場約 95，吻合 DePodesta 預測。

X = np.array(runs['RD']).reshape(-1, 1) 
y = np.array(runs['W']).reshape(-1, 1) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 

regr = LinearRegression() 
regr.fit(X_train, y_train)

print("R^2",regr.score(X_test, y_test))  # R^2 約 0.885，模型解釋力高

regr.predict(np.array(133).reshape(-1, 1))  # 預測 RD=133 時勝場約 95

print("預測勝場",regr.predict(np.array(133).reshape(-1, 1)))

model=smf.ols(formula ='W ~ RD',data=runs).fit().summary()
print(model)

obp = runs[runs.Year == 2001][runs.Team == 'OAK'][['Team','OBP','SLG']]
print(obp)


model=smf.ols(formula ='RA ~ OOBP + OSLG',data=runs).fit().summary()
print(model)

oobp = runs[runs.Year == 2001][runs.Team == 'OAK'][['Team','OOBP','OSLG']]
print(oobp)