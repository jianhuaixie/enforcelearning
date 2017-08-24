import pandas as pd
import numpy as np

s = pd.Series([14,6,9,np.nan,44,55])
print(s)

dates = pd.date_range('20160101',periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
print(df)
df1 = pd.DataFrame(np.arange(12).reshape((3,4)))  # 默认行和列
print(df1)

print(df1.dtypes)
print(df1.index)
print(df1.columns)
print(df1.values)
print(df1.describe())  # 运算一些数字类型
print(df1.T)
print(df1.sort_index(axis=1,ascending=False))
print(df1.sort_values(by=1))

print(df['a'],df.a)

print("#################")
# print(df[0:3])
# select by label:loc
# print(df.loc['20160102'])
# print(df.loc[:,['a','b']])
# select by position
# print(df.iloc[3])
# print(df.iloc[1:3,:])
# mixed selection :ix
# print(df.ix[:3,['a','d']])
# Boolean selection
# print(df[df.a>0])

# df.iloc[2,2] = 1111
# df[df.a>0] = 0
# df.a[df.a>0] = 0
# df['e'] = np.nan
# print(df)


df.iloc[0,1] =  np.nan
print(df)
# print(df.dropna(axis=0,how='any'))  # how=['any','all']
# print(df.fillna(value=0))
print(df.isnull())
print(np.any(df.isnull()) == True)
print(np.all(df.isnull()) == True)

# data = pd.read_csv('D:\\wiair\\URL2.csv')
# print(data)

# concatnating
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)
res2 = pd.concat([df1,df2,df3],axis=1,ignore_index=True)
# print(res)
# print(res2)

# join,['inner','outer']
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'])
res = pd.concat([df1,df2])  # outer
# print(res)
res2 = pd.concat([df1,df2],join='inner',ignore_index=True)
# print(res2)

# join_axes
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
res3 = pd.concat([df1,df2],axis=1)
res4 = pd.concat([df1,df2],axis=1,join_axes=[df1.index])
# print(res3)
# print(res4)

# append
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
res = df1.append([df2,df3],ignore_index=True)
# print(res)

# merging two df by key/keys (may be used in database)
# simple example
left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                      'A':['A0','A1','A2','A3'],
                      'B':['B0','B1','B2','B3']})
right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                      'C':['C0','C1','C2','C3'],
                      'D':['D0','D1','D2','D3']})
print(left)
print(right)
res = pd.merge(left,right,on='key')
# print(res)
# consider two keys
left = pd.DataFrame({'key1':['K0','K0','K1','K2'],
                     'key2':['K0','K1','K0','K1'],
                      'A':['A0','A1','A2','A3'],
                      'B':['B0','B1','B2','B3']})
right = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                      'key2':['K0','K0','K0','K0'],
                      'C':['C0','C1','C2','C3'],
                      'D':['D0','D1','D2','D3']})
res = pd.merge(left,right,on=['key1','key2'])  # inner
print(res)
res2 = pd.merge(left,right,how='outer',on=['key1','key2'])
print(res2)

# indicator
df1 = pd.DataFrame({'col1':[0,1],'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print(df1)
print(df2)
res = pd.merge(df1,df2,on='col1',how='outer',indicator=True)
print(res)

# merged by index


import matplotlib.pyplot as plt
# plot data
# Series
data = pd.Series(np.random.randn(1000),index=np.arange(1000))
data = data.cumsum()
# data.plot()
# plt.show()

# DataFrame
data = pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list("ABCD"))
data = data.cumsum()
print(data.head())
# data.plot()
# plot methods:
# 'bar','hist','box','kde','area','scatter','hexxbin','pie'
ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class_1')
data.plot.scatter(x='A',y='C',color='DarkGreen',label='Class_2',ax=ax)
plt.show()

