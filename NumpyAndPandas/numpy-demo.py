import numpy as np

# array = np.array([[1,2,3],
#  [2,3,4]])
# print(array)
# print(array.size)
# print(array.shape)
# print(array.ndim)
# print(array.argmax())
# print("############################")
# a = np.array([2,22,44],dtype=np.int)
# print(a.dtype)  # 更精确需要更多空间
#
# v = np.array([[2,22,222],[3,33,333]])
# print(v)
#
# zero_v = np.zeros((3,4))
# print(zero_v)
#
# one_v = np.ones((3,2))
# print(one_v)
#
# empty_v = np.empty((3,3))
# print(empty_v)
#
# b = np.arange(12).reshape((3,4))
# print(b)
#
# line = np.linspace(1,10,20)
# print(line)
#
# line=line.reshape(2,10)
# print(line)
#
# a = np.array([10,20,30,40])
# b = np.arange(4)
# print(a,b)
# c = a+b
# print(c)
# d = 10*np.sin(a)
# print(d)
#
# a = np.array([[1,2],[3,4]])
# b = np.arange(4).reshape((2,2))
#
# c = a*b
# d = np.dot(a,b)
# e = a.dot(b)
# print(c)
# print(d)
# print(e)
#
# a = np.random.random((2,4))
# print(a)
# print(np.sum(a,axis=1))
# print(np.min(a))
# print(np.max(a))

# A = np.arange(14,2,-1).reshape((3,4))
# print(np.argmax(A))  #寻找到索引
# print(A)
# print(A.mean())
# print(np.median(A))  # 中位数
# print(np.cumsum(A))  # 逐步累加
# print(np.diff(A))    # 逐步累差
# print(np.nonzero(A)) # (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
# print(np.sort(A))    # 逐行排序
# print(A.T)           # 矩阵转置
# print(np.clip(A,5,9))# 矩阵截取
# print(np.mean(A,axis=0)) # 0 是列的运算，1 是行的运算

# A = np.arange(3,15)
# print(A)
# print(A[3])
# A = A.reshape(3,4)
# print(A)
# print(A[2])
# print(A[2][1])
# print(A[2,1])
# print(A[2,:])
# print(A[:,2])

# for row in A:
#     print(row)  # 默认迭代行
#
# for col in A.T:  # 迭代列，只要将矩阵转置
#     print(col)

# print(A.flatten())
#
# for element in A.flat:
#     print(element)


A = np.array([1,1,2])
B = np.array([2,2,2])
C = np.vstack((A,B))
D = np.hstack((A,B))
print(C)
print(D)
print(A)
print(A[np.newaxis,:].shape)
print(A[:,np.newaxis])
print(np.hstack((A[:,np.newaxis],B[:,np.newaxis])))

C = np.concatenate((A,B,B,A),axis=0)
print(C)

A = np.arange(12).reshape((3,4))
#  分割
print(np.split(A,2,axis=1))
print(np.array_split(A,3,axis=1))  #不等量分割
print(np.vsplit(A,3))
print(np.hsplit(A,2))

a = np.arange(4)
b = a
c = a
d = b
a[0] = 111
print(a)
print(a is b)

e = a.copy()   # 只是赋值，没有关联起来，deep copy
print(e is a)
