import numpy as np 

def featureNormalize(X):
	mu = np.mean(X,axis=0)
	sigma = np.std(X,axis=0)
	for i in range(X.shape[1]):
		X[:,i] = (X[:,i] - mu[i]) / sigma[i]
	return X,mu,sigma
'''
#二维降一维
x = np.array([[0.9,2.4,1.2,0.5,0.3,1.8,0.5,0.3,2.5,1.3],
	[1,2.6,1.7,0.7,0.7,1.4,0.6,0.6,2.6,1.1]])
x = x.T
x,mu,sigma = featureNormalize(x)#1.对样本X进行归一化(正态标准化)
c = np.cov(x.T)			        #2.求每个样本之间的协方差矩阵
w,v = np.linalg.eig(c)          #3.求协方差矩阵的特征值和单位特征向量
v1 = v[:,0] 			        #4.取最大k特征值对应的特征向量构成矩阵P
v1 = v1.T
y =np.dot(x,v1)                 #5.Y = XP   
print(y)

'''

#三维降二维
import numpy as np 

x = np.array([[0.9,2.4,1.2,0.5,0.3,1.8,0.5,0.3,2.5,1.3],
	[1,2.6,1.7,0.7,0.7,1.4,0.6,0.6,2.6,1.1],[1,2.5,1.1,0.6,0.6,1.3,0.5,207,208,1.0]])
x = x.T
x,mu,sigma = featureNormalize(x)
c = np.cov(x.T)
w,v = np.linalg.eig(c) #协方差矩阵的特征值和特征向量
v = v[:,0:2] #降为二维 取前两列
'''
v1,v2 = np.array([v[:,0]]).T,np.array([v[:,1]]).T
#或者v1,v2 = v[:,0].reshape(-1,1),v[:,1].reshape(-1,1)

v = np.hstack((v1,v2)) #矩阵水平拼接 np.vstack()为竖直拼接
#或者v = np.concatenate((v1,v2),axis=1) axis=1为水平拼接，=0为竖直拼接
'''
y = np.dot(x,v)
print(y)

