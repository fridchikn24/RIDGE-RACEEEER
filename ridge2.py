import numpy as np
import pandas as pd


n = 100
sim_data = pd.DataFrame({'intercept': 1,'x1': np.random.normal(size=n, ),'x2': np.random.normal(size=n, ),
	'x3': np.random.normal(size=n, ),'x4': np.random.normal(size=n, ),'y': np.random.normal(size=n, )})
lambda_list = [.01, .05,.1,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.99]


y = np.array(sim_data['y'])
x = np.array(sim_data[['intercept','x1','x2','x3','x4']])
columnz = ['lambda','SSE', 'alpha', 'b1','b2','b3','b4']
dfw = pd.DataFrame(index = range(0,15),columns = columnz)

for i in range(len(lambda_list)):
	lamb = lambda_list[i]
	xl = np.array(x.T@x + lamb*np.identity(len(x.T@x)))
	br = np.linalg.solve(xl,x.T@y)
	yhat = x@br
	sse = np.sum(((yhat-y)**2))
	dfw.iloc[i] = [lamb,sse,br[0],br[1],br[2],br[3],br[4]]

z = pd.DataFrame(dfw['SSE'])
u = min(dfw["SSE"])
print(dfw.loc[dfw["SSE"] == u])



