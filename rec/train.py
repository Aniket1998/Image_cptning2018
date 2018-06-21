import numpy as np
#from scipy.sparse import csr_matrix,csc_matrix
#import pandas as pd
#import matplotlib.pyplot as plt
import time

file=open("./yelp","r")
data=[]
for line in file:
    s=line.replace('\n','\t').split('\t')[0:4]
    ap=[]
    for i in s:
        ap.append(float(i))
    data.append(ap)

data= np.array(data)
data=data[data[:, 3].argsort()]
#data = data[0:int(len(data)/1000)]
l=len(data)
print(l)
train=data[0:int(0.90*l)]
test=data[int(0.90*l):]
print(test.shape)
umax = len(np.unique(data[:,0]))
imax = len(np.unique(data[:,1]))
u_count=len(np.unique(train[:,0]))
i_count=len(np.unique(train[:,1]))

print(umax)
print(imax)
#mat=np.zeros(shape=(u_count+1,i_count+1))
#mat= csr_matrix((umax,imax),dtype =np.int8)
W = {}
count = {}
print(W)
print(count)
uc=0
ic=0
users_map={}
items_map={}

plot_x=[]
plot_y=[]
for t in range(len(train)):
    user=train[t][0]
    item=train[t][1]
    if user in users_map:
        u=users_map[user]
    else:
        users_map[user]=uc
        u=uc
        uc+=1
        count[u]={}
    if item in items_map:
        i=items_map[item]
    else:
        items_map[item]=ic
        i=ic
        ic+=1
        W[i]={}
    if train[t][2]> 0:
    	#mat[u,i]=train[t][2]
    	(W[i])[u]=1
    	(count[u])[i]=train[t][2]

print(W[0])
print(len(W[0]))
c0 = 512
a = 0.4
b = 0.7
f = np.zeros(imax,)
for i in range(i_count):
    f[i]=len(W[i])
fu = np.zeros(umax,)
for u in range(u_count):
    fu[u]= sum([k**b for k in count[u].values()])
for u in range(u_count):
    for i in count[u].keys():
            W[i][u]=len(count[u])*((count[u][i])**b)/fu[u]
print(f)
print(fu)
#r = lil_matrix(W)
f = np.power(f,a)
c = c0*f/np.sum(f)
w_new = 4

K=128
hits=0	
hitratio =[]
prediction_items = {}
prediction_users = {}
Sq = np.zeros((K,K)) #items cache
Sp = np.zeros((K,K)) #users cache
M=u_count #users count
N=i_count#items count
#Randomly initialize P and Q (user feature and item feature matrix)
P=np.random.normal(0,0.01,[M,K])
Q=np.random.normal(0,0.01,[N,K])
#Calculate R_hat usinf eqn 1
# R_hat=np.matmul(P,np.transpose(Q))
lmda = 0.01

def update_user(u):
    global M
    global N
    global K
    global c
    global Sp
    global Sq
    global W
    global count
    global mat
    global P
    global Q
    item_list = list(count[u].keys())
    #item_list = np.nonzero(W[u])[0]
    if len(item_list)==0:
        return
    for i in item_list:
        prediction_items[i] = np.dot(P[u],Q[i])
    for f in range(K):
        numer = 0
        denom = 0
        for k in range(K):
            if k!=f:
                numer -= P[u,k]*Sq[f,k]
        for i in item_list:
            prediction_items[i] -= P[u,f]*Q[i,f]
            numer += ((W[i][u]-(W[i][u]-c[i])*prediction_items[i]))*Q[i,f]
            denom += (W[i][u]-c[i])*Q[i,f]*Q[i,f]

        denom +=Sq[f,f] + lmda
        P[u,f]= numer/denom
        for i in item_list:
            prediction_items[i] += P[u,f]*Q[i,f]

    return

def update_item(i):
    global M
    global N
    global K
    global c
    global Sp
    global Sq
    global W
    global count
    global mat
    global P
    global Q
    user_list = list(W[i].keys())
    #user_list = np.nonzero(W[:,i])[0]
    if len(user_list)==0:
        return
    for u in user_list:
        prediction_users[u] = np.dot(P[u],Q[i])
    for f in range(K):
        numer =0
        denom =0
        for k in range(K):
            if k!=f:
                numer -= Q[i,k]*Sp[f,k]
        numer*=c[i]
        for u in user_list:
            prediction_users[u] -= P[u,f]*Q[i,f]
            numer += ((W[i][u]-(W[i][u]-c[i])*prediction_users[u]))*P[u,f]
            denom += (W[i][u]-c[i])*P[u,f]*P[u,f]

        denom += c[i]*Sp[f,f] + lmda
        Q[i,f]=numer/denom
        for u in user_list:
            prediction_users[u]+= P[u,f]*Q[i,f]

    return

def update_model(u,i,t):
    global M
    global N
    global Sq
    global Sp
    global P
    global Q
    global W
    global count
    global c
    global mat
    global hits
    global hitratio
    global w_new
    new = False
    newu = False
    if u >= M:
        P=np.append(P,np.random.normal(0,0.01,[1,K]),axis=0)
        #W=np.append(W,np.zeros((1,N)),axis=0)
        count[u]={}
        M+=1
        newu = True
    if i >= N:
        Q=np.append(Q,np.random.normal(0,0.01,[1,K]),axis=0)
        #W=np.append(W,np.zeros((M,1)),axis=1)
        #c=np.append(c,[c0/N])
        W[i]={}
        new = True
        N+=1
    W[i][u]=w_new
    if i not in count[u]:
    	count[u][i]=1
    else:
    	count[u][i]+=1
    c[i]=c0/N
    maxscore = np.dot(P[u],Q[i])
    score=[]
    greater = 0
    for item in range(N):
    	score.append(np.dot(P[u],Q[item]))
    	if score[item]>maxscore:
    		greater+=1
    	if greater >= 100:
    		hitratio.append(hits/(t+1))
    		break
    if(greater<100):
    	hits+=1
    	hitratio.append(hits/(t+1))
    if new :
    	Sq+=c[i]*np.matmul(Q[i].reshape(-1,1),Q[i].reshape(1,-1))	
    oldu = P[u]
    update_user(u)
    for f in range(K):
    	for k in range(f+1):
    		if newu:
    			val = Sp[f,k]+(P[u,f]*P[u,k])
    		else:
    			val = Sp[f,k]-(oldu[f]*oldu[k])+(P[u,f]*P[u,k])
    		Sp[f,k]=val
    		Sp[k,f]=val	

    oldi = Q[i] 
    update_item(i)
    Sq = Sq-(c[i]*np.matmul(oldi.reshape(-1,1),oldi.reshape(1,-1)))+(c[i]*np.matmul(Q[i].reshape(-1,1),Q[i].reshape(1,-1)))

    return
print("Offline training started!")
#tart=time.time()
#R_hat=np.matmul(P,np.transpose(Q))
#end=time.time()
#print(end-start)
#print(R_hat)
for j in range(100):
    satisfied=False
    while(not satisfied):
        #update user factors
        Sq = np.zeros((K,K)) #items cache
        Sp = np.zeros((K,K)) #users cache
        for i in range(N):
            Sq+=c[i]*np.matmul(Q[i].reshape(-1,1),Q[i].reshape(1,-1))
        for u in range(M):
            update_user(u)
        Sp = np.matmul(P.T,P)
        for i in range(N):
            update_item(i)
                 #iteratively update
        satisfied=True
    print(j)
    #R_hat=np.matmul(P,np.transpose(Q))
    #print(R_hat)
#p_df=pd.DataFrame(P)
#q_df=pd.DataFrame(Q)
print("Saving P and Q")
#p_df.to_csv('./p.csv')
#q_df.to_csv('./q.csv')
#P = np.load('P.npy')
#Q = np.load('Q.npy')

np.save('./P',P)
np.save('./Q',Q)
start = time.time()
R_hat=np.matmul(P,np.transpose(Q))
end = time.time()
print(end-start)
start = time.time()
R_hat=np.matmul(P[0],np.transpose(Q))
end = time.time()
print(end-start)
start = time.time()
R_hat=np.matmul(P,np.transpose(Q[0]))
end = time.time()
print(end-start)

#rhat=pd.DataFrame(R_hat)
#rhat.to_csv('./rhat.csv')
print(R_hat)
print("online training started")
for t in range(len(test)):
	user_str=test[t][0]
	item_str=test[t][1]
	user=0
	item=0
	if(not (user_str in users_map)):
		user=M
		users_map[user_str]=M
	else:
		user=users_map[user_str]
	if(not (item_str in items_map)):
		item=N
		items_map[item_str]=N
	else:
		item=items_map[item_str]
	update_model(user,item,t)
	plot_x.append(t)
	print(t)
	print(hitratio[t])
#plt.plot(x_plot,hitratio)
np.save("./hitratio",hitratio)
np.save("./x_plot",plot_x)
#np.save('./W',W)
#rr=pd.DataFrame(R)
#rr.to_csv('./R.csv')
print("online training done!")



