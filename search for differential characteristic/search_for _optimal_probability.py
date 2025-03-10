from z3 import *
import numpy as np
from time import time,localtime
#Based on 'Efficient Algorithms for Computing Differential Properties of Addition'
block_size=32#block size
word_size=int(block_size/2)

global f1,f2,f3,f4,f5
f1=0
f2=0
f3=0
f4=0
f5=0
for i in range(int(word_size/4)):
    f1=(f1<<4)+0x5
    f2=(f2<<4)+0x3
    f5=(f5<<4)+0xf
    if(i%2==0):
        f3=(f3<<4)+0
        f4=(f4<<4)+0
    else:
        f3=(f3<<4)+0xf
        f4=(f4<<4)+0x1
#print(hex(f1),hex(f2),hex(f3),hex(f4),hex(f5))
        

def hw(a):#Obtain the hamming weight of 'a' 
    num = (a & f1) + ((a >> 1) & f1)
    num = (num & f2) + ((num >> 2) & f2)
    num = (num & f3) + ((num >> 4) & f3)
    num = ((num * f4) & ((1 << (word_size)) - 1)) >> (word_size-8)
    return num

def Speck_SAT_diff(R,probability,parameter):
    global X,Y
    X=[BitVec('x%d' % i,word_size) for i in range(R+1)]
    Y=[BitVec('y%d' % i,word_size) for i in range(R+1)]
    X1=[BitVec('x_1%d' % i,word_size) for i in range(R)]
    V=[BitVec('v%d' % i,word_size) for i in range(R)]  
    P=[Int('p%d' % i) for i in range(R)]#The probabilities in each round
    
    s=Solver()
    
    #The difference probability is the sum of probabilities in each round
    s.add(probability==Sum(P))
    for i in range(R):
        
        s.add(X1[i]==RotateRight(X[i],parameter[0]))
        
        s.add( (((~(X1[i]<<1)) ^ (Y[i]<<1)) & ((~(X1[i]<<1)) ^ (X[i+1]<<1))) & (X1[i] ^ Y[i] ^ X[i+1] ^ (Y[i]<<1))== 0)
                            
        s.add( V[i] == ~(((~(X1[i])) ^ (Y[i])) & ((~(X1[i])) ^ (X[i+1]))))
        
        s.add(P[i]== BV2Int(hw(V[i]<<1)))
        s.add(Y[i+1] == RotateLeft(Y[i],parameter[1]) ^ X[i+1])

    return s


def find_max_Probability(parameter,Round):
    #Round=10
    Probability=1
    while(1):
        s=Speck_SAT_diff(Round,Probability,parameter)
        flag=s.check()    
        if(flag==sat):
            break
        else:
            Probability=Probability+1
    return Probability

Parameter=np.load('parameter.npy')


Round=6#The number of rounds for optimal differential probability
t0=time()
Path=str(Round)+'round_optimal_Probability'

Pt=[]

for p in Parameter:
    p1=p.tolist()
    Probability=find_max_Probability(p1,Round)
    t1=time()
    print(p,Probability,t1-t0)
    Pt.append(int(Probability))
    t0=time()


np.save(Path+'.npy',np.array(Pt))