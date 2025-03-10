from z3 import *
import numpy as np
from time import time,localtime
#Based on 'Efficient Algorithms for Computing Differential Properties of Addition'
block_size=32
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
        

def hw(a):
    num = (a & f1) + ((a >> 1) & f1)
    num = (num & f2) + ((num >> 2) & f2)
    num = (num & f3) + ((num >> 4) & f3)
    num = ((num * f4) & ((1 << (word_size)) - 1)) >> (word_size-8)
    return num

def Speck_SAT_diff(Round,Probability,parameter):
    global X,Y
    X=[BitVec('x%d' % i,word_size) for i in range(Round+1)]
    Y=[BitVec('y%d' % i,word_size) for i in range(Round+1)]
    X1=[BitVec('x_1%d' % i,word_size) for i in range(Round)]
    V=[BitVec('v%d' % i,word_size) for i in range(Round)]
    P=[Int('p%d' % i) for i in range(Round)]
    
    s=Solver()

    s.add(Probability==Sum(P))
    for i in range(Round):
        
        s.add(X1[i]==RotateRight(X[i],parameter[0]))
        
        s.add( (((~(X1[i]<<1)) ^ (Y[i]<<1)) & ((~(X1[i]<<1)) ^ (X[i+1]<<1))) & (X1[i] ^ Y[i] ^ X[i+1] ^ (Y[i]<<1))== 0)
                            
        s.add( V[i] == ~(((~(X1[i])) ^ (Y[i])) & ((~(X1[i])) ^ (X[i+1]))))
        
        s.add(P[i]== BV2Int(hw(V[i]<<1)))
        s.add(Y[i+1] == RotateLeft(Y[i],parameter[1]) ^ X[i+1])

    return s

def find_path(PARAMETER,ROUND,Prob):
    diff=[]
    print('Round=',ROUND,sep='',end='\n')
    print('Probability=',Prob,sep='',end='\n')
    print('parameter=',PARAMETER,sep='',end='\n')
    s=Speck_SAT_diff(ROUND,Prob,PARAMETER)

    flag=s.check()
    if(flag==sat):
        while(flag==sat):
            m = s.model()
            x=[]
            y=[]
            for i in range(ROUND+1):
                x.append(int(str(m[X[i]])))
                y.append(int(str(m[Y[i]])))
            print(hex(x[0]),hex(y[0]))
            diff.append((x[0],y[0]))
            np.save('SPECK32_Round='+str(ROUND)+'_parameter='+str(PARAMETER)+'.npy',np.array(diff,dtype=np.uint64))
            s.push()        
            s.add(Or(And(X[0] != m.eval(X[0]),Y[0]!= m.eval(Y[0])),  And(X[0] == m.eval(X[0]),Y[0]!= m.eval(Y[0]))  , And(X[0] != m.eval(X[0]),Y[0]== m.eval(Y[0])) ))
            flag=s.check()

t0=time()

Round=5
Parameter=np.load('parameter.npy')
Pro=np.load(str(Round)+'Round_max_Probability.npy')

for i in range(len(Parameter)):

    find_path(Parameter[i].tolist(),Round,int(Pro[i]))
    t1=time()
    print(t1-t0)
    t0=time()