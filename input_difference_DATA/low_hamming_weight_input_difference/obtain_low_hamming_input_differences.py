# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 19:47:14 2021

@author: ZeZhou Hou
"""

import numpy as np
  
def hw(v):
    res = np.zeros(v.shape,dtype=np.uint8);
    for i in range(16):
        res = res + ((v >> i) & 1)
    return res

x=1
Round=7

r=[]
for i in range(x+1):
    r.append((i,x-i))
print(r)
diff=[]
for i in r:
    low_weight = np.array(range(2**16), dtype=np.uint16)
    low_weight_l = np.array(low_weight[hw(low_weight) == i[0]], dtype=np.uint16) 
    low_weight_r = np.array(low_weight[hw(low_weight) == i[1]], dtype=np.uint16) 
    
    ll=len(low_weight_l)
    lr=len(low_weight_r)
    
    low_weight_l = np.tile(low_weight_l, lr) 
    low_weight_r = np.repeat(low_weight_r, ll)
    
    result=np.vstack((low_weight_l,low_weight_r))
    result=result.T
    result=result.tolist()
    diff=diff+result



for i in diff:
    l=list(bin(i[0])[2:])
    r=list(bin(i[1])[2:])
    
    l=np.array(l,dtype=np.uint16)
    r=np.array(r,dtype=np.uint16)
    
    if((sum(l)+sum(r))!=x):
        print(i)

#no duplicate elements
for i in range(len(diff)-1):
    flag=0
    if(i%1000==0):
        print("Progress Bar：",i/1000)
    for j in range(i+1,len(diff)):
        if(diff[i]==diff[j]):
            flag=flag+1
    if(flag>1):
        print(i)
 
np.save('SPECK32_Round='+str(Round)+'hw='+str(x)+'.npy',np.array(diff,dtype=np.uint64))