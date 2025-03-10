# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 20:40:07 2021

@author: ZeZhou Hou
"""
import numpy as np
result=[]
for b in range(16):
    for a in range(16):
        result.append([a,b])
new_result=np.array(result,dtype=np.uint32)
np.save('parameter.npy',new_result)