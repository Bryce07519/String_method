'''
Given the initial string, calculate the saddle point exactly
'''
import numpy as np

def sd_dynamic(phi,tao):
    gradient=gradient_V(phi)
    return gradient-2*np.matmul(tao,gradient)*tao

def find_sd(zfinal):
    length=len(zfinal)
    sd=[]
    for i in range(1,length-1):
        if zfinal[i]>zfinal[i-1] and zfinal[i]>zfinal[i+1]:
            sd.append(i)
    return sd