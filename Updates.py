'''including three steps to update the string, a sequnce of images'''
import numpy as np
from numpy import linalg as LA
from utils import *
from basic_function import *
def Evolve_String(Phi,dt,niu=2,eps=1e-10):
    '''
    Phi:a array,the sequence of images {phi_0,...,phi_N}(N+1 images)
    dt:timestep   
    '''
    stable=False#default the start point is not strictly local minimum
    Phi_new=np.zeros_like(Phi)
    if stable:
        Phi_new[0,:]=Phi[0,:]#the start point is fixed phi_0=a， if the start point is stable 
    else:#if not stable,evove the start point with the gradient flow
        Phi_new[0,:]=Phi[0,:]-dt*gradient_V(Phi[0,:])
    #after gradient descent, to determine whether the start point is stable or not
    stable=V(Phi[0,:])-V(Phi_new[0,:])<eps

    N=len(Phi)
    Normal=np.zeros([N-1,2])
    for i in range(N-1):
        phi=Phi[i+1,:]
        difference=phi-Phi[i,:]
        tao=difference/LA.norm(difference)
        normal_force=gradient_V(phi)-np.dot(gradient_V(phi),tao)*tao
        Normal[i,:]=normal_force #calculate equation 16 to determine whether out or not
        #gradient_V should be the Delta V, which is a function calculating the gradient of V
        Phi_new[i+1,:]=Phi[i,:]-dt*normal_force#iteration scheme
    #for last term
    phi=Phi[-1,:]
    difference=phi-Phi[-2,:]
    tao=difference/LA.norm(difference)
    normal_force=gradient_V(phi)-np.dot(gradient_V(phi),tao)*tao
    Phi_new[-1,:]=phi+dt*(-gradient_V(phi)+niu*np.dot(gradient_V(phi),tao)*tao)

    Normal[-1,:]=gradient_V(phi)

    return Phi_new,Normal,stable


def Trunc(Phi,stable):
    '''
    truncate the sequance to ganrantee energy monotonically increasing

    if Phi is monotonically increasing,skip this step
    oterhwise~
    '''
    Phi_trunc=[]
    a=list(map(V,Phi))
    flag=all([a[i] < a[i+1] for i in range(len(a)-1)])
    if flag:
        # print(flag)
        return Phi
    #if not  monotonically increasing, truncte the sequence  
    else:
        if stable:#if the start point is strictly local minimum
            print('The the start point is strictly local minimum')
            print('------begin truncating the images-------')
            # Phi_trunc.append(Phi[0,:])
            for i in range(len(Phi)-1):
                if not(V(Phi[i+1,:])<V(Phi[i,:])):
                    # print(i)
                    #V is the potential function
                    Phi_trunc.append(Phi[i,:])
                else:
                    break
            return np.array(Phi_trunc)
        else:#if the start point is strictly local minimum, ignore it and compare the energy from the second point
            Phi_trunc.append(Phi[0,:])#append the start point
            print('The the start point is not strictly local minimum')
            print('------begin truncating the images-------')
            # Phi_trunc.append(Phi[0,:])
            for i in range(len(Phi)-2):
                if not(V(Phi[i+2,:])<V(Phi[i+1,:])):
                    # print(i)
                    #V is the energy function
                    Phi_trunc.append(Phi[i+1,:])
                else:
                    break
            return np.array(Phi_trunc)

def Repara(Phi,N=20):
    #Phi is the truncated images
    length=len(Phi)
    s=[]
    s.append(0)
    for i in range(length-1):
        s.append(s[i]+LA.norm(Phi[i+1,:]-Phi[i,:]))
    alpha=np.array(s)/s[-1]
    #spline interpolation twice for (alpha,phi_1) and (alpha,phi_2)(phi is two dimensional
    temp=np.array(Phi)
    d1=temp[:,0]
    d2=temp[:,1]
    d1_re=spline_repara(alpha,d1,N).reshape(-1,1)
    d2_re=spline_repara(alpha,d2,N).reshape(-1,1)
    
    return np.concatenate([d1_re,d2_re],axis=1)