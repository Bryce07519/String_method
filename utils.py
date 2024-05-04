''' functions'''
import numpy as np
from numpy import linalg as LA
from basic_function import *
from scipy.interpolate import interp1d




def Init_string(A,B,N=20):
    '''
    construct the initial string between local minimum A and its perturbation B. divided into N+1 images
    '''
    alpha=np.array(range(N+1))/N
    return np.array(list(map(lambda x:(1-x)*A+x*B,alpha)))
def Init_stringAA(A,B,N=20):
    
    temp=np.array([A,B])
    d1=temp[:,0]
    d2=temp[:,1]
    alpha=np.array([0,1])
    d1_re=Init_interpolation(alpha,d1,N).reshape(-1,1)
    d2_re=Init_interpolation(alpha,d2,N).reshape(-1,1)

    return np.concatenate([d1_re,d2_re],axis=1)
def Init_interpolation(x,y,N=20):
    f=interp1d(x, y, kind='linear')
    return f(np.array(range(N+1))/N)

    
def spline_repara(x,y,N):
    '''spline interpolation 1d
    divede into N images'''
    from scipy.interpolate import splev, splrep
    spl = splrep(x, y)
    return splev(np.array(range(N+1))/N, spl)

def Solve_LineanSystem(x,f,eps=1e-5,yita=1e-6,maxiter=10000):
    '''
    solvig Hp=f with Lanczos iteration,sysmetric LQ,f is a vector
    '''

    p0=np.zeros_like(f)
    v0=np.zeros_like(f)
    v1=f
    w_bar0=np.zeros_like(f)
    beta1=1;e0=1;s_=0;s0=0;c_=-1;c0=-1;zeta_=1;zeta0=1
    k=0
    while True:
        k=k+1
        #k=1,2,3,...
        Hv1=(gradient_V(x+eps*v1)-gradient_V(x))/eps
        alpha=np.dot(v1,Hv1)
        u=Hv1-alpha*v1-beta1*v0

        beta2=LA.norm(u)
        v2=u/beta2
        
        eps_1=s_*beta1
        delta_1_bar=-c_*beta1
        delta_1=c0*delta_1_bar+s0*alpha
        
        gamma_1_bar=s0*delta_1_bar-c0*alpha
        gamma_1=np.sqrt(gamma_1_bar*gamma_1_bar+beta2*beta2)
        
     
        c1=gamma_1_bar/gamma_1#c1
        s1=beta2/gamma_1
        
        zeta1=-(eps_1*zeta_+delta_1*zeta0)/gamma_1
        w_bar1=s0*w_bar0-c0*v1
        e1=e0*LA.norm(c0*s1/c1)
        #if converge, break out
        if e1<yita:
            p=p0+zeta1*w_bar1/c1
            break
        #otherwise update paras    
        w1=c1*w_bar1+s1*v2
        p1=p0+zeta1*w1
        
        #update paras
        p0=p1
        v0=v1
        v1=v2

        w_bar0=w_bar1
        beta1=beta2
        e0=e1

        s_=s0
        s0=s1

        c_=c0
        c0=c1

        zeta_=zeta0
        zeta0=zeta1
        
        if k>maxiter:
            print('Iterate more than 10000 times, break out')
            p=None
            break
    return p