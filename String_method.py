import numpy as np
from numpy import linalg as LA
from scipy.interpolate import interp1d

def Init_stringA(A,B,N=20):
    '''
    Divide Initial string into N+1 images
    A:start point
    B:end point
    N:number of images
    '''
    def Init_interpolation(x,y,N=20):
        f=interp1d(x, y, kind='linear')
        return f(np.array(range(N+1))/N)

    temp=np.array([A,B])
    d1=temp[:,0]
    d2=temp[:,1]
    alpha=np.array([0,1])
    d1_re=Init_interpolation(alpha,d1,N).reshape(-1,1)
    d2_re=Init_interpolation(alpha,d2,N).reshape(-1,1)
    return np.concatenate([d1_re,d2_re],axis=1)
def V(x):
    #V(x,y)=(x^2-1)^2+y^2
    Y=x[-1]
    X=x[0]
    return 1+np.sin(X)*np.sin(X)+np.sin(Y)*np.sin(Y)-np.exp(-(X*X+Y*Y))
#     return (X*X-1)*(X*X-1)+Y*Y

def gradient_V(x):
    '''gradient of V'''
    y=x[-1]
    x=x[0]
#     d1=4*X*X*X-4*X
#     d2=2*Y
    d1=2*np.sin(x)*np.cos(x)+2*x*np.exp(-(x*x+y*y))
    d2=2*np.sin(y)*np.cos(y)+2*y*np.exp(-(x*x+y*y))
    return np.array([d1,d2])

# def V(x):
#     #V(x,y)=(1-x^2-y^2)^2+y^2/(x^2+y^2)
#     Y=x[-1]
#     X=x[0]
#     return (1-X*X-Y*Y)*(1-X*X-Y*Y)+Y*Y/(X*X+Y*Y)
# def gradient_V(x):
#     '''gradient of V'''
#     Y=x[-1]
#     X=x[0]
#     d1=-4*X*(1-X*X-Y*Y)-Y*Y*2*X/((X*X+Y*Y)*(X*X+Y*Y))
#     d2=-4*Y*(1-X*X-Y*Y)+Y*2*X*X/((X*X+Y*Y)*(X*X+Y*Y))
#     return np.array([d1,d2])

def rungeKutta(phi,dt,f):
    '''
    solve autonomous ODE phi_t=f(phi),eq(6) in the paper with forth order
    phi: ith image of string, to update from nth iteration to (n+1)th iteration
    dt: time step
    f: a function
    return only the rungekutta term, i.e: phi_n+1 -phi_n
    '''
    k1=dt*f(phi)
    k2=dt*f(phi+1/2*k1)
    k3=dt*f(phi+1/2*k2)
    k4=dt*f(phi+k3)
    rg=-(k1/6+k2/3+k3/3+k4/6)#f=gradient V, there should minus
    return rg

def euler(phi,dt,gradient):
    '''
    explicit euler, gradient is \Delta V(phi)
    '''
    return phi-dt*gradient

def Repara(Phi,N=20):
    '''Phi is the string discretized by N images, each image is M-dimension'''
    def spline_repara(x,y,N=20):
        '''spline interpolation 1d
        divede into N images'''
        length=len(x)
        f=interp1d(x, y, kind='cubic')
        return f(np.array(range(N+1))/N)

    length=len(Phi)
    #alpha^star
    s=[]
    s.append(0)#s_0=0
    for i in range(length-1):
        s.append(s[i]+LA.norm(Phi[i+1,:]-Phi[i,:]))
    alpha=np.array(s)/s[-1]
    #cubic spline interpolation for every dimension eg: (alpha,phi_1) and (alpha,phi_2)(if phi is two dimensional
    d_list=[]
    for i in range(Phi.shape[-1]):
        d=Phi[:,i]
        d_list.append(spline_repara(alpha,d,N).reshape(-1,1))#cubic spline interpolatation on every dimension
    return np.concatenate(d_list,axis=1)
