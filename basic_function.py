import numpy as np
def V(x):
    Y=x[-1]
    X=x[0]
    return 1+np.sin(X)*np.sin(X)+np.sin(Y)*np.sin(Y)-np.exp(-(X*X+Y*Y))

def gradient_V(x):
    '''gradient of V'''
    y=x[-1]
    x=x[0]
    d1=2*np.sin(x)*np.cos(x)+2*x*np.exp(-(x*x+y*y))
    d2=2*np.sin(y)*np.cos(y)+2*y*np.exp(-(x*x+y*y))
    return np.array([d1,d2])

# def V(x):
#     Y=x[-1]
#     X=x[0]
#     return 1+np.sin(X)*np.sin(X)+np.sin(Y)*np.sin(Y)
# def gradient_V(x):
#     '''gradient of V'''
#     y=x[-1]
#     x=x[0]
#     d1=2*np.sin(x)*np.cos(x)
#     d2=2*np.sin(y)*np.cos(y)
#     return np.array([d1,d2])

def Hessian(x):
    y=x[-1]
    x=x[0]
    h11=2*np.cos(2*x)+2*np.exp(-(x*x+y*y))-4*x*x*np.exp(-(x*x+y*y))
    h12=-4*x*y*np.exp(-(x*x+y*y))
    h22=2*np.cos(2*y)+2*np.exp(-(x*x+y*y))-4*y*y*np.exp(-(x*x+y*y))
    return np.array([[h11,h12],[h12,h22]])
