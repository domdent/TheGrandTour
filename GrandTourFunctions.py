import numpy as np


def getAlpha(d):
    """
    NEEDS IMPLEMENTATION
    Should produce 1xd(d-1)/2 array of position in grand tour.

    """
    p = d*(d-1)/2
    alpha = np.zeros(p) #alpha(t) parameters defining grand tour in G2,d

    for i in range(0,p):
        alpha[i] = np.exp(i) % 1

    alpha = 0.0001**alpha
    return alpha


def getAngles(alpha,d):
    """""
    Inputs: 
    alpha = 1xd(d-1)/2 array defining position on grand tour
    d = dimensions of data
    Outputs a dxd array of angles required for the transformation
    """
    theta = np.zeros((d,d));
    i = 0;
    k = 0;
    
    while i < d-1:
        j = i + 1;
        
        while j < d:
            theta[i][j] = alpha[k];
            j += 1;
            k += 1;
    
        i+= 1;
        
    return theta;


def RotationMatrix(i, j, d, theta):
    """
    Inputs:
    i = first indicie of rotating plane
    j = second indicie of rotating plane
    d = dimensions of data
    theta = dxd array of angle of rotation of rotating plane

    Outputs a rotating matrix to rotate plane of ixj plane by theta_ij
    """
    R = np.identity(d)
    R[i,i] = np.cos(theta)
    R[i,j] = -1*np.sin(theta)
    R[j,i] = np.sin(theta)
    R[j,j] = np.cos(theta)
    return R


def BetaFn(d, theta):
    """
    Inputs:
    d = dimensions of data
    theta = dxd array of angle of rotation ixj plane

    Outputs the full matrix transformation for all rotations
    """
    beta = RotationMatrix(1, 2, d, theta[1,2])
    i = 1
    j = 2
    for i in range(d):
        for j in range(d):
            if j <= i:
                continue
            if i==1 and j==2:
                continue
            beta = np.matmul(beta, RotationMatrix(i, j, d, theta[i,j]))
    return beta


def GrandTour(data, nsteps):
    """
    Inputs:
    data = array of data points, dimensions x npoints
    Outputs a 3D array number of points x t x dimensions, where t
    the time step at that point in the tour
    """

    d = np.size(data)[0] #dimensions of data
    nPoints = np.size(data)[1] #number of data points
    tData = np.zeros((nsteps,d,nPoints)) #initialise 3d matrix to store stransforemd data at each timestep
    
    alpha = getAlpha(d)
    
    for t in range(0, nsteps):

        alpha = t**alpha
        theta = getAngles(alpha, d)
        B = BetaFn(d, theta)
        a = np.matmul(B, data)
        tData[t,:,:] = a

    return tData



