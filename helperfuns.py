
import numpy as np
from numpy import linalg as LA
from helperfuns import *
import scipy.stats as scs
import matplotlib.pyplot as plt

# find K given s and the normal curve
def Kfun(s, norm = scs.norm(0.2,1)):
    # Finds K, the probability of socially learning
    #input: d real positive numbe or -1, and a normmal curve
    #output K
    K = norm.cdf(s) - norm.cdf(-s)
    return(K)
# find pc given s and the normal curve
def pcfun(s, norm = scs.norm(0.2,1)):
    # Finds pc, the probability of individual learning correctly
    #input: d real positive numbe or -1, and a normmal curve
    #output pc
    pc = 1 - norm.cdf(s)
    return(pc)

# find pw given s and the normal curve
def pwfun(s,norm = scs.norm(0.2,1)):
    # Finds pw, the probability of individual learning incorrectly
    #input: d real positive numbe or -1, and a normmal curve
    #output pw
    pw = norm.cdf(-s)
    return(pw)

# find phi(p_i)
def phi_fun(p_i):
    # Finds $\phi(p_i)$
    return(p_i*(1-p_i)*(2*p_i - 1))

# Find the right side of the recursion shown in Eqs 4a, 4c, or 4e (in the text)
def Wv_fun(p_i,v,r_i,D,K,pc):
    # This is the recursion for u_i, x_i, y_i, times W... It's the left side of the recursion equations
    #v is a stand in for u, x, or y
    soc_learn = K*(p_i + D*phi_fun(p_i))
    to_return = v*(1+r_i)*(soc_learn + pc)
    return(to_return)

# Find the right side of the recursion shown in Eqs 4b, 4d, or 4f (in the text)
def Wvbar_fun(p_1,p_2,v,D,K,pc):
    # This is the left side of the recursion equations for \bar{u}, \bar{x}, and \bar{y}
    pw = 1 - K - pc
    soc_learn = K*(1 - p_1 - p_2 - D*phi_fun(p_1) - D*phi_fun(p_2))
    to_return = v*(soc_learn + pw)
    return(to_return)

# The resource recursion
def ri_fun(r_i,p_i,beta, eta = 1):
    # The recursion for resource $i$
    ri_next = r_i*(1 + eta - beta*p_i)/(1+eta*r_i)
    return(ri_next)

# Iterates the system one time step.
def NextGen(uvec,xvec,yvec,rvec, D, K,pc,beta,deltas = [0, 0, 0], eta=1):
    # Uses the recursion to get the frequencies in the next generation (u_1', u_2', ...)
    #@inputs: curr is current values of u_1,u_2,bu,...
    #         The other inputs are the parameters
    #         deltas are [delta_D, delta_K, delta_pc]
    
    D_y = D + deltas[0]; K_x = K + deltas[1]; pc_x = pc + deltas[2];
    u = sum(uvec)
    x = sum(xvec)
    y = sum(yvec)

    p1 = uvec[0] + xvec[0] + yvec[0]
    p2 = uvec[1] + xvec[1] + yvec[1]
    
    Wu1 = Wv_fun(p1,u,rvec[0],D,K,pc)
    Wu2 = Wv_fun(p2,u,rvec[1],D,K,pc)
    Wbu = Wvbar_fun(p1,p2,u,D,K,pc)
    
    Wx1 = Wv_fun(p1,x,rvec[0],D,K_x,pc_x)
    Wx2 = Wv_fun(p2,x,rvec[1],D,K_x,pc_x)
    Wbx = Wvbar_fun(p1,p2,x,D,K_x,pc_x)
    
    Wy1 = Wv_fun(p1,y,rvec[0],D_y,K,pc)
    Wy2 = Wv_fun(p2,y,rvec[1],D_y,K,pc)
    Wby = Wvbar_fun(p1,p2,y,D_y,K,pc)

    
    W = Wu1 + Wu2 + Wbu + Wx1 + Wx2 + Wbx + Wy1 + Wy2 + Wby
    freqs = (1/W)*np.array([Wu1, Wu2, Wbu, Wx1, Wx2, Wbx, Wy1, Wy2, Wby])
    uvec = freqs[0:3]; xvec = freqs[3:6]; yvec = freqs[6:9]
    
    rvec = [ri_fun(rvec[0], p1, beta,eta), ri_fun(rvec[1],p2,beta, eta)]
    return(uvec, xvec, yvec, rvec,W)

# helper functions for jacobian
def theta_fun(p):
    # Finds theta(p_i)
    # derived from phi(p + delta_p) = phi(p) + delta_p theta(phi)
    # helper function for jacobian
    return(6*p*(1-p) - 1)

# The Jacobian for internal linear stability analysis
def Jac_UR(uvec, rvec, K, D, W, beta, eta =1):
    # This is J^*
    u1 = uvec[0]; u2 = uvec[1]; bu = uvec[2]; r1 = rvec[0]; r2 = rvec[1]
    A1 = W*u1/(1+r1)
    A2 = W*u2/(1+r2)
    def fun(item):
        #checks if the value is really close to 0, and can just be rounded to 0
        if np.allclose(item,0,rtol = 1e-10,atol = 1e-10):
            return(0)
        else:
            return(item)
    u1,u2,bu,r1,r2 = [fun(item) for item in [u1,u2,bu,r1,r2]]    
    U1 = K*(1 + D*theta_fun(u1))
    U2 = K*(1 + D*theta_fun(u2))
    
    
    Rpv = [ -beta*rvec[i]/(1+rvec[i]) for i in [0,1]]; 
    Rv = [1/(1+rvec[i]) for i in [0,1]]
    
    row1 = np.array([U1*(1+r1-u1*r1), -u1*r2*U2,A1*(1-u1),-u1*A2])
    row1 = row1/(W)
    row2 = np.array([-u2*r1*U1, U2*(1+r2-u2*r2),-u2*A2,A2-u2*A2])
    row2 = row2/(W)
   
    row3 =np.array([Rpv[0],0,Rv[0],0])
    row4 = np.array([0,Rpv[1],0,Rv[1]])
    J = np.array([row1,row2,row3,row4])
    return(J)

# Finds C_s for external stability analysis
def Grad_s(uvec, rvec, W, D, s, mu):
    # ln(lambda_x) \approx delta_d * C_d
    # This function finds C_d
    norm = scs.norm(mu,1)
    u1 = uvec[0]
    u2 = uvec[1]
    r1 = rvec[0]
    r2 = rvec[1]
    fs = norm.pdf(s)
    fminuss = norm.pdf(-s)
    # find numerator
    C_s = (1/W)*(-fs + (fs+fminuss)*(r1*u1+r1*D*phi_fun(u1) + r2*u2 + r2*D*phi_fun(u2)))
    return(C_s)

# Finds C_D for external stability analysis
def Grad_D(uvec, rvec, W, K, D):
    #ln(lambda_y) \approx delta_D * C_D
    # this function finds C_D
    u1 = uvec[0]
    u2 = uvec[1]
    r1 = rvec[0]
    r2 = rvec[1]
    CD = (K/W)*(r1*phi_fun(u1)+r2*phi_fun(u2))
    return(CD)



