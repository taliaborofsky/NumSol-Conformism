
import numpy as np
from numpy import linalg as LA
from helperfuns import *
import scipy.stats as scs
import matplotlib.pyplot as plt

def Kfun(d, norm = scs.norm(0.2,1)):
    # Finds K, the probability of socially learning
    #input: d real positive numbe or -1, and a normmal curve
    #output K
    K = norm.cdf(d) - norm.cdf(-d)
    return(K)
def pcfun(d, norm = scs.norm(0.2,1)):
    # Finds pc, the probability of individual learning correctly
    #input: d real positive numbe or -1, and a normmal curve
    #output pc
    pc = 1 - norm.cdf(d)
    return(pc)
def pwfun(d,norm = scs.norm(0.2,1)):
    # Finds pw, the probability of individual learning incorrectly
    #input: d real positive numbe or -1, and a normmal curve
    #output pw
    pw = norm.cdf(-d)
    return(pw)

def phi_fun(p_i):
    # Finds $\phi(p_i)$
    return(p_i*(1-p_i)*(2*p_i - 1))

def Wv_fun(p_i,v,r_i,D,K,pc):
    # This is the recursion for u_i, x_i, y_i, times W... It's the left side of the recursion equations
    #v is a stand in for u, x, or y
    soc_learn = K*(p_i + D*phi_fun(p_i))
    to_return = v*(1+r_i)*(soc_learn + pc)
    return(to_return)
def Wvbar_fun(p_1,p_2,v,D,K,pc):
    # This is the left side of the recursion equations for \bar{u}, \bar{x}, and \bar{y}
    pw = 1 - K - pc
    soc_learn = K*(1 - p_1 - p_2 - D*phi_fun(p_1) - D*phi_fun(p_2))
    to_return = v*(soc_learn + pw)
    return(to_return)

def ri_fun(r_i,p_i,beta, eta = 1):
    # The recursion for resource $i$

    

    ri_next = r_i*(1 + eta - beta*p_i)/(1+eta*r_i)
    return(ri_next)

def NextGen(uvec,xvec,yvec,rvec, D, K,pc,beta,deltas = [0, 0, 0], eta=1):
    # Uses the recursion to get the frequencies in the next generation (u_1', u_2', ...)
    #@inputs: curr is current values of u_1,u_2,bu,...
    #         The other inputs are the parameters
    #         deltas are [delta_D, delta_K, delta_pc]
    
    D_y = D + deltas[0]; K_x = K + deltas[1]; pc_x = pc + deltas[2];
    u = sum(uvec)
    x = sum(xvec)
    y = sum(yvec)
    n = len(uvec)
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

def Ai_jac(uvec,rvec,W,i):
    # helper function for Jacobian.
    # given equilibrium values of u, r, W. i says which u, r to use. i = 0 or 1
    # output: a common constant in jacobian
    if i in [0,1]:
        Ai = W*uvec[i]/(1 + rvec[i])
        return(Ai)
    else:
        print("i out of range")

def Jac_UR(uvec, rvec, K, D, W, beta, eta =1):
    # This is J^*
    u1 = uvec[0]; u2 = uvec[1]; bu = uvec[2]; r1 = rvec[0]; r2 = rvec[1]
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



def FindEquilibrium(uvec,xvec,yvec,rvec,K,pc,D,tsteps,deltas=[0,0,0],beta=1,eta=1):
    # uvec,xvec,yvec,rvec have initial values
    # runs recursions until a stable point is reached
    # deltas are [delta_D, delta_K, delta_pc]
    
    #clause to make it faster in certain cases
    reached_eq = 0
    for t in range(0,tsteps): # could also do While reached_eq= 0, but could lead to infinite loop if no stable point
        ucurr, xcurr, ycurr, rcurr, Wcurr = NextGen(uvec,xvec,yvec,rvec, D, K,pc,deltas, beta, eta)
        u1 = ucurr[0]; u2 = ucurr[1]; r1 = rcurr[0]; r2 = rcurr[1]
        reached_eq = Check_reached_eq(u1,u2,r1,r2,D,K,pc,beta, Wcurr)
        if (reached_eq): # checks that the two vectors are equal
            #rcurr = 1 - beta*ucurr # makes sure r is where it should be
            return(ucurr, xcurr, ycurr, rcurr, Wcurr, t, reached_eq)
        else:
            uvec,xvec,yvec,rvec = ucurr, xcurr, ycurr, rcurr
    
    return(uvec,xvec,yvec,rvec,Wcurr,t,reached_eq) 
def Check_reached_eq(u1,u2,r1,r2,D,K,pc,beta, W):
    if K == 0:
        return(1) # because in this case, I started the system at the calculated equilibrium
    atol = 1e-10
    rtol = 1e-10
    r1eq = 1 - beta*u1
    r2eq = 1 - beta*u2

    # if u1 = u2
    #check if r's are correct and then check Wu1 = rhs of Wu1 eqn for u1 = u2
    if np.allclose([u1,u2,r1,r2], [u2,u1,r1eq,r2eq], atol =atol ,rtol = rtol ):
        shouldEqual0 = u1isu2_eq(u1,r1,K,D,pc,W)
        if np.allclose(shouldEqual0,0,atol=atol, rtol = rtol):
            return(1)
        else:
            return(0)
    elif np.allclose(u1,u2,atol = atol, rtol = rtol): #u1 = u2 but r1 and r2 not correct values 
        return(0)

    # check if at equilibrium for u1 != u2
    
    shouldEqual0 = u1notu2_eq(u1,u2,D,K,pc,beta)
    n = len(shouldEqual0)
    if np.allclose(shouldEqual0,np.zeros(n),atol=1e-10,rtol = 1e-10):
        return(1)
    else:
        return(0)

def u1isu2_eq(u1,r1,K,D,pc,W):
    Wu1 = ( 1 + r1 )*(K*(u1 + D*phi_fun(u1)) + pc)
    return(Wu1 - W*u1)
def u1notu2_eq(u1,u2,D,K,pc,beta): 
    a = D*K*u1**2*u2*(-2*u1*beta - 2*u2*beta+3*beta +4)
    b = -K*u1*u2*(2*D*u2**2*beta -3*D*u2*beta -4*D*u2 + D*beta +6*D - beta)
    c = 2*pc
    shouldEqual0 = a + b + c
    return(shouldEqual0)

def eval_x(uvec, rvec, W, D, dk, dpc):
    # Find the eigenvalue of J_x
    u1 = uvec[0]
    u2 = uvec[1]
    r1 = rvec[0]
    r2 = rvec[1]
    dpw = -dk - dpc
    eval_x = 1 + (1/W)*(dpc + r1*dk*(u1 + D*phi_fun(u1)) + r2*dk*(u2+D*phi_fun(u2)))
    return(eval_x)
  
def eval_y(uvec, rvec, W, K, dD):
    # Find the eigenvalue of J_y
    u1 = uvec[0]
    u2 = uvec[1]
    r1 = rvec[0]
    r2 = rvec[1]

    eval_y = 1 + (1/W)*K*dD*(r1*phi_fun(u1) + r2*phi_fun(u2))
    return(eval_y)

def Grad_d(uvec, rvec, W, D, s, mu):
    # ln(lambda_x) \approx delta_d * C_d
    # This function finds C_d
    norm = scs.norm(mu,1)
    u1 = uvec[0]
    u2 = uvec[1]
    r1 = rvec[0]
    r2 = rvec[1]
    fs = norm.pdf(d)
    fminuss = norm.pdf(-s)
    # find numerator
    C_s = (1/W)*(-fs + (fs+fminuss)*(r1*u1+r1*D*phi_fun(u1) + r2*u2 + r2*D*phi_fun(u2)))
    return(C_s)

def Grad_D(uvec, rvec, W, K, D):
    #ln(lambda_y) \approx delta_D * C_D
    # this function finds C_D
    u1 = uvec[0]
    u2 = uvec[1]
    r1 = rvec[0]
    r2 = rvec[1]
    CD = (K/W)*(r1*phi_fun(u1)+r2*phi_fun(u2))
    return(CD)



