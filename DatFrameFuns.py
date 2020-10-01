import numpy as np
from numpy import linalg as LA
from helperfuns import *
import scipy.stats as scs
import pandas as pd
#for parallelizing:
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.optimize import fsolve

# def param_grid_noInits(Dvals =np.arange(0,1.2,0.2), dvals=np.arange(0,4.5,0.5),
#                        muvals=np.array([-2,-1,-0.5,-0.2,0,0.2]), betavals = np.arange(0,1.25,0.25)):
#     Dmesh, dmesh, mumesh, betamesh = np.meshgrid(Dvals,dvals,muvals,betavals)
#     Dvec,dvec,muvec,betavec = [np.ndarray.flatten(item) for item in [Dmesh, dmesh, mumesh, betamesh]]
#     norms = scs.norm(muvec,1)
#     Kvec = Kfun(dvec,norms)
#     pcvec = pcfun(dvec,norms)
#     Dvec,dvec,muvec,betavec, Kvec, pcvec = [np.repeat(item,10) for item in [Dvec,dvec,muvec,betavec, Kvec, pcvec]] # can later get rid of rows of nan's
#     n = len(Dvec)

#     data = {'mu':muvec,'K': Kvec, 'pc': pcvec,'d':dvec, 'D':Dvec, 'beta': betavec, 'u1eq': np.empty(n).fill(np.nan), 
#             'u2eq':np.empty(n).fill(np.nan), 'bueq':np.empty(n).fill(np.nan)}
#     df = pd.DataFrame(data = data)
def get_param_grid(Dvals =np.arange(-2,1.2,0.2), dvals=np.arange(0.5,4.5,0.5), 
                   muvals=np.array([-2,-1,-0.5,-0.2,0,0.2]), betavals = np.arange(0,1.25,0.25)):
    # gives parameter grids, with u1/u2 initials for the u2 side of the simplex (need to be reflected later)
    u1init = [0.05, 0.3,0.48, 0.08]
    u2init = [0.1,0.6,0.5,0.9]
    which_uinit = [0,1,2,3]
    r1init = [0.05, 0.45, 0.85, 0.1, 0.1,  0.5,  0.9,  0.9] #each point's reflection accross the 45˚ line is included
    r2init = [0.1,  0.5,  0.9,  0.9, 0.05, 0.45, 0.85, 0.1]
    which_rinit = [0,1,2,3,4,5,6,7]

    # make mesh
    Dmesh, dmesh, betamesh, which_uinit_mesh, which_rinit_mesh, mumesh = np.meshgrid(Dvals,dvals,betavals,which_uinit, which_rinit,muvals)
    # flatten
    [Dvec,dvec,betavec,which_uinit_vec, which_rinit_vec, muvec] = [np.ndarray.flatten(item) for item in [Dmesh, dmesh, betamesh, which_uinit_mesh, which_rinit_mesh, mumesh]]
    # get rid of invalid 'rows' where bu is < 0
    u1vec = np.array([u1init[i] for i in which_uinit_vec])
    u2vec = np.array([u2init[i] for i in which_uinit_vec])
    buvec = 1 - u1vec-u2vec
    r1vec = np.array([r1init[i] for i in which_rinit_vec])
    r2vec = np.array([r2init[i] for i in which_rinit_vec])
    whichOK = buvec >=0
    [Dvec,dvec,betavec,u1vec,u2vec, r1vec, r2vec, muvec] = [item[whichOK] for item in [Dvec,dvec,betavec,u1vec,u2vec,r1vec,r2vec, muvec]]


    norms = scs.norm(muvec,1)
    Kvec = Kfun(dvec,norms)
    pcvec = pcfun(dvec,norms)
    pwvec = 1 - Kvec - pcvec
    n = len(Dvec)

    data = {'mu':muvec,'K': Kvec, 'pc': pcvec,'d':dvec, 'D':Dvec, 'beta': betavec, 'u1init': u1vec, 'u2init':u2vec, 
            'buinit':buvec, 'r1init':r1vec, 'r2init':r2vec, 'u1eq': np.zeros(n), 'u2eq': np.zeros(n), 'bueq':np.zeros(n),
            'r1eq':np.zeros(n), 'r2eq':np.zeros(n), 'Weq': np.zeros(n), 'time': np.zeros(n), 'reached_eq': np.zeros(n),
           'URstable': np.zeros(n)}
    df = pd.DataFrame(data = data)
    df = df.sort_values(by=['mu','beta','d','D'])
    df = df.reindex(np.arange(len(df.index)))
    return(df)

# def param_grid_phase_plot(Dvals =np.arange(0,1.2,0.2), dvals=np.arange(0,4.5,0.5),
#                           muvals=np.linspace(-0.2,0.2,3), betavals = np.arange(0,1.25,0.25), tsteps = 10, FineGrid = False):
#     #sets up parameter grid for phase plot diagram
    
#     if (FineGrid):
#         u1init = np.linspace(0.0001,0.99999,20)
#         u2init = np.linspace(0.0002,0.99998,20)
#     else:
#         u1init = np.linspace(0.01,0.98,10)
#         u2init = np.linspace(0.02,0.99,10)
        
    
#     r1init = [0.05,0.45,0.85,0.1,0.1,0.5,0.9,0.9] #each point's reflection accross the 45˚ line is included
#     r2init = [0.1,0.5,0.9,0.9,0.05,0.45,0.85,0.1]
#     which_rinit = [0,1,2,3,4,5,6,7]

#     # make mesh
#     Dmesh, dmesh, betamesh, u1mesh, u2mesh, which_rinit_mesh, mumesh = np.meshgrid(Dvals,dvals,betavals,u1init,u2init, which_rinit,muvals)
#     # flatten
#     [Dvec,dvec,betavec,u1vec,u2vec, which_rinit_vec, muvec] = [np.ndarray.flatten(item) for item in [Dmesh, dmesh, betamesh, u1mesh,
#                                                                                                      u2mesh, which_rinit_mesh, mumesh]]
#     r1vec = np.array([r1init[i] for i in which_rinit_vec])
#     r2vec = np.array([r2init[i] for i in which_rinit_vec])
    
#     # get rid of invalid 'rows' where bu is < 0

#     buvec = 1 - u1vec-u2vec

#     whichOK = (buvec >=0)&(u2vec >= u1vec) # will have to reflect
#     [Dvec,dvec,betavec,u1vec,u2vec, r1vec, r2vec, muvec] = [item[whichOK] for item in [Dvec,dvec,betavec,u1vec,u2vec,
#                                                                                        r1vec,r2vec, muvec]]
#     buvec = 1 - u1vec - u2vec
#     #  learning parameters
#     norms = scs.norm(muvec,1)
#     Kvec = Kfun(dvec,norms)
#     pcvec = pcfun(dvec,norms)
    
#     # oranize into df
#     n = len(Kvec)
#     data = {'mu':muvec,'K': Kvec, 'pc': pcvec,'d':dvec, 'D':Dvec, 'beta': betavec, 'u1init': u1vec, 'u2init':u2vec, 
#             'buinit':buvec, 'r1init':r1vec, 'r2init':r2vec, 'u1_t': np.zeros(n), 'u2_t': np.zeros(n), 'bu_t':np.zeros(n),
#             'r1_t':np.zeros(n), 'r2_t':np.zeros(n), 'W_t': np.zeros(n)}
#     df = pd.DataFrame(data = data)
#     df = df.sort_values(by=['mu','beta','d','D'])
#     df.reset_index(inplace=True, drop=True)
#     df['tsteps'] = 10
#     return(df)

# def df_Xsteps_parallelize(df):
#     cores=mp.cpu_count()
#     df_split = np.array_split(df, cores, axis=0)
#     pool = Pool(cores)
#     df_out = np.vstack(pool.map(fun, df_split))
#     pool.close()
#     pool.join()
#     pool.clear()
#     df_final = pd.DataFrame(df_out, columns = df.columns)
#     return(df_final)


# def df_Xsteps(df):
#     # make df used for phase plot
#     # fill in u1_t, u2_t, bu_t, r1_t, r2_t, W_t
#     # should have already called df = param_grid_phase_plot(FineGrid = FineGrid)

#     tsteps = df.tsteps.values[0] # since they're all the same
#     result = GetXsteps(df, tsteps = tsteps)
#     umat, xmat, ymat, rmat, W = result
#     df.u1_t = umat[0]
#     df.u2_t = umat[1]
#     df.bu_t = umat[2]
#     df.r1_t = rmat[0]
#     df.r2_t = rmat[1]
#     df.W_t = W
    
#     # check if r1 vals at eq
#     df['r1_at_eq'] = np.equal(df.r1_t.values, 1 - df.beta.values*df.u1_t)
#     df['r2_at_eq'] = np.equal(df.r1_t.values, 1 - df.beta.values*df.u2_t)
    
#     # check u1,u2, r1, r2 and parameters satisfy equilibrium equations
#     df['shouldEqual0'] = Check_Eq_Equations(df) # fill in
    
#     return(df)


def Check_Stable_D(row):
# We have a bunch of points with C_D = 0. Check_Stable_D checks if they're actually stable

    
    # get vectors of post perturb values of all the frequencies
    umat,ymat,rmat = Perturb(row)
    umat = np.array(umat); ymat = np.array(ymat); rmat = np.array(rmat)
    # check stability to increase in D
    
    dD =  0.01
    n = len(umat[0]) 
    y_pos_invades = False # default is that it is stable
    if D <= 0.99:
        for i in range(0,n):
            result = NextGen(umat[:,i],[0,0,0],ymat[:,i],rmat[:,i], 
                             row.D, row.K,row.pc,row.beta,
                             deltas = [dD, 0, 0], eta=1)
            yvec = result[2]
            y = sum(yvec)
            if y > 0.01: 
                y_pos_invades = True
                break
    
    dD = -0.01
    y_neg_invades = False # default is that it is stable
    #check stability to decrease in D
    if D >= -1.99: # otherwise it's stable because D can't go down further
        for i in range(0,n):
            result = NextGen(umat[:,i],[0,0,0],ymat[:,i],rmat[:,i], 
                             row.D, row.K,row.pc,row.beta,
                             deltas = [dD, 0, 0], eta=1)
            yvec = result[2]
            y = sum(yvec)
            if y > 0.01: 
                y_neg_invades = True 
                break
    
    row.loc['y_pos_invades'] = y_pos_invades
    row.loc['y_neg_invades'] = y_neg_invades
    return(row)
def Perturb(row):
# After the [u1,u2,bu,r1,r2] eq is perturbed with the addition of the a or b allele, get new frequencies    

    # use dyvec here but dxvec could work too
    uvec = [row.u1eq, row.u2eq, row.bueq]
    rvec = [row.r1eq, row.r2eq]
    # get new post-perturb vectors
    
    # no need to check if valid
    dy = 0.01
    dy1vec = np.array([0.05, 0.3,0.48, 0.08])*dy
    dy2vec = np.array([0.1,0.6,0.5,0.9])*dy
    dbyvec = dy - dy1vec - dy2vec
    which_y = [0,1,2,3]
    
    #dr... prune r + dr values that are invalid
    
    dr1=0 #the default is not changing r1
    if rvec[0]>0:
        dr1 = np.array([-0.01,0.01])
        check_r1 = (rvec[0] + dr1 > 0)&(rvec[0] + dr1 < 1)
        dr1 = dr1[check_r1]
    dr2 = 0
    if rvec[1]>0:
        dr2 = np.array([-0.01,0.01])
        check_r2 = (rvec[1] + dr2 > 0) & (rvec[1] + dr2 < 1)
        dr2 = dr2[check_r2]
    
    # du perturbations... look similar to x perturbs, but opposite direction
    du = -dy
    du1vec = np.array([0.015, 0.29,0.49, 0.07])*du
    du2vec = np.array([0.11,0.49,0.51,0.89])*du
    dbuvec = du - du1vec - du2vec
    which_u = [0,1,2,3]
    for i in which_u:
        duvec = [du1vec[i],du2vec[i],dbuvec[i]]
        [du1vec[i],du2vec[i],dbuvec[i]] = Perturb_EdgeCase(uvec,duvec)
    
    # now find all the combinations
    which_y_mesh, which_u_mesh, DR1, DR2 = np.meshgrid(which_y, which_u, dr1, dr2)
    [which_y, which_u, dr1, dr2] = [np.ndarray.flatten(item) for 
                                    item in [which_y_mesh, which_u_mesh, DR1, DR2]]
    y1 = dy1vec[which_y]; y2 = dy2vec[which_y]; by = dbyvec[which_y]
    u1 = uvec[0] + du1vec[which_u]
    u2 = uvec[1] + du2vec[which_u]
    bu = uvec[2] + dbuvec[which_u]
    r1 = rvec[0] + dr1
    r2 = rvec[1] + dr2
    
    
    return([u1,u2,bu],[y1,y2,by],[r1,r2])
def Perturb_EdgeCase(uvec,duvec):
    #recursively checks for edge cases and adjusts duvec if needed
    # CHECK it works
    
    # make sure using numpy arrays
    du = sum(duvec)
    uvec = np.array(uvec); duvec = np.array(duvec);
    # find indices of edge cases
    edge_bool = uvec + duvec <= 0
    
    n = sum(edge_bool)
    if n>0:
        duvec[edge_bool] = -uvec[edge_bool] +0.00001 # so not at exactly 0
        du_remain = du - sum(duvec)
        duvec[~edge_bool] = duvec[~edge_bool] + (1/np.float(3-n))*du_remain
        
        # make sure that we didn't cause a different frequency to be negative:
        return(Perturb_EdgeCase(uvec,duvec))

    else:
        return(duvec)
    

          
def GetXsteps(param_grid, tsteps):
        
    u1init = param_grid.u1init.values
    u2init = param_grid.u2init.values
    buinit = param_grid.buinit.values
    r1init = param_grid.r1init.values
    r2init = param_grid.r2init.values
    Kvec = param_grid.K.values
    Dvec = param_grid.D.values
    pcvec = param_grid.pc.values
    betavec = param_grid.beta.values
    
    n = len(u2init)
    uvec = [u1init, u2init, buinit]
    xvec = [np.zeros(n),np.zeros(n), np.zeros(n)]
    yvec = [np.zeros(n),np.zeros(n), np.zeros(n)]
    rvec = [r1init, r2init]
    
    for i in range(0,tsteps):
        result = NextGen(uvec,xvec,yvec,rvec, Dvec, Kvec,pcvec ,beta = betavec)
        uvec, xvec, yvec, rvec, W = result
    return(result)

def GetEq_DF(row, retry = 0, tsteps = 50000, dxy = 0):
    # if if_perturb = True, should only be given dataframe which is unstable in direction being perturbed
    #assign vals to uv, rv, etc
    #print(row.u1init)
    if dxy > 0:
        tsteps = 50000
        deltas = [row.dD, row.dk, row.dpc]
        du1 = 0.3*dxy; du2 = 0.41*dxy; dbu = 0.29*dxy
        
        uv = np.array([row.u1eq - du1, row.u2eq - du2, row.bueq - dbu])
        xvec = [row.dx1, row.dx2, row.dbx]
        yvec = [row.dy1, row.dy2, row.dby]
        rv = [row.r1eq, row.r2eq]
        Dval = row.D; Kval = row.K; pcval = row.pc; betaval = row.beta
        ans = FindEquilibrium(uv,xvec,yvec,rv,Kval,pcval,Dval,tsteps, deltas,betaval)
        uvec,xvec,yvec,rvec,Wcurr,t,reached_eq = ans
        [row.u1eq2, row.u2eq2, row.bueq2] = uvec
        [row.x1eq, row.x2eq, row.bxeq] = xvec
        [row.y1eq, row.y2eq, row.byeq] = yvec
        [row.r1eq2, row.r2eq2] = rvec
        row.Weq2 = Wcurr
        row.time2 = t
        row.reached_eq2 = reached_eq
        return(row)
    else:
        if retry ==0:
            uv = [row.u1init, row.u2init, row.buinit]
            rv = [row.r1init, row.r2init]
            tsteps_0 = 0
        else:
            uv = [row.u1eq, row.u2eq, row.bueq]
            rv = [row.r1eq, row.r2eq]
            tsteps_0 = row.time
        xvec = yvec = deltas = [0,0,0]
        
        Dval = row.D; Kval = row.K; pcval = row.pc; betaval = row.beta
        # num steps to iterate over to find equilibrium
        ans = FindEquilibrium(uv,xvec,yvec,rv,Kval,pcval,Dval,tsteps, deltas,betaval)
        uvec_eq, xvec, yvec, rvec_eq, Weq, t, reached_eq = ans
        [row.u1eq, row.u2eq, row.bueq] = uvec_eq
        [row.r1eq, row.r2eq] = rvec_eq
        row.Weq = Weq 
        row.time = t + tsteps_0 
        row.reached_eq = reached_eq
        return(row)
    
def JstarStable(row):
    # Checks Jstar stability... 1 if stable, 0 if not, -1 if leading eval is 1 (or -1)
    # adds in the absolute value of leading eigenvalue
    
    uvec = [row.u1eq, row.u2eq, row.bueq]
    rvec = [row.r1eq, row.r2eq]
    K = row.K
    D = row.D
    W = row.Weq
    beta = row.beta
    Jstar = Jac_UR(uvec, rvec, K, D, W, beta)
    evals = np.linalg.eigvals(Jstar).real
    abs_evals = np.abs(evals)
    maxval = max(abs_evals)
    leadval = evals[np.where(np.abs(evals) == maxval)[0][0]]
    row.lambdastar = leadval
    if maxval<1:
        row.URstable = 1.0

    elif maxval ==1:
        row.URstable = -1.0
    else:
        row.URstable = 0.0
    return(row)    
def get_1side_df(df):
    #make sure indexed correctly
    tsteps = 50000
    result = GetXsteps(df, tsteps = 50000)
    umat, xmat, ymat, rmat, W = result
    
    #check_eq 
    
    
    df.u1eq = umat[0]
    df.u2eq = umat[1]
    df.bueq = umat[2]
    df.r1eq = rmat[0]
    df.r2eq = rmat[1]
    df.Weq = W
    df.time = 50000
    return(df)
    
    
def reflect_df(df):
    u1init = df.u1init
    u2init = df.u2init
    r1init = df.r1init
    r2init = df.r2init
    u1eq = df.u1eq
    u2eq = df.u2eq
    r1eq = df.r1eq
    r2eq = df.r2eq
    df2 = df.copy()
    df2.u2init = u1init
    df2.u1init = u2init
    df2.r1init = r2init
    df2.r2init = r1init
    df2.u1eq = u2eq
    df2.u2eq = u1eq
    df2.r1eq = r2eq
    df2.r2eq = r1eq
    
    df = df.append(df2)
    return(df)
    #add in 500 at a time, updating the file
    # print where you're at each time
    #after loop, check how many rows remain

def reflect_df_Xsteps(df):
    u1init = df.u1init
    u2init = df.u2init
    r1init = df.r1init
    r2init = df.r2init
    u1_t = df.u1_t
    u2_t = df.u2_t
    r1_t = df.r1_t
    r2_t = df.r2_t
    df2 = df.copy()
    df2.u2init = u1init
    df2.u1init = u2init
    df2.r1init = r2init
    df2.r2init = r1init
    df2.u1_t = u2_t
    df2.u2_t = u1_t
    df2.r1_t = r2_t
    df2.r2_t = r1_t
    
    df = df.append(df2)
    return(df)

def Check_Eq_Equations(df):
    # checks whether u1, u2, r1, r2 at equilibrium after 10 steps.

    u1_t = df.u1_t.values
    u2_t = df.u2_t.values
    u1isu2 = np.equal(u1_t,u2_t)
    r1_t = df.r1_t.values
    r2_t = df.r2_t.values
    W = df.W_t.values
    D = df.D.values
    K = df.K.values
    pc = df.pc.values
    beta = df.beta.values
    
    # check values at t where u1 = u2
    shouldEqual0 = np.zeros(len(u1_t))
    inputs = [item[u1isu2] for item in [u1_t, r1_t, K, D, pc, W]]
    shouldEqual0[u1isu2] = u1isu2_eq(inputs[0],inputs[1],inputs[2],inputs[3], inputs[4], inputs[5])
    
    
    # check values at t where u1 != u2
    inputs = [item[~u1isu2] for item in [u1_t, u2_t,  D, K, pc, W]]
    shouldEqual0[~u1isu2] = u1notu2_eq(*inputs)
    
    return(shouldEqual0)

def Check_Eq_MainDF(df):

    u1 = df.u1eq.values
    u2 = df.u2eq.values
    u1isu2 = np.isclose(u1,u2, atol = 1e-10, rtol = 1e-10)
    r1 = df.r1eq.values
    r2 = df.r2eq.values
    W = df.Weq.values
    D = df.D.values
    K = df.K.values
    pc = df.pc.values
    beta = df.beta.values
    
    # check values at t where u1 = u2
    
     #u1isu2_eq(u1,r1,K,D,pc,W):

    shouldEqual0 = np.zeros(len(u1))
    inputs = [item[u1isu2] for item in [u1, r1, K, D, pc, W]]
    shouldEqual0[u1isu2] = u1isu2_eq(*inputs)
    
    # u1notu2_eq(u1,u2,D,K,pc,beta): 
    
    # check values at t where u1 != u2
    inputs = [item[~u1isu2] for item in [u1, u2, D,K, pc, beta]]
    shouldEqual0[~u1isu2] = u1notu2_eq(*inputs)
    shouldEqual0 = np.isclose(shouldEqual0,np.zeros(len(shouldEqual0)),atol=1e-10,rtol=1e-10)
    
    # next check r vals are right
    
    r1correct = 1 - beta*u1
    r2correct = 1 - beta*u2
    
    isr1correct = np.isclose(r1,r1correct,atol=1e-10,rtol=1e-10)
    isr2correct = np.isclose(r2,r2correct,atol=1e-10,rtol=1e-10)
    
    rvalsCorrect = isr1correct & isr2correct
    
    AtEquilibrium = shouldEqual0&rvalsCorrect
    
    return(AtEquilibrium)

def retry_findeq(df):
    # tries another 50,000 iterations on rows that didn't reach equilibrium in the dataframe
    # TO-DO: check
    
    # assuming df has only failed entries and has already been split up
    df.apply(lambda row: GetEq_DF(row, retry=1, tsteps = 200000), axis = 1)
    
#     mask = (df.reached_eq==0)
#     df_fail = df[mask]
#     num = 50
#     n = len(df_fail); 
#     num_its = int(n/num)
#     remainder = n%num; 
#     print('retrying rows that didnt reach equilibrium')

#     starti = 0; endi = num
#     df4 = df_fail.copy().iloc[starti:endi,:]
#     df4 = df4.apply(lambda row: GetEq_DF(row, retry=1, tsteps = 200000), axis = 1)
#     print('finished first %d' %num)
#     for i in range(1,num_its):
#         starti = endi + 1
#         endi = endi + num
#         df5 = df_fail.copy().iloc[starti:endi,:]
#         df5 = df5.apply(lambda row: GetEq_DF(row, retry=1), axis = 1)    
#         print('Finished filling in entries from row %d to %d ' %(starti,endi))
#         df4 = df4.append(df5)
#     if remainder>0:
#         starti = endi + 1
#         endi = n -1
#         df5 = df_fail.copy().loc[starti:endi,:]
#         df5 = df5.apply(lambda row: GetEq_DF(row, retry=1), axis = 1)
#         df4 = df4.append(df5)
        
#     df[mask] = df4
    return(df)

def get_UniqueEquilibria(df,if_save=False):
    df_eq = df.round(6)[(df.reached_eq==1)].groupby(['K','pc','d','mu','D','beta','u1eq','u2eq','bueq',
                                                     'r1eq','r2eq','Weq','URstable'], as_index = False)
    df_eq = df_eq['u2init'].count()
    df_eq.rename(columns={'u2init':'NumInitials'}, inplace=True)
    # df_eq.reset_index(inplace=True, drop=True)
    df_eq = df_eq.apply(lambda row: JstarStable(row), axis = 1)
    df_eq = get_gradients(df_eq)
    if if_save:
        df_eq.to_csv('UniqueEquilibria.csv', index = False)
    return(df_eq)
def get_gradients(df):
    df_use = df.copy()
    u1vec = df_use.u1eq
    u2vec = df_use.u2eq
    r1vec = df_use.r1eq
    r2vec = df_use.r2eq
    Wvec = df_use.Weq
    Kvec = df_use.K
    Dvec = df_use.D
    dvec = df_use.d
    muvec = df_use.mu
    Cdvec = [Grad_d([u1,u2], [r1,r2], W, D, d, mu) for u1,
             u2,r1,r2,W,D,d,mu in zip(u1vec,u2vec,r1vec,r2vec,Wvec,Dvec,dvec,muvec)]
    CDvec = [Grad_D([u1,u2], [r1,r2], W, K,D) for u1,u2,r1,r2,W,K,D in zip(u1vec,u2vec,r1vec,r2vec,Wvec,Kvec,Dvec)]
    df_use['C_d'] = Cdvec
    df_use['C_D'] = CDvec
    return(df_use)

def get_lambdas(row, dmat, mumu, dk_pos, dk_neg, dpc_pos, dpc_neg):
    d = row.d
    mu = row.mu
    ind_row = np.where(mumu==mu)[0][0]
    ind_col = np.where(dmat==d)[1][0]
    dk_pos = dk_pos[ind_row,ind_col]
    dk_neg = dk_neg[ind_row,ind_col]
    dpc_pos = dpc_pos[ind_row,ind_col]
    dpc_neg = dpc_neg[ind_row,ind_col]
    
    row.lambdax_pos = eval_x([row.u1eq, row.u2eq], [row.r1eq, row.r2eq], row.Weq, row.D, dk_pos, dpc_pos).round(4)
    row.lambdax_neg = eval_x([row.u1eq, row.u2eq], [row.r1eq, row.r2eq], row.Weq, row.D, dk_neg, dpc_neg).round(4)
    row.lambday_pos = eval_y([row.u1eq, row.u2eq], [row.r1eq, row.r2eq], row.Weq, row.K, row.D, row.dD_pos).round(4)
    row.lambday_neg = eval_y([row.u1eq, row.u2eq], [row.r1eq, row.r2eq], row.Weq, row.K, row.D, row.dD_neg).round(4)
    return(row)

def fill_in_noneq(df):
    df_fail = df[df.reached_eq==0]
    print('%d rows did not reach equilibrium. Running these rows more' %(len(df_fail)))
    df_fail_try2 = retry_findeq(df_fail)
    df = df[df.reached_eq==1].append(df_fail_try2)
    return(df)

def fsolve_failed_eq(df_fail):
    # take rows that failed to reach equilibrium, use their final states and plug in as initials to fsolve
    def EqSystem(freqs, params):
        [u1,u2,bu,r1,r2, W] = freqs
        K,pc,D,beta = params
        uvec = [u1,u2,bu]; xvec = [0,0,0]; yvec = [0,0,0]; rvec = [r1,r2]
        uvec, xvec, yvec, rvec,W = NextGen(uvec,xvec,yvec,rvec, D, K,pc,beta = beta)
        return_vec = np.array(freqs) - np.array([*uvec,*rvec,W])
        # return_vec of [Wu1 - Wu1func, ... r1 - r1fun, r2 - r2fun]
        return(return_vec)
    def fsolve_rows(row):
        freqs = [row.u1eq, row.u2eq, row.bueq, row.r1eq, row.r2eq, row.Weq]
        params = [row.K, row.pc, row.D, row.beta]
        res, infodict, ier,mesg = fsolve(EqSystem, freqs, args=params, full_output=True, xtol = 1e-10)
        row.u1eq, row.u2eq, row.bueq, row.r1eq, row.r2eq, row.Weq = res
        row.reached_eq = ier
        row.time = -1
        return(row)
    new_eq = df_fail.apply(lambda row: fsolve_rows(row), axis = 1)
    # now check these eq... FILL IN
    return(new_eq)

def IterateCheck_UR_Stable(row):
    # takes row from df of unique eq
    # take equilibrium value
    
    new_u_vals_mat = np.array(uvec).reshape(n,3) - np.transpose(du1vec, du2vec, dbuvec)
    
    # 
    newdf = pd.dataframe(data = new_u_vals_mat, columns = ['u1','u2','bu'])
    # try different perturbations of magnitude 0.01
    # results in equilibrium more than 0.01 away from original, then unstable
    # I may not need to iterate to equilibrium... just iterate a couple steps
    
def Perturb_URdir(row):
    K = row.K; pc = row.pc; D = row.D; beta = row.beta
    du12 = 0.01
    du1 = np.array([0.01, 0.5,0.9])*0.01; du1 = np.concatenate((du1,-du1))
    du2 = np.array([0.02, 0.51,0.92])*0.01; du2 = np.concatenate((du2,-du2))
    dr1 = np.array([0.01, 0.4,0.8])*0.01; dr1 = np.concatenate((dr1,-dr1))
    dr2 = np.array([0.01, 0.45,0.85])*0.01; dr2 = np.concatenate((dr2,-dr2))
    dU1, dU2,dR1,dR2 = np.meshgrid(du1,du2,dr1,dr2)
    du1vec, du2vec, dr1vec, dr2vec = [np.ndarray.flatten(dU1), np.ndarray.flatten(dU2),
                                     np.ndarray.flatten(dR1), np.ndarray.flatten(dR2)]
    dbuvec = -du1vec - du2vec
    mat_new = np.array([row.u1eq, row.u2eq, row.bueq, row.r1eq, row.r2eq]) + np.transpose([du1vec, du2vec, 
                                                                              dbuvec, dr1vec, 
                                                                              dr2vec])
    
    df = pd.DataFrame(data = mat_new, columns = ['u1','u2','bu','r1','r2'])
    df['u'] = mat_new[:,0]+mat_new[:,1]+mat_new[:,2]
    mat = df.values
    mask = [np.sum((row>=0)&(row<=1))==6 for row in mat]
    mask = mask & (df.u == 1)
    df = df[mask]
    df['W'] = Wv_fun(df.u1,1,df.r1,D,K,pc) + Wv_fun(df.u2,1,df.r2,D,K,pc)+Wvbar_fun(df.u1,df.u2,1,D,K,pc)
    df['K'] = K; df['pc'] = pc; df['D']=D; df['beta']=beta
    df['u1_new'] = 0; df['u2_new'] = 0; df['bu_new'] = 0; df['r1_new'] = 0; df['r2_new'] = 0; 
    def EqSystem(freqs, params):
        [u1,u2,bu,r1,r2, W] = freqs
        K,pc,D,beta = params
        uvec = [u1,u2,bu]; xvec = [0,0,0]; yvec = [0,0,0]; rvec = [r1,r2]
        uvec, xvec, yvec, rvec,W = NextGen(uvec,xvec,yvec,rvec, D, K,pc,beta = beta)
        return_vec = [*uvec, *rvec,W] - freqs
        # return_vec of [Wu1 - Wu1func, ... r1 - r1fun, r2 - r2fun]
        return(return_vec)
    def fsolve_rows(row):
        freqs = [row.u1, row.u2, row.bu, row.r1, row.r2, row.W]
        params = [row.K, row.pc, row.D, row.beta]
        res, infodict, ier,mesg = fsolve(EqSystem, freqs, args=params, full_output=True, xtol = 1e-10)
        row.u1_new, row.u2_new, row.bu_new, row.r1_new, row.r2_new, W = res
        return(row)
    
    df  = df.apply(lambda row:fsolve_rows(row), axis = 1)
    change = df.loc[['u1_new','u2_new', 'bu_new', 'r1_new', 'r2_new']] - [row.u1eq, row.u2eq, row.bueq, row.r1eq, row.r2eq]
    if_changed = change>0.01
    if sum(sum(if_changed))>0:
        row.URstable_checked = 0
    else:
        row.URstable_checked = 1
    return(row)
def Equilibrium_post_perturb(row):

    #fill in choosing different starting points
    
    def EqSystem(freqs, params, dx = 0.01):
        [u1,u2,bu,x1,x2,bx,y1,y2,by,r1,r2, W] = freqs
        K,pc,D,dD, dk, dpc, beta = params
        uvec = [u1,u2,bu]; xvec = [0,0,0]; yvec = [0,0,0]; rvec = [r1,r2]
        uvec, xvec, yvec, rvec,W = NextGen(uvec,xvec,yvec,rvec, D, K,pc,deltas = [dD, dk, dpc], beta = beta)
        return_vec = np.array(freqs) - np.array([*uvec,*xvec, *yvec, *rvec,W])
        # return_vec of [Wu1 - Wu1func, ... r1 - r1fun, r2 - r2fun]
        return(return_vec)
    def fsolve_rows(row, dx = 0.01):
        freqs = [row.u1eq, row.u2eq, row.bueq, row.r1eq, row.r2eq, row.Weq]
        params = [row.K, row.pc, row.D, row.dD, row.dk, row.dpc, row.beta]
        res, infodict, ier,mesg = fsolve(EqSystem, freqs, args=params, full_output=True, xtol = 1e-10)
        result = res
        reached_eq = ier
        to_return = np.concatenate((result, reached_eq), axis = 1)
        return(to_return)
    return(df_fail.apply(lambda row:fsolve_rows(row), axis = 1))