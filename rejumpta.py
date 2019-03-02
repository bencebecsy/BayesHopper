################################################################################
#
#(PT)^2AMCMC -- Parallel Tempering Pulsar Timing Array MCMC
#
#Bence Bécsy (bencebecsy@montana.edu) -- 2019
################################################################################

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################

def run_ptmcmc(N, T_max, n_chain, pta, regular_weight=3, PT_swap_weight=1,
               Fe_proposal_weight=0, draw_from_prior_weight=0, de_weight=0):
    #getting the number of dimensions
    ndim = len(pta.params)
    
    #fisher updating every n_fish_update step
    n_fish_update = 200
    #print out status every n_status_update step
    n_status_update = 10
    #add current sample to de history file every n_de_history step
    n_de_history = 10

    #array to hold Differential Evolution history
    history_size = 1000    
    de_history = np.zeros((n_chain, history_size, ndim))
    #start DE after de_start_iter iterations
    de_start_iter = 100

    #setting up temperature ladder (geometric spacing)
    c = T_max**(1.0/(n_chain-1))
    Ts = c**np.arange(n_chain)
    print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\
 Temperature ladder is:\n".format(n_chain,c),Ts)

    #setting up array for the fisher eigenvalues
    eig = np.ones((n_chain, ndim, ndim))*0.1

    #setting up array for the samples and filling first sample with random draw
    samples = np.zeros((n_chain, N, ndim))
    for j in range(n_chain):
        samples[j,0,:] = np.hstack(p.sample() for p in pta.params)

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros(n_chain+1)
    a_no=np.zeros(n_chain+1)
    swap_record=[]

    #set up probabilities of different proposals
    total_weight = (regular_weight + PT_swap_weight + Fe_proposal_weight + 
                    draw_from_prior_weight + de_weight)
    swap_probability = PT_swap_weight/total_weight
    fe_proposal_probability = Fe_proposal_weight/total_weight
    regular_probability = regular_weight/total_weight
    draw_from_prior_probability = draw_from_prior_weight/total_weight
    de_probability = de_weight/total_weight
    print("Percentage of steps doing different jumps:\nPT swaps: {0:.2f}%\n\
Fe-proposals: {1:.2f}%\nJumps along Fisher eigendirections: {2:.2f}%\n\
Draw from prior: {3:.2f}%\nDifferential evolution jump: {4:.2f}%".format(swap_probability*100,
          fe_proposal_probability*100, regular_probability*100, draw_from_prior_probability*100,
          de_probability*100))

    for i in range(int(N-1)):
        #add current sample to DE history
        if i%n_de_history==0 and i>=de_start_iter:
            de_hist_index = int((i-de_start_iter)/n_de_history)%history_size
            de_history[:,de_hist_index,:] = samples[:,i,:]
        #print out run state every 10 iterations
        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            print('Progress: {0:2.2f}% '.format(i/N*100) +
                  'Acceptance fraction (swap, each chain): ({0:1.2f} '.format(acc_fraction[0]) +
                  ' '.join([',{{{}:1.2f}}'.format(i) for i in range(n_chain)]).format(*acc_fraction[1:]) +
                  ')' + '\r',end='')
        #update our eigenvectors from the fisher matrix every 100 iterations
        if i%n_fish_update==0:
            for j in range(n_chain):
                eigenvectors = get_fisher_eigenvectors(samples[j,i,:], pta, T_chain=Ts[j])
                #check if eigenvector calculation was succesful
                #if not, we just keep the initializes eig full of 0.1 values              
                if np.all(eigenvectors):
                    eig[j,:,:] = eigenvectors
        #draw a random number to decide which jump to do
        jump_decide = np.random.uniform()
        if jump_decide<swap_probability:
            do_pt_swap(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, swap_record)
        #global proposal based on Fe-statistic
        elif jump_decide<swap_probability+fe_proposal_probability:
            do_fe_global_jump(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, fe_file)
        #draw from prior move
        elif jump_decide<swap_probability+fe_proposal_probability+draw_from_prior_probability:
            do_draw_from_prior_move(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no)
        #do DE jump
        elif (jump_decide<swap_probability+fe_proposal_probability+
             draw_from_prior_probability+de_probability and i>=de_start_iter):
            do_de_jump(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, de_history)
        #regular step
        else:
            regular_jump(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, eig)
    acc_fraction = a_yes/(a_no+a_yes)
    return samples, acc_fraction, swap_record

################################################################################
#
#DIFFERENTIAL EVOLUTION PROPOSAL
#
################################################################################

def do_de_jump(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, de_history):
    de_indices = np.random.choice(de_history.shape[1], size=2, replace=False)

    #setting up our two x arrays and replace them with a random draw if the
    #have not been filled up yet with history
    x1 = de_history[:,de_indices[0],:]
    if np.array_equal(x1, np.zeros((n_chain, ndim))):
        for j in range(n_chain):
            x1[j,:] = np.hstack(p.sample() for p in pta.params)
    
    x2 = de_history[:,de_indices[1],:]
    if np.array_equal(x2, np.zeros((n_chain, ndim))):
        for j in range(n_chain):
            x2[j,:] = np.hstack(p.sample() for p in pta.params)

    alpha = 1.0
    if np.random.uniform()<0.9:
        alpha = np.random.normal(scale=2.38/np.sqrt(2*ndim))

    for j in range(n_chain):
        new_point = samples[j,i,:] + alpha*(x1[j,:]-x2[j,:])
        
        log_acc_ratio = pta.get_lnlikelihood(new_point[:])
        log_acc_ratio += pta.get_lnprior(new_point[:])
        log_acc_ratio += -pta.get_lnlikelihood(samples[j,i,:])
        log_acc_ratio += -pta.get_lnprior(samples[j,i,:])

        acc_ratio = np.exp(log_acc_ratio)**(1/Ts[j])
        if np.random.uniform()<=acc_ratio:
            for k in range(ndim):
                samples[j,i+1,k] = new_point[k]
            a_yes[j+1]+=1
        else:
            for k in range(ndim):
                samples[j,i+1,k] = samples[j,i,k]
            a_no[j+1]+=1


################################################################################
#
#DRAW FROM PRIOR MOVE
#
################################################################################

def do_draw_from_prior_move(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no):
    for j in range(n_chain):
        #make a rendom draw from the prior
        new_point = np.hstack(p.sample() for p in pta.params)

        #calculate acceptance ratio
        log_acc_ratio = pta.get_lnlikelihood(new_point[:])
        log_acc_ratio += pta.get_lnprior(new_point[:])
        log_acc_ratio += -pta.get_lnlikelihood(samples[j,i,:])
        log_acc_ratio += -pta.get_lnprior(samples[j,i,:])
        
        acc_ratio = np.exp(log_acc_ratio)**(1/Ts[j])
        if np.random.uniform()<=acc_ratio:
            for k in range(ndim):
                samples[j,i+1,k] = new_point[k]
            a_yes[j+1]+=1
        else:
            for k in range(ndim):
                samples[j,i+1,k] = samples[j,i,k]
            a_no[j+1]+=1

################################################################################
#
#GLOBAL PROPOSAL BASED ON FE-STATISTIC
#
################################################################################

def do_fe_global_jump(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, fe_file):    
    pass

################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS)
#
################################################################################

def regular_jump(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, eig):
    for j in range(n_chain):
        jump_select = np.random.randint(ndim)
        jump = eig[j,jump_select,:]

        new_point = samples[j,i,:] + jump*np.random.normal()

        log_acc_ratio = pta.get_lnlikelihood(new_point[:])
        log_acc_ratio += pta.get_lnprior(new_point[:])
        log_acc_ratio += -pta.get_lnlikelihood(samples[j,i,:])
        log_acc_ratio += -pta.get_lnprior(samples[j,i,:])

        acc_ratio = np.exp(log_acc_ratio)**(1/Ts[j])
        if np.random.uniform()<=acc_ratio:
            for k in range(ndim):
                samples[j,i+1,k] = new_point[k]
            a_yes[j+1]+=1
        else:
            for k in range(ndim):
                samples[j,i+1,k] = samples[j,i,k]
            a_no[j+1]+=1

################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, swap_record):
    swap_chain = np.random.randint(n_chain-1)

    log_acc_ratio = -pta.get_lnlikelihood(samples[swap_chain,i,:])/Ts[swap_chain]
    log_acc_ratio += -pta.get_lnprior(samples[swap_chain,i,:])/Ts[swap_chain]
    log_acc_ratio += -pta.get_lnlikelihood(samples[swap_chain+1,i,:])/Ts[swap_chain+1]
    log_acc_ratio += -pta.get_lnprior(samples[swap_chain+1,i,:])/Ts[swap_chain+1]
    log_acc_ratio += pta.get_lnlikelihood(samples[swap_chain+1,i,:])/Ts[swap_chain]
    log_acc_ratio += pta.get_lnprior(samples[swap_chain+1,i,:])/Ts[swap_chain]
    log_acc_ratio += pta.get_lnlikelihood(samples[swap_chain,i,:])/Ts[swap_chain+1]
    log_acc_ratio += pta.get_lnprior(samples[swap_chain,i,:])/Ts[swap_chain+1]

    acc_ratio = np.exp(log_acc_ratio)
    if np.random.uniform()<=acc_ratio:
        for j in range(n_chain):
            if j==swap_chain:
                for k in range(ndim):
                    samples[j,i+1,k] = samples[j+1,i,k]
            elif j==swap_chain+1:
                for k in range(ndim):
                    samples[j,i+1,k] = samples[j-1,i,k]
            else:
                for k in range(ndim):
                    samples[j,i+1,k] = samples[j,i,k]
        a_yes[0]+=1
        swap_record.append(swap_chain)
    else:
        for j in range(n_chain):
            for k in range(ndim):
                samples[j,i+1,k] = samples[j,i,k]
        a_no[0]+=1

################################################################################
#
#FISHER EIGENVALUE CALCULATION
#
################################################################################
def get_fisher_eigenvectors(params, pta, T_chain=1, epsilon=1e-5):
    #get dimension and set up an array for the fisher matrix    
    dim = params.shape[0]
    fisher = np.zeros((dim,dim))

    #lnlikelihood at specified point
    nn = pta.get_lnlikelihood(params)
    
    #calculate diagonal elements
    for i in range(dim):
        #create parameter vectors with +-epsilon in the ith component
        paramsPP = np.copy(params)
        paramsMM = np.copy(params)
        paramsPP[i] += 2*epsilon
        paramsMM[i] -= 2*epsilon
        
        #lnlikelihood at +-epsilon positions
        pp = pta.get_lnlikelihood(paramsPP)
        mm = pta.get_lnlikelihood(paramsMM)
        
        #calculate diagonal elements of the Hessian from a central finite element scheme
        #note the minus sign compared to the regular Hessian
        fisher[i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

    #calculate off-diagonal elements
    for i in range(dim):
        for j in range(i+1,dim):
            #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            paramsPM = np.copy(params)
            paramsMP = np.copy(params)

            paramsPP[i] += epsilon
            paramsPP[j] += epsilon
            paramsMM[i] -= epsilon
            paramsMM[j] -= epsilon
            paramsPM[i] += epsilon
            paramsPM[j] -= epsilon
            paramsMP[i] -= epsilon
            paramsMP[j] += epsilon

            #lnlikelihood at those positions
            pp = pta.get_lnlikelihood(paramsPP)
            mm = pta.get_lnlikelihood(paramsMM)
            pm = pta.get_lnlikelihood(paramsPM)
            mp = pta.get_lnlikelihood(paramsMP)

            #calculate off-diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher[i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
            fisher[j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
    
    #correct for the given temperature of the chain    
    fisher = fisher/T_chain
  
    try:
        #Filter nans and infs and replace them with 1s
        #this will imply that we will set the eigenvalue to 100 a few lines below
        FISHER = np.where(np.isfinite(fisher), fisher, 1.0)
        if not np.array_equal(FISHER, fisher):
            print("Changed some nan elements in the Fisher matrix to 1.0")

        #Find eigenvalues and eigenvectors of the Fisher matrix
        w, v = np.linalg.eig(FISHER)

        #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
        eig_limit = 100.0    
        W = np.where(np.abs(w)>eig_limit, w, eig_limit)

        eig = (np.sqrt(1.0/np.abs(W))*v).T

    except:
        print("An Error occured in the eigenvalue calculation")
        eig = np.array(False)
    
    return eig

################################################################################
#
#MAKE AN ARRAY CONTAINING GLOBAL PROPOSAL DENSITY FROM F_E-STATISTICS
#
################################################################################
def make_fe_global_proposal(fe_func, f_min=1e-9, f_max=1e-7, n_freq=400, NSIDE=8):
    m = np.zeros((n_freq, hp.nside2npix(NSIDE)))
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)

    idx = np.arange(hp.nside2npix(NSIDE))
    for i, f in enumerate(freqs):
        print("{0}th freq out of {1}".format(i, n_freq))
        m[i,:] = fe_func(f, np.array(hp.pix2ang(NSIDE, idx)))

    return m, freqs


