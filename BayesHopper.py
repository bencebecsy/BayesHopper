################################################################################
#
#BayesHopper -- **Bayes**ian **H**yp**o**thesis testing for **p**ulsar timing arrays with **per**iodic signals
#
#Bence Bécsy (bencebecsy@montana.edu) -- 2019
################################################################################

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import json

import enterprise
from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils
from enterprise.signals import selections

from enterprise_extensions.frequentist import Fe_statistic
from enterprise_extensions import deterministic

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################

def run_ptmcmc(N, T_max, n_chain, pulsars, max_n_source=1, n_source_prior='flat', n_source_start='random', RJ_weight=0,
               regular_weight=3, noise_jump_weight=3, PT_swap_weight=1, T_ladder = None, T_dynamic=False, T_dynamic_nu=2000, T_dynamic_t0=2000, PT_hist_length=1000,
               Fe_proposal_weight=0, fe_file=None, Fe_pdet=0.5, Fe_alpha=0.1,
               prior_recovery=False, cw_amp_prior='uniform', gwb_amp_prior='uniform', rn_amp_prior='uniform', per_psr_rn_amp_prior='uniform',
               gwb_log_amp_range=[-18,-11], n_comp_common=30, n_comp_per_psr_rn=30, rn_log_amp_range=[-18,-11], per_psr_rn_log_amp_range=[-18,-11],
               vary_gwb_gamma=True, vary_rn_gamma=True,
               cw_log_amp_range=[-18,-11], cw_f_range=[3.5e-9,1e-7],
               vary_white_noise=False, efac_start=1.0,
               include_gwb=False, gwb_switch_weight=0, include_psr_term=False,
               include_rn=False, include_per_psr_rn=False, vary_rn=False, vary_per_psr_rn=False, rn_params=[-13.0,1.0], per_psr_rn_start_file=None,
               rn_on_prior=0.5, rn_switch_weight=0, jupyter_notebook=False,
               gwb_on_prior=0.5, rn_gwb_on_prior=None, include_equad_ecorr=False, wn_backend_selection=False, noisedict_file=None,
               save_every_n=10000, savefile=None, resume_from=None, n_status_update = 100,
               rn_gwb_move_weight=0):

    ptas = get_ptas(pulsars, vary_white_noise=vary_white_noise, include_equad_ecorr=include_equad_ecorr, wn_backend_selection=wn_backend_selection, noisedict_file=noisedict_file, include_rn=include_rn, include_per_psr_rn=include_per_psr_rn, vary_rn=vary_rn, vary_per_psr_rn=vary_per_psr_rn, include_gwb=include_gwb, max_n_source=max_n_source, efac_start=efac_start, rn_amp_prior=rn_amp_prior, rn_log_amp_range=rn_log_amp_range, per_psr_rn_amp_prior=per_psr_rn_amp_prior, per_psr_rn_log_amp_range=per_psr_rn_log_amp_range, rn_params=rn_params, gwb_amp_prior=gwb_amp_prior, gwb_log_amp_range=gwb_log_amp_range, n_comp_common=n_comp_common, n_comp_per_psr_rn=n_comp_per_psr_rn, vary_gwb_gamma=vary_gwb_gamma, vary_rn_gamma=vary_rn_gamma, cw_amp_prior=cw_amp_prior, cw_log_amp_range=cw_log_amp_range, cw_f_range=cw_f_range, include_psr_term=include_psr_term, prior_recovery=prior_recovery)

    print(ptas)
    for i in range(max_n_source+1):
        for j in range(1+int(include_gwb)):
            for k in range(1+int(include_rn)):
                print(i, j, k)
                print(ptas[i][j][k].params)
                #point_to_test = np.tile(np.array([0.0, 0.54, 1.0, -8.0, -13.39, 2.0, 0.5]),i+1)
                #print(PTA.summary())

    print("In Fe-proposal we will use p_det={0} and alpha={1}".format(Fe_pdet, Fe_alpha))

    #fisher updating every n_fish_update step
    n_fish_update = 200 #50

    #setting up temperature ladder
    if T_ladder is None:
        #using geometric spacing
        c = T_max**(1.0/(n_chain-1))
        Ts = c**np.arange(n_chain)
        print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\
 Temperature ladder is:\n".format(n_chain,c),Ts)
    else:
        Ts = np.array(T_ladder)
        n_chain = Ts.size
        print("Using {0} temperature chains with custom spacing: ".format(n_chain),Ts)
 
    #make highest temperature inf if dynamic T ladder is used
    if T_dynamic:
        Ts[-1] = np.inf

    if T_dynamic:
        print("Dynamic temperature adjustment: ON")
    else:
        print("Dynamic temperature adjustment: OFF")

    #set up array to hold acceptance probabilities of last PT_hist_length PT swaps
    PT_hist = np.ones((n_chain-1,PT_hist_length))*np.nan #initiated with NaNs
    PT_hist_idx = np.array([0]) #index to keep track of which row to update in PT_hist

    #GWB and RN on/off priors
    if include_gwb and include_rn and rn_gwb_on_prior is None:
        rn_gwb_on_prior = np.array([[(1-rn_on_prior)*(1-gwb_on_prior), rn_on_prior*(1-gwb_on_prior)],
                                    [(1-rn_on_prior)*gwb_on_prior, rn_on_prior*gwb_on_prior]])
    else:
        rn_gwb_on_prior = np.array(rn_gwb_on_prior)
        rn_gwb_on_prior = rn_gwb_on_prior/np.sum(rn_gwb_on_prior)

    #printitng out the prior used on GWB and RN on/off
    if include_gwb and include_rn:
        print("Prior for GWB and RN both off: {0}%".format(rn_gwb_on_prior[0,0]*100))
        print("Prior for GWB and RN both on: {0}%".format(rn_gwb_on_prior[1,1]*100))
        print("Prior for GWB on and RN off: {0}%".format(rn_gwb_on_prior[1,0]*100))
        print("Prior for GWB off and RN on: {0}%".format(rn_gwb_on_prior[0,1]*100))
        print(rn_gwb_on_prior)

    #set up and print out prior on number of sources
    if max_n_source!=0:
        if n_source_prior=='flat':
            n_source_prior = np.ones(max_n_source+1)/(max_n_source+1)
        else:
            n_source_prior = np.array(n_source_prior)
            n_prior_norm = np.sum(n_source_prior)
            n_source_prior *= 1.0/n_prior_norm
        print("Prior on number of sources: ", n_source_prior)

    #setting up array for the samples
    num_params = max_n_source*7+1
    if include_gwb:
        if vary_gwb_gamma:
            num_params += 2
        else:
            num_params += 1
    
    num_per_psr_params = 0
    num_noise_params = 0
    if vary_white_noise:
        num_per_psr_params += len(pulsars)
        num_noise_params += len(pulsars)
    if vary_rn:
        if vary_rn_gamma:
            num_noise_params += 2
        else:
            num_noise_params += 1
    if vary_per_psr_rn:
        num_per_psr_params += 2*len(pulsars)
        num_noise_params += 2*len(pulsars)

    num_params += num_noise_params
    print("# of parameters: ", num_params)
    print("# of noise parameters: ", num_noise_params)
    print("# of per psr parameters: ", num_per_psr_params)


    if resume_from is not None:
        print("Resuming from file: " + resume_from)
        npzfile = np.load(resume_from)
        swap_record = list(npzfile['swap_record'])
        log_likelihood_resume = npzfile['log_likelihood']
        betas_resume = npzfile['betas']
        PT_acc_resume = npzfile['PT_acc']
        samples_resume = npzfile['samples']
        
        N_resume = samples_resume.shape[1]
        print("# of samples sucessfully read in: " + str(N_resume))

        samples = np.zeros((n_chain, N_resume+N, num_params))
        samples[:,:N_resume,:] = np.copy(samples_resume)

        log_likelihood = np.zeros((n_chain,N_resume+N))
        log_likelihood[:,:N_resume] = np.copy(log_likelihood_resume)
        betas = np.ones((n_chain,N_resume+N))
        betas[:,:N_resume] = np.copy(betas_resume)
        PT_acc = np.zeros((n_chain-1,N_resume+N))
        PT_acc[:,:N_resume] = np.copy(PT_acc_resume)
    else:
        samples = np.zeros((n_chain, N, num_params))

        #set up log_likelihood array
        log_likelihood = np.zeros((n_chain,N))

        #set up betas array with PT inverse temperatures
        betas = np.ones((n_chain,N))
        #fill first row with initial betas
        betas[:,0] = 1/Ts
        print("Initial beta (1/T) ladder is:\n",betas[:,0])

        #set up array holding PT acceptance rate for each iteration
        PT_acc = np.zeros((n_chain-1,N))

        #filling first sample with random draw
        for j in range(n_chain):
            if n_source_start is 'random':
                n_source = np.random.choice(max_n_source+1)
            else:
                n_source = n_source_start
            samples[j,0,0] = n_source
            print(n_source)
            if n_source!=0:
                samples[j,0,1:n_source*7+1] = np.hstack(p.sample() for p in ptas[n_source][0][0].params[:n_source*7])
            if vary_white_noise and not vary_per_psr_rn:
                samples[j,0,max_n_source*7+1:max_n_source*7+1+len(pulsars)] = np.ones(len(pulsars))*efac_start
            elif vary_per_psr_rn and not vary_white_noise:
                if per_psr_rn_start_file==None:
                    samples[j,0,max_n_source*7+1:max_n_source*7+1+2*len(pulsars)] = np.hstack(p.sample() for p in ptas[0][0][0].params[:2*len(pulsars)])
                else:
                    RN_noise_data = np.load(per_psr_rn_start_file)
                    samples[j,0,max_n_source*7+1:max_n_source*7+1+2*len(pulsars)] = RN_noise_data['RN_start']
            elif vary_per_psr_rn and vary_white_noise: #vary both per psr RN and WN
                samples[j,0,max_n_source*7+1:max_n_source*7+1+3*len(pulsars)] = np.hstack(p.sample() for p in ptas[0][0][0].params[:3*len(pulsars)])
            if vary_rn:
                if vary_rn_gamma:
                    samples[j,0,max_n_source*7+1+num_per_psr_params:max_n_source*7+1+num_noise_params] = np.array([ptas[n_source][0][1].params[n_source*7+num_per_psr_params].sample(), ptas[n_source][0][1].params[n_source*7+num_per_psr_params+1].sample()])
                else:
                    samples[j,0,max_n_source*7+1+num_per_psr_params:max_n_source*7+1+num_noise_params] = ptas[n_source][0][1].params[n_source*7+num_per_psr_params].sample()
            if include_gwb:
                if vary_gwb_gamma:
                    samples[j,0,max_n_source*7+1+num_noise_params:max_n_source*7+1+num_noise_params+2] = np.array([ptas[n_source][1][1].params[n_source*7+num_noise_params].sample(), ptas[n_source][1][1].params[n_source*7+num_noise_params+1].sample()])

                else:
                    samples[j,0,max_n_source*7+1+num_noise_params] = ptas[n_source][1][1].params[n_source*7+num_noise_params].sample()
        #printing info about initial parameters
        for j in range(n_chain):
            print(j)
            print(samples[j,0,:])
            n_source = get_n_source(samples,j,0)
            gwb_on = get_gwb_on(samples,j,0,max_n_source,num_noise_params)
            rn_on = get_rn_on(samples,j,0,max_n_source,num_per_psr_params)
            first_sample = strip_samples(samples[j,0,:],n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
            print(first_sample)
            log_likelihood[j,0] = ptas[n_source][gwb_on][rn_on].get_lnlikelihood(first_sample)
            print(log_likelihood[j,0])
            print(ptas[n_source][gwb_on][rn_on].get_lnprior(first_sample))

    #setting up array for the fisher eigenvalues
    #one for cw parameters which we will keep updating
    eig = np.ones((n_chain, max_n_source, 7, 7))*0.1
    
    #one for GWB and common rn parameters, which we will keep updating
    if include_gwb:
        if vary_gwb_gamma and vary_rn_gamma:
            eig_gwb_rn = np.broadcast_to( np.array([[1.0,0,0,0], [0,0.3,0,0], [0,0,1.0,0], [0,0,0,0.3]]), (n_chain, 4, 4)).copy()
        elif not vary_gwb_gamma and vary_rn_gamma:
            eig_gwb_rn = np.broadcast_to( np.array([[1.0,0,0], [0,0.3,0], [0,0,0.3]]), (n_chain, 3, 3)).copy()
        elif vary_gwb_gamma and not vary_rn_gamma:
            eig_gwb_rn = np.broadcast_to( np.array([[0.3,0,0], [0,1.0,0], [0,0,0.3]]), (n_chain, 3, 3)).copy()
        else:
            eig_gwb_rn = eig_gwb_rn = np.broadcast_to( np.array([[0.3,0], [0,0.3]]), (n_chain, 2, 2)).copy()
    else:
        if vary_rn_gamma:
            eig_gwb_rn = np.broadcast_to( np.array([[1.0,0], [0,0.3]]), (n_chain, 2, 2)).copy()
        else:
            eig_gwb_rn = np.broadcast_to( np.array([[0.3,],]), (n_chain, 1, 1)).copy()

    print(eig_gwb_rn[0,:,:])
    #and one for per psr noise (WN and RN) parameters, which we will not update
    if vary_white_noise and not vary_per_psr_rn:
        eig_per_psr = np.broadcast_to(np.eye(len(pulsars))*0.1, (n_chain,len(pulsars), len(pulsars)) ).copy()
        #calculate wn eigenvectors
        for j in range(n_chain):
            n_source = get_n_source(samples,j,0)
            per_psr_eigvec = get_fisher_eigenvectors(strip_samples(samples[j,i,:],n_source,0,0,max_n_source,num_per_psr_params,num_noise_params,num_params), ptas[n_source][0][0], T_chain=1/betas[j,0], n_source=1, dim=len(pulsars), offset=n_source*7)
            eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]
    elif vary_per_psr_rn and not vary_white_noise:
        eig_per_psr = np.broadcast_to(np.eye(2*len(pulsars))*0.1, (n_chain,2*len(pulsars), 2*len(pulsars)) ).copy()
        #calculate wn eigenvectors
        for j in range(n_chain):
            n_source = get_n_source(samples,j,0)
            per_psr_eigvec = get_fisher_eigenvectors(strip_samples(samples[j,i,:],n_source,0,0,max_n_source,num_per_psr_params,num_noise_params,num_params), ptas[n_source][0][0], T_chain=1/betas[j,0], n_source=1, dim=2*len(pulsars), offset=n_source*7)
            eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]
            #if j==0: print(eig_per_psr[0,:,:])
    elif vary_per_psr_rn and vary_white_noise: #vary both per psr RN and WN
        eig_per_psr = np.broadcast_to(np.eye(3*len(pulsars))*0.1, (n_chain,3*len(pulsars), 3*len(pulsars)) ).copy()
        #calculate wn eigenvectors
        for j in range(n_chain):
            n_source = get_n_source(samples,j,0)
            per_psr_eigvec = get_fisher_eigenvectors(strip_samples(samples[j,i,:],n_source,0,0,max_n_source,num_per_psr_params,num_noise_params,num_params), ptas[n_source][0][0], T_chain=1/betas[j,0], n_source=1, dim=3*len(pulsars), offset=n_source*7)
            eig_per_psr[j,:,:] = per_psr_eigvec[0,:,:]

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros((8,n_chain)) #columns: chain number; rows: proposal type (RJ_CW, gwb_switch, rn_switch, rn_gwb, PT, Fe, fisher, noise_jump)
    a_no=np.zeros((8,n_chain))
    acc_fraction = a_yes/(a_no+a_yes)
    if resume_from is None:
        swap_record = []
    rj_record = []

    #read in fe data if we will need it
    if Fe_proposal_weight+RJ_weight>0:
        if fe_file==None:
            raise Exception("Fe-statistics data file is needed for Fe global propsals")
        npzfile = np.load(fe_file)
        freqs = npzfile['freqs']
        fe = npzfile['fe']
        inc_max = npzfile['inc_max']
        psi_max = npzfile['psi_max']
        phase0_max = npzfile['phase0_max']
        h_max = npzfile['h_max']
        
        if Fe_proposal_weight>0:
            #psi
            psi_hist, psi_bin_edges = np.histogram(psi_max.flatten(), bins=100, density=True)
            psi_bin_centers = []
            for k in range(len(psi_bin_edges)-1):
                psi_bin_centers.append((psi_bin_edges[k+1]+psi_bin_edges[k])/2)
            psi_bin_centers = np.array(psi_bin_centers)
            def psi_pdf(x):
                if x>psi_bin_edges[0] and x<psi_bin_edges[-1]:
                    bin_idx = (np.abs(psi_bin_centers - x)).argmin()
                    return psi_hist[bin_idx]
                else:
                    return 0.0
            #inc
            inc_hist, inc_bin_edges = np.histogram(np.cos(inc_max.flatten()), bins=100, density=True)
            inc_bin_centers = []
            for k in range(len(inc_bin_edges)-1):
                inc_bin_centers.append((inc_bin_edges[k+1]+inc_bin_edges[k])/2)
            inc_bin_centers = np.array(inc_bin_centers)
            def cos_inc_pdf(x):
                if x>inc_bin_edges[0] and x<inc_bin_edges[-1]:
                    bin_idx = (np.abs(inc_bin_centers - x)).argmin()
                    return inc_hist[bin_idx]
                else:
                    return 0.0
            #phase0
            phase0_hist, phase0_bin_edges = np.histogram(phase0_max.flatten(), bins=100, density=True)
            phase0_bin_centers = []
            for k in range(len(phase0_bin_edges)-1):
                phase0_bin_centers.append((phase0_bin_edges[k+1]+phase0_bin_edges[k])/2)
            phase0_bin_centers = np.array(phase0_bin_centers)
            def phase0_pdf(x):
                if x>phase0_bin_edges[0] and x<phase0_bin_edges[-1]:
                    bin_idx = (np.abs(phase0_bin_centers - x)).argmin()
                    return phase0_hist[bin_idx]
                else:
                    return 0.0
            #h
            h_hist, h_bin_edges = np.histogram(np.log10(h_max.flatten()), bins=1000, range=(cw_log_amp_range[0],cw_log_amp_range[1]), density=True)
            h_bin_centers = []
            for k in range(len(h_bin_edges)-1):
                h_bin_centers.append((h_bin_edges[k+1]+h_bin_edges[k])/2)
            h_bin_centers = np.array(h_bin_centers)
            def log_h_pdf(x):
                if x>h_bin_edges[0] and x<h_bin_edges[-1]:
                    bin_idx = (np.abs(h_bin_centers - x)).argmin()
                    return h_hist[bin_idx]
                else:
                    return 0.0

    #set up probabilities of different proposals
    total_weight = (regular_weight + PT_swap_weight + Fe_proposal_weight + 
                    RJ_weight + gwb_switch_weight + noise_jump_weight + rn_switch_weight + rn_gwb_move_weight)
    swap_probability = PT_swap_weight/total_weight
    fe_proposal_probability = Fe_proposal_weight/total_weight
    regular_probability = regular_weight/total_weight
    RJ_probability = RJ_weight/total_weight
    gwb_switch_probability = gwb_switch_weight/total_weight
    rn_switch_probability = rn_switch_weight/total_weight
    noise_jump_probability = noise_jump_weight/total_weight
    rn_gwb_move_probability = rn_gwb_move_weight/total_weight
    print("Percentage of steps doing different jumps:\nPT swaps: {0:.2f}%\nRJ moves: {3:.2f}%\nGWB-switches: {4:.2f}%\nRN-switches: {5:.2f}%\nRN-GWB moves: {6:.2f}%\n\
Fe-proposals: {1:.2f}%\nJumps along Fisher eigendirections: {2:.2f}%\nNoise jump: {7:.2f}%".format(swap_probability*100, fe_proposal_probability*100, regular_probability*100,
          RJ_probability*100, gwb_switch_probability*100, rn_switch_probability*100, rn_gwb_move_probability*100, noise_jump_probability*100))

    if resume_from is None:
        start_iter = 0
        stop_iter = N
    else:
        start_iter = N_resume-1 #-1 because if only 1 sample is read in that's the same as having a different starting point and start_iter should still be 0
        stop_iter = N_resume-1+N

    for i in range(int(start_iter), int(stop_iter-1)): #-1 because ith step here produces (i+1)th sample based on ith sample
        #print(samples[:,i,:])
        ########################################################
        #
        #logging PT acceptance fraction
        #
        ########################################################
        #logging cumulative acc fraction
        #acc_fraction = a_yes/(a_no+a_yes)
        #PT_acc[:,i] = np.copy(acc_fraction[5,:])

        #logging logging mean acc probability over last PT_hist_length swaps
        PT_acc[:,i] = np.nanmean(PT_hist, axis=1) #nanmean so early on when we still have nans we only use the actual data
        ########################################################
        #
        #update temperature ladder
        #
        ########################################################
        if i>0:
            if T_dynamic and PT_hist_idx>0: #based on arXiv:1501.05823 and https://github.com/willvousden/ptemcee
                kappa = 1.0/T_dynamic_nu * T_dynamic_t0/(PT_hist_idx+T_dynamic_t0)
                #dSs = kappa * (acc_fraction[5,:-2] - acc_fraction[5,1:-1])
                dSs = kappa * (PT_acc[:-1,i] - PT_acc[1:,i])
                deltaTs = np.diff(1 / betas[:-1,i-1])
                deltaTs *= np.exp(dSs)

                new_betas = 1 / (np.cumsum(deltaTs) + 1 / betas[0,i-1])

                #set new betas
                betas[-1,i] = 0.0
                betas[1:-1,i] = np.copy(new_betas)
            else:
                #copy betas from previous iteration
                betas[:,i] = betas[:,i-1]
        ########################################################
        #
        #write results to file every save_every_n iterations
        #
        ########################################################
        if savefile is not None and i%save_every_n==0 and i!=start_iter:
            np.savez(savefile, samples=samples[:,:i,:], acc_fraction=acc_fraction, swap_record=swap_record, log_likelihood=log_likelihood[:,:i],
                     betas=betas[:,:i], PT_acc=PT_acc[:,:i])
        ########################################################
        #
        #print out run state every n_status_update iterations
        #
        ########################################################
        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            if jupyter_notebook:
                print('Progress: {0:2.2f}% '.format(i/N*100) + '\r',end='')
            else:
                print('Progress: {0:2.2f}% '.format(i/N*100) +
                      'Acceptance fraction #columns: chain number; rows: proposal type (RJ_CW, gwb_switch, rn_switch, rn_gwb, PT, Fe, fisher, noise_jump):')
                print(acc_fraction)
                #print(PT_hist)
                print(PT_acc[:,i])
        #################################################################################
        #
        #update our eigenvectors from the fisher matrix every n_fish_update iterations
        #
        #################################################################################
        if i%n_fish_update==0:
            #only update T>1 chains every 10th time
            if i%(n_fish_update*10)==0:
                for j in range(n_chain):
                    n_source = get_n_source(samples,j,i)
                    if include_gwb:
                        gwb_on = get_gwb_on(samples,j,i,max_n_source,num_noise_params)
                    else:
                        gwb_on = 0
                    #GWB-RN eigenvectors
                    if include_gwb:
                        if vary_gwb_gamma and vary_rn_gamma:
                            dim = 4
                        elif (not vary_gwb_gamma and vary_rn_gamma) or (vary_gwb_gamma and not vary_rn_gamma):
                            dim = 3
                        else:
                            dim = 2
                    else:
                        if vary_rn_gamma:
                            dim = 2
                        else:
                            dim = 1
                    eigvec_rn = get_fisher_eigenvectors(strip_samples(samples[j,i,:],n_source,1,1,max_n_source,num_per_psr_params,num_noise_params,num_params), ptas[n_source][1][1], T_chain=1/betas[j,i], n_source=1, dim=dim, offset=n_source*7+num_per_psr_params)
                    if np.all(eigvec_rn):
                        eig_gwb_rn[j,:,:] = eigvec_rn[0,:,:]

                    #CW eigenvectors
                    if n_source!=0:
                        eigenvectors = get_fisher_eigenvectors(strip_samples(samples[j,i,:],n_source,1,1,max_n_source,num_per_psr_params,num_noise_params,num_params), ptas[n_source][1][1], T_chain=1/betas[j,i], n_source=n_source)
                        if np.all(eigenvectors):
                            eig[j,:n_source,:,:] = eigenvectors
            elif samples[0,i,0]!=0:
                n_source = get_n_source(samples,0,i)
                if include_gwb:
                    gwb_on = get_gwb_on(samples,0,i,max_n_source,num_noise_params)
                else:
                    gwb_on = 0
                eigenvectors = get_fisher_eigenvectors(strip_samples(samples[j,i,:],n_source,1,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params), ptas[n_source][gwb_on][1], T_chain=1/betas[0,i], n_source=n_source)
                #check if eigenvector calculation was succesful
                #if not, we just keep the initializes eig full of 0.1 values              
                if np.all(eigenvectors):
                    eig[0,:n_source,:,:] = eigenvectors
        ###########################################################
        #
        #Do the actual MCMC step
        #
        ###########################################################
        #draw a random number to decide which jump to do
        jump_decide = np.random.uniform()
        #PT swap move
        if jump_decide<swap_probability:
            do_pt_swap(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_params, num_noise_params, num_per_psr_params, log_likelihood, PT_hist, PT_hist_idx)
        #global proposal based on Fe-statistic
        elif jump_decide<swap_probability+fe_proposal_probability:
            do_fe_global_jump(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, freqs, fe, inc_max, psi_max, phase0_max, h_max, vary_white_noise, include_gwb, num_params, num_noise_params, num_per_psr_params, Fe_pdet, Fe_alpha, psi_pdf, cos_inc_pdf, phase0_pdf, log_h_pdf, log_likelihood)
        #do RJ move
        elif (jump_decide<swap_probability+fe_proposal_probability+RJ_probability):
            do_rj_move(n_chain, max_n_source, n_source_prior, ptas, samples, i, betas, a_yes, a_no, freqs, fe, inc_max, psi_max, phase0_max, h_max, rj_record, vary_white_noise, include_gwb, num_params, num_noise_params, num_per_psr_params, log_likelihood)
        #do GWB switch move
        elif (jump_decide<swap_probability+fe_proposal_probability+RJ_probability+gwb_switch_probability):
            gwb_switch_move(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, vary_white_noise, include_gwb, num_params, num_noise_params, num_per_psr_params, rn_gwb_on_prior, gwb_log_amp_range, vary_gwb_gamma, log_likelihood)
        #do noise jump
        elif (jump_decide<swap_probability+fe_proposal_probability+RJ_probability+gwb_switch_probability+noise_jump_probability):
            noise_jump(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, eig_per_psr, include_gwb, num_params, num_noise_params, num_per_psr_params, vary_white_noise, log_likelihood)
        #do RN switch move
        elif (jump_decide<swap_probability+fe_proposal_probability+RJ_probability+gwb_switch_probability+noise_jump_probability+rn_switch_probability):
            rn_switch_move(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, vary_white_noise, include_rn, num_params, num_noise_params, num_per_psr_params, rn_gwb_on_prior, rn_log_amp_range, vary_rn_gamma, log_likelihood)
        #do RN-GWB move
        elif (jump_decide<swap_probability+fe_proposal_probability+RJ_probability+gwb_switch_probability+noise_jump_probability+rn_switch_probability+rn_gwb_move_probability):
            rn_gwb_move(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, vary_white_noise, include_rn, include_gwb, num_params, num_noise_params, num_per_psr_params, rn_gwb_on_prior, rn_log_amp_range, gwb_log_amp_range, vary_rn_gamma, vary_gwb_gamma, log_likelihood)
        #regular step
        else:
            regular_jump(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, eig, eig_gwb_rn, include_gwb, num_params, num_noise_params, num_per_psr_params, vary_rn, vary_rn_gamma, vary_gwb_gamma, log_likelihood)
    
    acc_fraction = a_yes/(a_no+a_yes)
    return samples, acc_fraction, swap_record, rj_record, ptas, log_likelihood, betas, PT_acc

################################################################################
#
#RN - GWB MOVE (EXCHANGE IF ONE IS ON, MIXING IF BOTH IS ON)
#
################################################################################
def rn_gwb_move(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, vary_white_noise, include_rn, include_gwb, num_params, num_noise_params, num_per_psr_params, rn_gwb_on_prior, rn_log_amp_range, gwb_log_amp_range, vary_rn_gamma, vary_gwb_gamma, log_likelihood):
    #print("RN GWB")
    if not include_rn or not include_gwb:
       raise Exception("Both include_rn and include_gwb must be True to use this move")
    for j in range(n_chain):
        n_source = get_n_source(samples,j,i)
        gwb_on = get_gwb_on(samples,j,i,max_n_source,num_noise_params)
        rn_on = get_rn_on(samples,j,i,max_n_source,num_per_psr_params)

        #no gwb or rn on -- nothing to vary
        if gwb_on==0 and rn_on==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[3,j] += 1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("Nothing to vary!")
        #Both are on -- mixing move
        elif gwb_on==1 and rn_on==1:
            samples_current = np.copy(samples[j,i,:])
            old_gwb_log_amp = np.copy(samples_current[1+max_n_source*7+num_noise_params])
            old_rn_log_amp = np.copy(samples_current[1+max_n_source*7+num_per_psr_params])

            #draw randomly exchange fraction, i.e. fraction of GWB to add to RN (if positive) or fraction of RN to add to GWB (if negative)
            x_fraction = np.random.uniform(low=-1, high=1)

            if x_fraction > 0.0: #removing power from GWB and adding it to RN
                amp_change = x_fraction * 10**old_gwb_log_amp
                new_gwb_log_amp = np.log10(10**old_gwb_log_amp - amp_change)
                new_rn_log_amp = np.log10(10**old_rn_log_amp + amp_change)
            elif x_fraction < 0.0: #removing power from RN and adding it to GWB
                amp_change = -x_fraction * 10**old_rn_log_amp
                new_gwb_log_amp = np.log10(10**old_gwb_log_amp + amp_change)
                new_rn_log_amp = np.log10(10**old_rn_log_amp - amp_change)

            new_point = np.copy(samples[j,i,:])
            #set gwb and rn amplitude to new values
            new_point[1+max_n_source*7+num_noise_params] = new_gwb_log_amp #gwb
            new_point[1+max_n_source*7+num_per_psr_params] = new_rn_log_amp #rn
    
            new_point_stripped = strip_samples(new_point,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
            samples_current_stripped = strip_samples(samples_current,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)

            log_L = ptas[n_source][1][1].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[n_source][1][1].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][1][1].get_lnprior(samples_current_stripped)
            
            acc_ratio = np.exp(log_acc_ratio)

            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,:] = np.copy(new_point)
                a_yes[3,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[3,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
        
        #Switching from GWB to RN
        elif gwb_on==1 and rn_on==0:
            samples_current = np.copy(samples[j,i,:])
            new_point = np.copy(samples[j,i,:])
            if vary_gwb_gamma:
                old_gamma = np.copy(samples_current[1+max_n_source*7+num_noise_params])
                old_log_amp = np.copy(samples_current[1+max_n_source*7+num_noise_params+1])
            else:
                old_log_amp = np.copy(samples_current[1+max_n_source*7+num_noise_params])

            #set gwb_amplitude (and if varied gwb_gamma) to zero
            new_point[1+max_n_source*7+num_noise_params] = 0.0
            if vary_gwb_gamma:
                new_point[1+max_n_source*7+num_noise_params+1] = 0.0
            
            #instead turn on RN with same amplitude as GWB had and w/ random spectral index
            if vary_gwb_gamma and vary_rn_gamma:
                new_point[1+max_n_source*7+num_per_psr_params] = old_gamma #gamma
                new_point[1+max_n_source*7+num_per_psr_params+1] = old_log_amp #amplitude
            elif not vary_gwb_gamma and vary_rn_gamma:
                new_gamma = ptas[n_source][1][1].params[n_source*7+num_per_psr_params].sample()
                new_point[1+max_n_source*7+num_per_psr_params] = new_gamma #gamma
                new_point[1+max_n_source*7+num_per_psr_params+1] = old_log_amp #amplitude
            else:
                new_point[1+max_n_source*7+num_per_psr_params] = old_log_amp

            new_point_stripped = strip_samples(new_point,n_source,1,0,max_n_source,num_per_psr_params,num_noise_params,num_params)
            samples_current_stripped = strip_samples(samples_current,n_source,0,1,max_n_source,num_per_psr_params,num_noise_params,num_params)

            log_L = ptas[n_source][0][1].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[n_source][0][1].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][1][0].get_lnprior(samples_current_stripped)

            acc_ratio = np.exp(log_acc_ratio)
            #apply proposal densities if needed
            if not vary_gwb_gamma and vary_rn_gamma:
                gamma_proposal_new = ptas[n_source][1][1].params[n_source*7+num_per_psr_params].get_pdf(new_gamma)
                acc_ratio *= 1/gamma_proposal_new
            elif vary_gwb_gamma and not vary_rn_gamma:
                gamma_proposal_old = ptas[n_source][1][1].params[n_source*7+num_noise_params].get_pdf(old_gamma)
                acc_ratio *= gamma_proposal_old

            #apply prior
            acc_ratio *= rn_gwb_on_prior[0,1]/rn_gwb_on_prior[1,0]

            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,:] = np.copy(new_point)
                a_yes[3,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[3,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]

        #Switching from RN to GWB
        else:
            samples_current = np.copy(samples[j,i,:])
            new_point = np.copy(samples[j,i,:])
            if vary_rn_gamma:
                old_log_amp = np.copy(samples_current[1+max_n_source*7+num_per_psr_params+1])
                old_gamma = np.copy(samples_current[1+max_n_source*7+num_per_psr_params])
            else:
                old_log_amp = np.copy(samples_current[1+max_n_source*7+num_per_psr_params])
            
            #set both amplitude (and gamma if varied) to zero
            new_point[1+max_n_source*7+num_per_psr_params] = 0.0
            if vary_rn_gamma:
                new_point[1+max_n_source*7+num_per_psr_params+1] = 0.0

            #instead turn on GWB with same amplitude as RN had
            if vary_rn_gamma and vary_gwb_gamma:
                new_point[1+max_n_source*7+num_noise_params] = old_gamma
                new_point[1+max_n_source*7+num_noise_params+1] = old_log_amp
            elif not vary_rn_gamma and vary_gwb_gamma:
                new_gamma = ptas[n_source][1][1].params[n_source*7+num_noise_params].sample()
                new_point[1+max_n_source*7+num_noise_params] = new_gamma
                new_point[1+max_n_source*7+num_noise_params+1] = old_log_amp
            else:
                new_point[1+max_n_source*7+num_noise_params] = old_log_amp

            new_point_stripped = strip_samples(new_point,n_source,0,1,max_n_source,num_per_psr_params,num_noise_params,num_params)
            samples_current_stripped = strip_samples(samples_current,n_source,1,0,max_n_source,num_per_psr_params,num_noise_params,num_params)

            log_L = ptas[n_source][1][0].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[n_source][1][0].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][0][1].get_lnprior(samples_current_stripped)

            acc_ratio = np.exp(log_acc_ratio)

            #apply proposal densities if needed
            if not vary_rn_gamma and vary_gwb_gamma:
                gamma_proposal_new = ptas[n_source][1][1].params[n_source*7+num_noise_params].get_pdf(new_gamma)
                acc_ratio *= 1/gamma_proposal_new
            elif vary_rn_gamma and not vary_gwb_gamma:
                gamma_proposal_old = ptas[n_source][1][1].params[n_source*7+num_per_psr_params].get_pdf(old_gamma)
                acc_ratio *= gamma_proposal_old

            #apply priors
            acc_ratio *= rn_gwb_on_prior[1,0]/rn_gwb_on_prior[0,1]

            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,:] = np.copy(new_point)
                a_yes[3,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[3,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]

################################################################################
#
#RN SWITCH (ON/OFF) MOVE
#
################################################################################
def rn_switch_move(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, vary_white_noise, include_rn, num_params, num_noise_params, num_per_psr_params, rn_gwb_on_prior, rn_log_amp_range, vary_rn_gamma, log_likelihood):
    #print("RN switch")
    if not include_rn:
       raise Exception("include_rn must be True to use this move")
    for j in range(n_chain):
        n_source = get_n_source(samples,j,i)
        gwb_on = get_gwb_on(samples,j,i,max_n_source,num_noise_params)
        rn_on = get_rn_on(samples,j,i,max_n_source,num_per_psr_params)
        
        #turning off ---------------------------------------------------------------------------------------------------------
        if rn_on==1:
            samples_current = np.copy(samples[j,i,:])
            new_point = np.copy(samples[j,i,:])
            if vary_rn_gamma:
                old_log_amp = np.copy(samples_current[1+max_n_source*7+num_per_psr_params+1])
                old_gamma = np.copy(samples_current[1+max_n_source*7+num_per_psr_params])
            else:
                old_log_amp = np.copy(samples_current[1+max_n_source*7+num_per_psr_params])
            #make a dummy enterprise parameter which we will use for getting the proposal density at the value of the old amplitude
            sampling_parameter = parameter.Uniform(rn_log_amp_range[0], rn_log_amp_range[1])('dummy')

            #set amplitude (and gamma if varied) to zero
            if vary_rn_gamma:
                new_point[1+max_n_source*7+num_per_psr_params] = 0.0 #gamma
                new_point[1+max_n_source*7+num_per_psr_params+1] = 0.0 #amplitude
            else:
                new_point[1+max_n_source*7+num_per_psr_params] = 0.0

            new_point_stripped = strip_samples(new_point,n_source,0,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
            samples_current_stripped = strip_samples(samples_current,n_source,1,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)            

            log_L = ptas[n_source][gwb_on][0].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[n_source][gwb_on][0].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][gwb_on][1].get_lnprior(samples_current_stripped)

            amp_proposal_old = sampling_parameter.get_pdf(old_log_amp)
            acc_ratio = np.exp(log_acc_ratio)*amp_proposal_old

            if vary_rn_gamma:
                gamma_proposal_old = ptas[n_source][gwb_on][1].params[n_source*7+num_per_psr_params].get_pdf(old_gamma)
                acc_ratio *= gamma_proposal_old
    
            #apply on/off prior
            acc_ratio *= rn_gwb_on_prior[gwb_on,0]/rn_gwb_on_prior[gwb_on,1]

            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,:] = np.copy(new_point)
                a_yes[2,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[2,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
        #turning on ----------------------------------------------------------------------------------------------------------
        else:
            samples_current = np.copy(samples[j,i,:])
            new_point = np.copy(samples[j,i,:])
            
            #draw new amplitude
            #make a dummy enterprise parameter which we will use for drawing from a log-uniform A distribution
            sampling_parameter = parameter.Uniform(rn_log_amp_range[0], rn_log_amp_range[1])('dummy')
            new_log_amp = sampling_parameter.sample()
            
            #draw new gamma if varied
            if vary_rn_gamma:
                new_gamma = ptas[n_source][gwb_on][1].params[n_source*7+num_per_psr_params].sample()

            #put in new parameters
            if vary_rn_gamma:
                new_point[1+max_n_source*7+num_per_psr_params+1] = new_log_amp
                new_point[1+max_n_source*7+num_per_psr_params] = new_gamma
            else:
                new_point[1+max_n_source*7+num_per_psr_params] = new_log_amp

            new_point_stripped = strip_samples(new_point,n_source,1,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
            samples_current_stripped = strip_samples(samples_current,n_source,0,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)

            log_L = ptas[n_source][gwb_on][1].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[n_source][gwb_on][1].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][gwb_on][0].get_lnprior(samples_current_stripped)

            amp_proposal_new = sampling_parameter.get_pdf(new_log_amp)
            acc_ratio = np.exp(log_acc_ratio)/amp_proposal_new

            if vary_rn_gamma:
                gamma_proposal_new = ptas[n_source][gwb_on][1].params[n_source*7+num_per_psr_params].get_pdf(new_gamma)
                acc_ratio *= 1/gamma_proposal_new

            #apply on/off prior
            acc_ratio *= rn_gwb_on_prior[gwb_on,1]/rn_gwb_on_prior[gwb_on,0]
            
            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,:] = np.copy(new_point)
                a_yes[2,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[2,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]

################################################################################
#
#GWB SWITCH (ON/OFF) MOVE
#
################################################################################
def gwb_switch_move(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, vary_white_noise, include_gwb, num_params, num_noise_params, num_per_psr_params, rn_gwb_on_prior, gwb_log_amp_range, vary_gwb_gamma, log_likelihood):
    #print("GWB switch")
    if not include_gwb:
       raise Exception("include_gwb must be True to use this move")
    for j in range(n_chain):
        n_source = get_n_source(samples,j,i)
        gwb_on = get_gwb_on(samples,j,i,max_n_source,num_noise_params)
        rn_on = get_rn_on(samples,j,i,max_n_source,num_per_psr_params)
        
        #turning off ---------------------------------------------------------------------------------------------------------
        if gwb_on==1:
            samples_current = np.copy(samples[j,i,:])
            new_point = np.copy(samples[j,i,:])
            if vary_gwb_gamma:
                old_gamma = np.copy(samples_current[1+max_n_source*7+num_noise_params])
                old_log_amp = np.copy(samples_current[1+max_n_source*7+num_noise_params+1])
            else:
                old_log_amp = np.copy(samples_current[1+max_n_source*7+num_noise_params])
            #make a dummy enterprise parameter which we will use for getting the proposal density at the value of the old amplitude
            sampling_parameter = parameter.Uniform(gwb_log_amp_range[0], gwb_log_amp_range[1])('dummy')

            #set amplitude (and gamma if varied) to zero
            if vary_gwb_gamma:
                new_point[1+max_n_source*7+num_noise_params] = 0.0
                new_point[1+max_n_source*7+num_noise_params+1] = 0.0
            else:
                new_point[1+max_n_source*7+num_noise_params] = 0.0

            new_point_stripped = strip_samples(new_point,n_source,rn_on,0,max_n_source,num_per_psr_params,num_noise_params,num_params)
            samples_current_stripped = strip_samples(samples_current,n_source,rn_on,1,max_n_source,num_per_psr_params,num_noise_params,num_params)

            log_L = ptas[n_source][0][rn_on].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[n_source][0][rn_on].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][1][rn_on].get_lnprior(samples_current_stripped)

            acc_ratio = np.exp(log_acc_ratio)*sampling_parameter.get_pdf(old_log_amp)

            if vary_gwb_gamma:
                gamma_proposal_old = ptas[n_source][1][1].params[n_source*7+num_noise_params].get_pdf(old_gamma)
                acc_ratio *= gamma_proposal_old

            #apply on/off prior
            acc_ratio *= rn_gwb_on_prior[0,rn_on]/rn_gwb_on_prior[1,rn_on]
            
            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,:] = np.copy(new_point)
                a_yes[1,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[1,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]
        #turning on ----------------------------------------------------------------------------------------------------------
        else:
            #if j==0: print("Turn on")
            samples_current = np.copy(samples[j,i,:])
            new_point = np.copy(samples[j,i,:])
            
            #draw new amplitude
            #make a dummy enterprise parameter which we will use for drawing from a log-uniform A distribution
            sampling_parameter = parameter.Uniform(gwb_log_amp_range[0], gwb_log_amp_range[1])('dummy')
            new_log_amp = sampling_parameter.sample()

            #draw new gamma if needed
            if vary_gwb_gamma:
                new_gamma = ptas[n_source][1][1].params[n_source*7+num_noise_params].sample()

            #put in new parameters
            if vary_gwb_gamma:
                new_point[1+max_n_source*7+num_noise_params] = new_gamma
                new_point[1+max_n_source*7+num_noise_params+1] = new_log_amp
            else:
                new_point[1+max_n_source*7+num_noise_params] = new_log_amp

            new_point_stripped = strip_samples(new_point,n_source,rn_on,1,max_n_source,num_per_psr_params,num_noise_params,num_params)
            samples_current_stripped = strip_samples(samples_current,n_source,rn_on,0,max_n_source,num_per_psr_params,num_noise_params,num_params)

            log_L = ptas[n_source][1][rn_on].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[n_source][1][rn_on].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][0][rn_on].get_lnprior(samples_current_stripped)

            acc_ratio = np.exp(log_acc_ratio)/sampling_parameter.get_pdf(new_log_amp)
            
            if vary_gwb_gamma:
                gamma_proposal_new = ptas[n_source][1][1].params[n_source*7+num_noise_params].get_pdf(new_gamma)
                acc_ratio *= 1/gamma_proposal_new

            #apply on/off prior
            acc_ratio *= rn_gwb_on_prior[1,rn_on]/rn_gwb_on_prior[0,rn_on]
            
            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,:] = np.copy(new_point)
                a_yes[1,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[1,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]

################################################################################
#
#REVERSIBLE-JUMP (RJ, aka TRANS-DIMENSIONAL) MOVE
#
################################################################################
def do_rj_move(n_chain, max_n_source, n_source_prior, ptas, samples, i, betas, a_yes, a_no, freqs, fe, inc_max, psi_max, phase0_max, h_max, rj_record, vary_white_noise, include_gwb, num_params, num_noise_params, num_per_psr_params, log_likelihood):
    #print("RJ")
    for j in range(n_chain):
        n_source = get_n_source(samples,j,i)

        if include_gwb:
            gwb_on = get_gwb_on(samples,j,i,max_n_source,num_noise_params)
        else:
            gwb_on = 0

        rn_on = get_rn_on(samples,j,i,max_n_source,num_per_psr_params)
        
        add_prob = 0.5 #flat prior on n_source-->same propability of addind and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()
        if n_source==0 or (direction_decide<add_prob and n_source!=max_n_source): #adding a signal------------------------------------------------------
            if j==0: rj_record.append(1)
            #alpha = 0.1
 
            #set limit used for rejection sampling below
            fe_limit = np.max(fe)
            #if the max is too high, cap it at Fe=200 (Neil's trick to not to be too restrictive)
            #if fe_limit>200:
            #    fe_limit=200
            
            log_f_max = float(ptas[1][gwb_on][1].params[3]._typename.split('=')[2][:-1])
            log_f_min = float(ptas[1][gwb_on][1].params[3]._typename.split('=')[1].split(',')[0])

            accepted = False
            while accepted==False:
                log_f_new = np.random.uniform(low=log_f_min, high=log_f_max)
                f_idx = (np.abs(np.log10(freqs) - log_f_new)).argmin()

                gw_theta = np.arccos(np.random.uniform(low=-1.0, high=1.0))
                gw_phi = np.random.uniform(low=0.0, high=2*np.pi)
                hp_idx = hp.ang2pix(hp.get_nside(fe), gw_theta, gw_phi)

                fe_new_point = fe[f_idx, hp_idx]
                if np.random.uniform()<(fe_new_point/fe_limit):
                    accepted = True

            cos_inc = np.random.uniform(low=-1.0, high=1.0)
            psi = np.random.uniform(low=0.0, high=np.pi)
            phase0 = np.random.uniform(low=0.0, high=2*np.pi)
            log10_h = ptas[-1][gwb_on][1].params[4].sample()
            
            prior_ext = (ptas[-1][gwb_on][1].params[1].get_pdf(cos_inc) * ptas[-1][gwb_on][1].params[6].get_pdf(psi) *
                         ptas[-1][gwb_on][1].params[5].get_pdf(phase0) * ptas[-1][gwb_on][1].params[4].get_pdf(log10_h))
            
            #cos_inc = np.cos(inc_max[f_idx, hp_idx]) + 2*alpha*(np.random.uniform()-0.5)
            #psi = psi_max[f_idx, hp_idx] + 2*alpha*(np.random.uniform()-0.5)
            #phase0 = phase0_max[f_idx, hp_idx] + 2*alpha*(np.random.uniform()-0.5)
            #log10_h = np.log10(h_max[f_idx, hp_idx]) + 2*alpha*(np.random.uniform()-0.5)

            samples_current = np.copy(samples[j,i,:])
            
            new_point = np.copy(samples[j,i,:])
            new_source = np.array([np.cos(gw_theta), cos_inc, gw_phi, log_f_new, log10_h, phase0, psi])
            new_point[1+n_source*7:1+(n_source+1)*7] = np.copy(new_source)
            new_point[0] = n_source+1
            
            new_point_stripped = strip_samples(new_point,n_source+1,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
            samples_current_stripped = strip_samples(samples_current,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)

            log_L = ptas[(n_source+1)][gwb_on][rn_on].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[(n_source+1)][gwb_on][rn_on].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][gwb_on][rn_on].get_lnprior(samples_current_stripped)

            healpy_pixel_area = hp.nside2pixarea(hp.get_nside(fe))
            log10f_resolution = np.diff(np.log10(freqs))[0]
            #first sum over sky location only
            fe_f = np.sum(fe, axis=1)
            #than sum over frequency, but only add half of the first and last bin, because their bin size is half of the others
            fe_sum = np.sum(fe_f[1:-1])+0.5*(fe_f[0] + fe_f[-1])
            #then the normalization constant is
            norm = fe_sum*healpy_pixel_area*log10f_resolution

            #normalization
            fe_new_point_normalized = fe_new_point/norm

            acc_ratio = np.exp(log_acc_ratio)/prior_ext/fe_new_point_normalized
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_source==0:
                acc_ratio *= 0.5
            if n_source==max_n_source-1:
                acc_ratio *= 2.0
            #accounting for n_source prior
            acc_ratio *= n_source_prior[int(n_source)+1]/n_source_prior[int(n_source)]
            
            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,:] = np.copy(new_point)
                a_yes[0,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]

        elif n_source==max_n_source or (direction_decide>add_prob and n_source!=0):   #removing a signal----------------------------------------------------------
            if j==0: rj_record.append(-1)
            #choose which source to remove
            remove_index = np.random.randint(n_source)
            
            samples_current = np.copy(samples[j,i,:])
            samples_current_stripped = strip_samples(samples_current,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
            new_point_stripped = np.delete(samples_current_stripped, range(remove_index*7,(remove_index+1)*7))
            
            log_L = ptas[(n_source-1)][gwb_on][rn_on].get_lnlikelihood(new_point_stripped)
            log_acc_ratio = log_L*betas[j,i]
            log_acc_ratio += ptas[(n_source-1)][gwb_on][rn_on].get_lnprior(new_point_stripped)
            log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
            log_acc_ratio += -ptas[n_source][gwb_on][rn_on].get_lnprior(samples_current_stripped)
            
            log_f_old = samples[j,i,1+remove_index*7+3]
            f_idx_old = (np.abs(np.log10(freqs) - log_f_old)).argmin()

            gw_theta_old = np.arccos(samples[j,i,1+remove_index*7+0])
            gw_phi_old = samples[j,i,1+remove_index*7+2]
            hp_idx_old = hp.ang2pix(hp.get_nside(fe), gw_theta_old, gw_phi_old)

            fe_old_point = fe[f_idx_old, hp_idx_old]
            healpy_pixel_area = hp.nside2pixarea(hp.get_nside(fe))
            log10f_resolution = np.diff(np.log10(freqs))[0]
            #first sum over sky location only
            fe_f = np.sum(fe, axis=1)
            #than sum over frequency, but only add half of the first and last bin, because their bin size is half of the others
            fe_sum = np.sum(fe_f[1:-1])+0.5*(fe_f[0] + fe_f[-1])
            #then the normalization constant is
            norm = fe_sum*healpy_pixel_area*log10f_resolution
            
            #normalization
            fe_old_point_normalized = fe_old_point/norm

            cos_inc = samples[j,i,1+remove_index*7+1]
            psi = samples[j,i,1+remove_index*7+6]
            phase0 = samples[j,i,1+remove_index*7+5]
            log10_h = samples[j,i,1+remove_index*7+4]

            prior_ext = (ptas[-1][gwb_on][1].params[1].get_pdf(cos_inc) * ptas[-1][gwb_on][1].params[6].get_pdf(psi) *
                         ptas[-1][gwb_on][1].params[5].get_pdf(phase0) * ptas[-1][gwb_on][1].params[4].get_pdf(log10_h))

            acc_ratio = np.exp(log_acc_ratio)*fe_old_point_normalized*prior_ext
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_source==1:
                acc_ratio *= 2.0
            if n_source==max_n_source:
                acc_ratio *= 0.5
            #accounting for n_source prior
            acc_ratio *= n_source_prior[int(n_source)-1]/n_source_prior[int(n_source)]
            
            if np.random.uniform()<=acc_ratio:
                samples[j,i+1,0] = n_source-1
                samples[j,i+1,1:(n_source-1)*7+1] = new_point_stripped[:(n_source-1)*7]
                samples[j,i+1,max_n_source*7+1:] = samples_current[1+max_n_source*7:]
                a_yes[0,j] += 1
                log_likelihood[j,i+1] = log_L
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0,j] += 1
                log_likelihood[j,i+1] = log_likelihood[j,i]

################################################################################
#
#GLOBAL PROPOSAL BASED ON FE-STATISTIC
#
################################################################################
def do_fe_global_jump(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, freqs, fe, inc_max, psi_max, phase0_max, h_max, vary_white_noise, include_gwb, num_params, num_noise_params, num_per_psr_params, Fe_pdet, Fe_alpha, psi_pdf, cos_inc_pdf, phase0_pdf, log_h_pdf, log_likelihood):
    #print("Fe")
       
    #set probability of deterministic vs flat proposal in extrinsic parameters
    p_det = Fe_pdet
    #set width of deterministic proposal
    alpha = Fe_alpha

    #set limit used for rejection sampling below
    fe_limit = np.max(fe)
    #if the max is too high, cap it at Fe=200 (Neil's trick to not to be too restrictive)
    #if fe_limit>200:
    #    fe_limit=200

    for j in range(n_chain):
        #check if there's any source -- stay at given point of not
        n_source = get_n_source(samples,j,i)
        if n_source==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[5,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("No source to vary!")
            continue

        if include_gwb:
            gwb_on = get_gwb_on(samples,j,i,max_n_source,num_noise_params)
        else:
            gwb_on = 0

        rn_on = get_rn_on(samples,j,i,max_n_source,num_per_psr_params)

        log_f_max = float(ptas[n_source][gwb_on][1].params[3]._typename.split('=')[2][:-1])
        log_f_min = float(ptas[n_source][gwb_on][1].params[3]._typename.split('=')[1].split(',')[0])

        accepted = False
        while accepted==False:
            log_f_new = np.random.uniform(low=log_f_min, high=log_f_max)
            f_idx = (np.abs(np.log10(freqs) - log_f_new)).argmin()

            gw_theta = np.arccos(np.random.uniform(low=-1.0, high=1.0))
            gw_phi = np.random.uniform(low=0.0, high=2*np.pi)
            hp_idx = hp.ang2pix(hp.get_nside(fe), gw_theta, gw_phi)

            fe_new_point = fe[f_idx, hp_idx]
            if np.random.uniform()<(fe_new_point/fe_limit):
                accepted = True

        if np.random.uniform()<p_det:
            deterministic=True
        else:
            deterministic=False

        if deterministic:
            cos_inc = np.cos(inc_max[f_idx, hp_idx]) + 2*alpha*(np.random.uniform()-0.5)
            psi = psi_max[f_idx, hp_idx] + 2*alpha*(np.random.uniform()-0.5)
            phase0 = phase0_max[f_idx, hp_idx] + 2*alpha*(np.random.uniform()-0.5)
            log10_h = np.log10(h_max[f_idx, hp_idx]) + 2*alpha*(np.random.uniform()-0.5)
        else:
            cos_inc = np.random.uniform(low=-1.0, high=1.0)
            psi = np.random.uniform(low=0.0, high=np.pi)
            phase0 = np.random.uniform(low=0.0, high=2*np.pi)
            log10_h = ptas[-1][gwb_on][1].params[4].sample()

        #choose randomly which source to change
        source_select = np.random.randint(n_source)
        samples_current = np.copy(samples[j,i,:])
        new_point = np.copy(samples[j,i,:])
        new_point[1+source_select*7:1+(source_select+1)*7] = np.array([np.cos(gw_theta), cos_inc, gw_phi, log_f_new,
                                                                               log10_h, phase0, psi])
        if fe_new_point>fe_limit:
            fe_new_point=fe_limit        
        
        new_point_stripped = strip_samples(new_point,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
        samples_current_stripped = strip_samples(samples_current,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)

        log_L = ptas[n_source][gwb_on][rn_on].get_lnlikelihood(new_point_stripped)
        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += ptas[n_source][gwb_on][rn_on].get_lnprior(new_point_stripped)
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -ptas[n_source][gwb_on][rn_on].get_lnprior(samples_current_stripped)

        #get ratio of proposal density for the Hastings ratio
        f_old = 10**samples[j,i,1+3+source_select*7]
        f_idx_old = (np.abs(freqs - f_old)).argmin()

        gw_theta_old = np.arccos(samples[j,i,1+source_select*7])
        gw_phi_old = samples[j,i,1+2+source_select*7]
        hp_idx_old = hp.ang2pix(hp.get_nside(fe), gw_theta_old, gw_phi_old)
        
        fe_old_point = fe[f_idx_old, hp_idx_old]
        if fe_old_point>fe_limit:
            fe_old_point = fe_limit

        cos_inc_old = np.cos(inc_max[f_idx_old, hp_idx_old])
        psi_old = psi_max[f_idx_old, hp_idx_old]
        phase0_old = phase0_max[f_idx_old, hp_idx_old]
        log10_h_old = np.log10(h_max[f_idx_old, hp_idx_old])
        
        old_params_fe = [cos_inc_old, log10_h_old, phase0_old, psi_old]
        
        new_params = [cos_inc, log10_h, phase0, psi]
        new_params_fe = [np.cos(inc_max[f_idx, hp_idx]), np.log10(h_max[f_idx, hp_idx]),
                        phase0_max[f_idx, hp_idx], psi_max[f_idx, hp_idx]]

        external_pdfs = [cos_inc_pdf, log_h_pdf, phase0_pdf, psi_pdf]
        
        hastings_extra_factor=1.0
        for k, old_param_fe, new_param, new_param_fe, ext_pdf in zip([1,4,5,6], old_params_fe, new_params, new_params_fe, external_pdfs):
            old_param = samples[j,i,1+k+source_select*7]
            #check if deterministic top-hat hits any boundaries
            #get prior boundaries
            upper_pb = float(ptas[n_source][gwb_on][1].params[k]._typename.split('=')[2][:-1])
            lower_pb = float(ptas[n_source][gwb_on][1].params[k]._typename.split('=')[1].split(',')[0])

            #for new params
            upper_diff_new = upper_pb - new_param_fe
            lower_diff_new = new_param_fe - lower_pb
            if np.abs(upper_diff_new)<alpha:
                prior_det_new = 1/(alpha+upper_diff_new)
            elif np.abs(lower_diff_new)<alpha:
                prior_det_new = 1/(alpha+lower_diff_new)
            else:
                prior_det_new = 1/(2*alpha)

            #for old params
            upper_diff_old = upper_pb - old_param_fe
            lower_diff_old = old_param_fe - lower_pb
            if np.abs(upper_diff_old)<alpha:
                prior_det_old = 1/(alpha+upper_diff_old)
            elif np.abs(lower_diff_old)<alpha:
                prior_det_old = 1/(alpha+lower_diff_old)
            else:
                prior_det_old = 1/(2*alpha)

            
            #True if the ith sample was at a place where we could jump with a deterministic jump
            #False otherwise            
            det_old = np.abs(old_param-old_param_fe)<alpha
            det_new = np.abs(new_param-new_param_fe)<alpha
            #get priors for old and new points
            prior_old = ptas[n_source][gwb_on][1].params[k].get_pdf(old_param)
            prior_new = ptas[n_source][gwb_on][1].params[k].get_pdf(new_param)
            
            #probability that it was put there as deterministic given that it's in a deterministic place
            if det_new and not det_old: #from non-det to det
                hastings_extra_factor *= prior_old / ( (1-p_det)*prior_new + p_det*prior_det_new*ext_pdf(new_param) ) 
            elif not det_new and det_old: #from det to non-det
                hastings_extra_factor *= ( (1-p_det)*prior_old + p_det*prior_det_old*ext_pdf(old_param) ) / prior_new
            elif det_new and det_old: #from det to det
                hastings_extra_factor *= ( (1-p_det)*prior_old + p_det*prior_det_old*ext_pdf(old_param) ) / ( (1-p_det)*prior_new + p_det*prior_det_new*ext_pdf(new_param) )
            elif not det_new and not det_old: #from non-det to non-det
                hastings_extra_factor *= prior_old / prior_new

        acc_ratio = np.exp(log_acc_ratio)*(fe_old_point/fe_new_point)*hastings_extra_factor
        
        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,:] = np.copy(new_point)
            a_yes[5,j]+=1
            log_likelihood[j,i+1] = log_L
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[5,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
    

################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN CW, GWB AND RN PARAMETERS)
#
################################################################################
def regular_jump(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, eig, eig_gwb_rn, include_gwb, num_params, num_noise_params, num_per_psr_params, vary_rn, vary_rn_gamma, vary_gwb_gamma, log_likelihood):
    #print("FISHER")
    for j in range(n_chain):
        n_source = get_n_source(samples,j,i)

        if include_gwb:
            gwb_on = get_gwb_on(samples,j,i,max_n_source,num_noise_params)
        else:
            gwb_on = 0

        rn_on = get_rn_on(samples,j,i,max_n_source,num_per_psr_params)

        samples_current = np.copy(samples[j,i,:])
        
        #decide if moving in CW source parameters or GWB/RN parameters
        #case #1: we can vary both
        if n_source!=0 and (gwb_on==1 or rn_on==1):
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'CW'
            else:
                what_to_vary = 'GWB'
        #case #2: we can only vary CW
        elif n_source!=0:
            what_to_vary = 'CW'
        #case #3: we can only vary GWB
        elif gwb_on==1 or rn_on==1:
            what_to_vary = 'GWB'
        #case #4: nothing to vary
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]
            #print("Nothing to vary!")
            continue
        
        if what_to_vary == 'CW':
            source_select = np.random.randint(n_source)
            jump_select = np.random.randint(7)
            jump_1source = eig[j,source_select,jump_select,:]
            jump = np.array([jump_1source[int(i-1-source_select*7)] if i>=1+source_select*7 and i<1+(source_select+1)*7 else 0.0 for i in range(samples_current.size)])
        elif what_to_vary == 'GWB':
            num_gwb_params = 1
            num_rn_params = 1
            if vary_rn_gamma:
                num_rn_params += 1
            if vary_gwb_gamma:
                num_gwb_params += 1
            jump_select = np.random.randint(num_rn_params+num_gwb_params)
            jump_gwb = eig_gwb_rn[j,jump_select,:]
            if gwb_on==1 and rn_on ==1: #both gwb and rn are on
                jump = np.array([jump_gwb[int(i-1-max_n_source*7-num_per_psr_params)] if i>=1+max_n_source*7+num_per_psr_params and i<1+max_n_source*7+num_noise_params+num_gwb_params else 0.0 for i in range(samples_current.size)])
            elif rn_on==1: #only rn is on
                jump_gwb[-1] = 0
                if vary_gwb_gamma:
                    jump_gwb[-2] = 0
                jump = np.array([jump_gwb[int(i-1-max_n_source*7-num_per_psr_params)] if i>=1+max_n_source*7+num_per_psr_params and i<1+max_n_source*7+num_noise_params else 0.0 for i in range(samples_current.size)])
            else: #only gwb is on
                jump_gwb[0] = 0
                jump_gwb[1] = 0
                jump = np.array([jump_gwb[int(i-1-max_n_source*7-num_per_psr_params)] if i>=1+max_n_source*7+num_per_psr_params and i<1+max_n_source*7+num_noise_params+num_gwb_params else 0.0 for i in range(samples_current.size)])

        new_point = samples_current + jump*np.random.normal()

        new_point_stripped = strip_samples(new_point,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
        samples_current_stripped = strip_samples(samples_current,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)

        log_L = ptas[n_source][gwb_on][rn_on].get_lnlikelihood(new_point_stripped)
        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += ptas[n_source][gwb_on][rn_on].get_lnprior(new_point_stripped)
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -ptas[n_source][gwb_on][rn_on].get_lnprior(samples_current_stripped)

        acc_ratio = np.exp(log_acc_ratio)
        
        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,:] = np.copy(new_point)
            a_yes[6,j]+=1
            log_likelihood[j,i+1] = log_L
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[6,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]

################################################################################
#
#NOISE MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN WHITE NOISE PARAMETERS)
#
################################################################################
def noise_jump(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, eig_per_psr, include_gwb, num_params, num_noise_params, num_per_psr_params, vary_white_noise, log_likelihood):
    #print("NOISE")
    for j in range(n_chain):
        n_source = get_n_source(samples,j,i)

        if include_gwb:
            gwb_on = get_gwb_on(samples,j,i,max_n_source,num_noise_params)
        else:
            gwb_on = 0

        rn_on = get_rn_on(samples,j,i,max_n_source,num_per_psr_params)

        samples_current = np.copy(samples[j,i,:])
        
        #do the wn jump
        jump_select = np.random.randint(eig_per_psr.shape[1])
        jump_wn = eig_per_psr[j,jump_select,:]
        jump = np.array([jump_wn[int(i-1-max_n_source*7)] if i>=1+max_n_source*7 and i<1+max_n_source*7+eig_per_psr.shape[1] else 0.0 for i in range(samples_current.size)])

        new_point = samples_current + jump*np.random.normal()

        new_point_stripped = strip_samples(new_point,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)
        samples_current_stripped = strip_samples(samples_current,n_source,rn_on,gwb_on,max_n_source,num_per_psr_params,num_noise_params,num_params)

        log_L = ptas[n_source][gwb_on][rn_on].get_lnlikelihood(new_point_stripped)
        log_acc_ratio = log_L*betas[j,i]
        log_acc_ratio += ptas[n_source][gwb_on][rn_on].get_lnprior(new_point_stripped)
        log_acc_ratio += -log_likelihood[j,i]*betas[j,i]
        log_acc_ratio += -ptas[n_source][gwb_on][rn_on].get_lnprior(samples_current_stripped)

        acc_ratio = np.exp(log_acc_ratio)
        
        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,:] = np.copy(new_point)
            a_yes[7,j]+=1
            log_likelihood[j,i+1] = log_L
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[7,j]+=1
            log_likelihood[j,i+1] = log_likelihood[j,i]


################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, max_n_source, ptas, samples, i, betas, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_params, num_noise_params, num_per_psr_params, log_likelihood, PT_hist, PT_hist_idx):
    #print("PT")
    
    #set up map to help keep track of swaps
    swap_map = list(range(n_chain))

    #get log_Ls from all the chains
    log_Ls = []
    for j in range(n_chain):
        log_Ls.append(log_likelihood[j,i])

    #loop through and propose a swap at each chain (starting from hottest chain and going down in T) and keep track of results in swap_map
    for swap_chain in reversed(range(n_chain-1)):
        log_acc_ratio = -log_Ls[swap_map[swap_chain]] * betas[swap_chain,i]
        log_acc_ratio += -log_Ls[swap_map[swap_chain+1]] * betas[swap_chain+1,i]
        log_acc_ratio += log_Ls[swap_map[swap_chain+1]] * betas[swap_chain,i]
        log_acc_ratio += log_Ls[swap_map[swap_chain]] * betas[swap_chain+1,i]

        acc_ratio = np.exp(log_acc_ratio)
        PT_hist[swap_chain,PT_hist_idx[0]%PT_hist.shape[1]] = np.minimum(acc_ratio, 1.0)
        PT_hist_idx += 1
        if np.random.uniform()<=acc_ratio:
            swap_map[swap_chain], swap_map[swap_chain+1] = swap_map[swap_chain+1], swap_map[swap_chain]
            a_yes[4,swap_chain]+=1
            swap_record.append(swap_chain)
        else:
            a_no[4,swap_chain]+=1

    #loop through the chains and record the new samples and log_Ls
    for j in range(n_chain):
        samples[j,i+1,:] = samples[swap_map[j],i,:]
        log_likelihood[j,i+1] = log_likelihood[swap_map[j],i]

################################################################################
#
#FISHER EIGENVALUE CALCULATION
#
################################################################################
def get_fisher_eigenvectors(params, pta, T_chain=1, epsilon=1e-4, n_source=1, dim=7, offset=0, use_prior=False):
    fisher = np.zeros((n_source,dim,dim))
    eig = []

    #print(params)

    #lnlikelihood at specified point
    if use_prior:
        nn = pta.get_lnlikelihood(params) + pta.get_lnprior(params)
    else:
        nn = pta.get_lnlikelihood(params)
    
    
    for k in range(n_source):
        #print(k)
        #calculate diagonal elements
        for i in range(dim):
            #create parameter vectors with +-epsilon in the ith component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            paramsPP[offset+i+k*dim] += 2*epsilon
            paramsMM[offset+i+k*dim] -= 2*epsilon
            #print(paramsPP)
            
            #lnlikelihood at +-epsilon positions
            if use_prior:
                pp = pta.get_lnlikelihood(paramsPP) + pta.get_lnprior(paramsPP)
                mm = pta.get_lnlikelihood(paramsMM) + pta.get_lnprior(paramsMM)
            else:
                pp = pta.get_lnlikelihood(paramsPP)
                mm = pta.get_lnlikelihood(paramsMM)

            #print(pp, nn, mm)
            
            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            #print('diagonal')
            #print(pp,nn,mm)
            #print(-(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon))
            fisher[k,i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

        #calculate off-diagonal elements
        for i in range(dim):
            for j in range(i+1,dim):
                #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
                paramsPP = np.copy(params)
                paramsMM = np.copy(params)
                paramsPM = np.copy(params)
                paramsMP = np.copy(params)

                paramsPP[offset+i+k*dim] += epsilon
                paramsPP[offset+j+k*dim] += epsilon
                paramsMM[offset+i+k*dim] -= epsilon
                paramsMM[offset+j+k*dim] -= epsilon
                paramsPM[offset+i+k*dim] += epsilon
                paramsPM[offset+j+k*dim] -= epsilon
                paramsMP[offset+i+k*dim] -= epsilon
                paramsMP[offset+j+k*dim] += epsilon

                #lnlikelihood at those positions
                if use_prior:
                    pp = pta.get_lnlikelihood(paramsPP) + pta.get_lnprior(paramsPP)
                    mm = pta.get_lnlikelihood(paramsMM) + pta.get_lnprior(paramsMM)
                    pm = pta.get_lnlikelihood(paramsPM) + pta.get_lnprior(paramsPM)
                    mp = pta.get_lnlikelihood(paramsMP) + pta.get_lnprior(paramsMP)
                else:
                    pp = pta.get_lnlikelihood(paramsPP)
                    mm = pta.get_lnlikelihood(paramsMM)
                    pm = pta.get_lnlikelihood(paramsPM)
                    mp = pta.get_lnlikelihood(paramsMP)

                #calculate off-diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
                #print('off-diagonal')
                #print(pp,mp,pm,mm)
                #print(-(pp - mp - pm + mm)/(4.0*epsilon*epsilon))
                fisher[k,i,j] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
                fisher[k,j,i] = -(pp - mp - pm + mm)/(4.0*epsilon*epsilon)
        
        #print(fisher)
        #correct for the given temperature of the chain    
        fisher = fisher/T_chain
      
        try:
            #Filter nans and infs and replace them with 1s
            #this will imply that we will set the eigenvalue to 100 a few lines below
            FISHER = np.where(np.isfinite(fisher[k,:,:]), fisher[k,:,:], 1.0)
            if not np.array_equal(FISHER, fisher[k,:,:]):
                print("Changed some nan elements in the Fisher matrix to 1.0")

            #Find eigenvalues and eigenvectors of the Fisher matrix
            w, v = np.linalg.eig(FISHER)

            #filter w for eigenvalues smaller than 100 and set those to 100 -- Neil's trick
            eig_limit = 1.0

            W = np.where(np.abs(w)>eig_limit, w, eig_limit)
            #print(W)
            #print(np.sum(v**2, axis=0))
            #if T_chain==1.0: print(W)
            #if T_chain==1.0: print(v)

            eig.append( (np.sqrt(1.0/np.abs(W))*v).T )
            #print(np.sum(eig**2, axis=1))
            #if T_chain==1.0: print(eig)

        except:
            print("An Error occured in the eigenvalue calculation")
            eig.append( np.array(False) )

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.imshow(np.log10(np.abs(np.real(np.array(FISHER)))))
        #plt.imshow(np.real(np.array(FISHER)))
        #plt.colorbar()
        
        #plt.figure()
        #plt.imshow(np.log10(np.abs(np.real(np.array(eig)[0,:,:]))))
        #plt.imshow(np.real(np.array(eig)[0,:,:]))
        #plt.colorbar()
    
    return np.array(eig)

################################################################################
#
#MAKE AN ARRAY CONTAINING GLOBAL PROPOSAL DENSITY FROM F_E-STATISTICS
#
################################################################################
def make_fe_global_proposal(fe_func, f_min=1e-9, f_max=1e-7, n_freq=400,
                            NSIDE=8, maximized_parameters=False):
    m = np.zeros((n_freq, hp.nside2npix(NSIDE)))
    if maximized_parameters:
        inc_max = np.zeros((n_freq, hp.nside2npix(NSIDE)))
        psi_max = np.zeros((n_freq, hp.nside2npix(NSIDE)))
        phase0_max = np.zeros((n_freq, hp.nside2npix(NSIDE)))
        h_max = np.zeros((n_freq, hp.nside2npix(NSIDE)))

    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)

    idx = np.arange(hp.nside2npix(NSIDE))
    for i, f in enumerate(freqs):
        print("{0}th freq out of {1}".format(i, n_freq))
        if maximized_parameters:
            m[i,:], inc_max[i,:], psi_max[i,:], phase0_max[i,:], h_max[i,:] = fe_func(f,
                                np.array(hp.pix2ang(NSIDE, idx)),
                                maximized_parameters=maximized_parameters)
        else:
            m[i,:] = fe_func(f, np.array(hp.pix2ang(NSIDE, idx)),
                             maximized_parameters=maximized_parameters)
    if maximized_parameters:
        return freqs, m, inc_max, psi_max, phase0_max, h_max
    else:
        return freqs, m

################################################################################
#
#MAKE PTA OBJECT FOR PRIOR RECOVERY
#
################################################################################
def get_prior_recovery_pta(pta):
    class prior_recovery_pta:
        def __init__(self, pta):
            self.pta = pta
            self.params = pta.params
            self.pulsars = pta.pulsars
        def get_lnlikelihood(self, x):
            return 0.0
        def get_lnprior(self, x):
            return self.pta.get_lnprior(x)
        
    return prior_recovery_pta(pta)

################################################################################
#
#POSTPROCESSING FOR TRANS-DIMENSIONAL RUNS
#
################################################################################
def transdim_postprocess(samples, pulsars=None, ptas=None, separation_method='match', f_tol=0.1, match_tol=0.5, chisq_tol=9.0, max_n_source=10, status_every=1000, include_gwb=False):
    N = samples.shape[0]
    
    if separation_method=='freq':
        freqs = []
        sample_dict = {}
        source_on_idxs = {}
        for i in range(N):
            if i%status_every==0:
                print('Progress: {0:2.2f}% '.format(i/N*100))
            for j,f in enumerate(samples[i,4:max_n_source*7:7]):
                if not np.isnan(f):
                    new = True
                    f_diff = []
                    for idx, freq in enumerate(freqs):
                        if np.abs(f-freq)<f_tol:
                            new = False
                            f_diff.append((np.abs(f-freq), idx))
                    if new:
                        freqs.append(f)
                        sample_dict[len(freqs)-1] = np.array([list(samples[i,1+7*j:1+7*(j+1)]),])
                        source_on_idxs[len(freqs)-1] = [i,]
                    else:
                        min_diff = np.inf
                        for diff, idx in f_diff:
                            if diff < min_diff:
                                min_diff = diff
                                freq_idx = idx
                        freqs[freq_idx] += (f - freqs[freq_idx]) / (len(source_on_idxs[freq_idx]) + 1)
                        sample_dict[freq_idx] = np.append(sample_dict[freq_idx], np.array([list(samples[i,1+7*j:1+7*(j+1)]),]), axis=0)
                        source_on_idxs[freq_idx].append(i)
        return sample_dict, source_on_idxs

    elif separation_method=="match-max-L":
        param_names = ['cos_gwtheta', 'cos_inc', 'gwphi', 'log10_fgw', 'log10_h', 'phase0', 'psi']
        psrlist = [p.name for p in pulsars]
        pta = get_ptas(pulsars,include_rn=False, include_gwb=False, vary_white_noise=False)[1][0]
        sample_dict = {}
        source_on_idxs = {}
        params_max_L = {}
        nparam_dict_max_L = {}
        n_found = 0
        found_all = False
        flagged_ij = []
        delta_L_dict = {} #dict to keep delta_L values so we don't need to calculate them every time we loop through the samples
        logL_dict = {}
        flagged_is = []
        while not found_all:
            print("Source #",n_found)
            max_logL = 0.0
            n_remaining = 0
            log_L = []
            #Find the sample with the highest likelihood
            for i in range(N):
                if i not in flagged_is:
                    if i%status_every==0:
                        print('Progress looking for next source: {0:2.2f}% '.format(i/N*100))
                    n_source = int(np.copy(samples[i,0]))
                    if n_source>0:
                        #print(n_source)
                        if include_gwb:
                            gwb_on = int(samples[i,-1]!=0.0)
                        else:
                            gwb_on = 0
                        samples_current = np.delete(samples[i,1:], range(n_source*7,max_n_source*7))
                        
                        if i in logL_dict:
                            logL = logL_dict[i]
                        else:
                            logL = ptas[n_source][gwb_on][1].get_lnlikelihood(samples_current)
                            logL_dict[i] = logL
            
                        if logL>max_logL:
                            max_logL = logL
                            maxL_i = i
            
            flagged_is.append(maxL_i)

            #Find the source within the highest likelihood sample that has the highest delta likelihood when the source is removed
            max_delta_L = 0.0
            n_source = int(samples[maxL_i, 0])
            if include_gwb:
                gwb_on = int(samples[maxL_i,-1]!=0.0)
            else:
                gwb_on = 0
            samples_current = np.delete(samples[maxL_i,1:], range(n_source*7,max_n_source*7))
            for remove_index in range(n_source):
                if (maxL_i, remove_index) not in flagged_ij:
                    n_remaining += 1
                    if (maxL_i, remove_index) in delta_L_dict:
                        delta_L = delta_L_dict[(maxL_i, remove_index)]
                    else:
                        new_point = np.delete(samples_current, range(remove_index*7,(remove_index+1)*7))
                        delta_L = ptas[n_source][gwb_on][1].get_lnlikelihood(samples_current) - ptas[(n_source-1)][gwb_on][1].get_lnlikelihood(new_point)
                        delta_L_dict[(maxL_i, remove_index)] = delta_L
                    if delta_L>max_delta_L:
                        max_delta_L = delta_L
                        max_L_ij = [maxL_i, remove_index]

            flagged_ij.append((max_L_ij[0],max_L_ij[1]))
            print("Remaining samples = ", n_remaining)
            print("Delta L = ",max_delta_L)
            print("SNR = ", np.sqrt(2*max_delta_L))
            print("n_source for max_L sample = ",samples[max_L_ij[0],0])
            print("gwb_on for max_L sample = ",int(samples[max_L_ij[0],-1]!=0.0))
            #if max_delta_L>8: #SNR>4
            #if max_delta_L>4.5: #SNR>3
            if max_logL>0.0:
                param_SNR = np.sqrt(get_similarity_matrix(pta, [{'0_'+pname:value for pname, value in zip(param_names, samples[max_L_ij[0],1+7*max_L_ij[1]:1+7*(max_L_ij[1]+1)])}, 
                                                                {'0_'+pname:value for pname, value in zip(param_names, samples[max_L_ij[0],1+7*max_L_ij[1]:1+7*(max_L_ij[1]+1)])}],
                                                          noise_param_dict={psr_name+"_efac":ef for psr_name,ef in zip(psrlist,samples[max_L_ij[0],1+7*max_n_source:1+7*max_n_source+len(psrlist)])})[0,0] )
                #check if SNR of suggested soure is higher than 5
                if param_SNR>5.0:
                    params_max_L[n_found] = {'0_'+pname:value for pname, value in zip(param_names, samples[max_L_ij[0],1+7*max_L_ij[1]:1+7*(max_L_ij[1]+1)])}
                    sample_dict[n_found] = np.array([list(samples[max_L_ij[0],1+7*max_L_ij[1]:1+7*(max_L_ij[1]+1)]),])
                    source_on_idxs[n_found] = [max_L_ij[0],]
                    nparam_dict_max_L[n_found] = {psr_name+"_efac":ef for psr_name,ef in zip(psrlist,samples[max_L_ij[0],1+7*max_n_source:1+7*max_n_source+len(psrlist)])}
                    #nparam_dict_max_L[n_found] = {psr_name+"_efac":1.0 for psr_name in psrlist}
                    print("Noise params: ", nparam_dict_max_L[n_found])

                    print("Params: ", params_max_L[n_found])
                    print("SNR from params = ", np.sqrt(get_similarity_matrix(pta, [params_max_L[n_found], params_max_L[n_found]], noise_param_dict=nparam_dict_max_L[n_found])[0,0] ) )

                    #find all samples corresponding to this source
                    n_added = 0
                    for i in range(N):
                        if i%status_every==0:
                            print('Progress finding all samples for source: {0:2.2f}% '.format(i/N*100))
                        for j in range(max_n_source):
                            if (i,j) not in flagged_ij:
                                params = samples[i,1+7*j:1+7*(j+1)]
                                if params[0]!=0.0:
                                    if np.abs(params[3]-params_max_L[n_found]['0_log10_fgw'])<f_tol:
                                        pp = {'0_'+pname:value for pname, value in zip(param_names, params)}
                                        #match_matrix = get_match_matrix(pta, [pp, params_max_L[n_found]], noise_param_dict=nparam_dict_max_L[n_found])
                                        #match = match_matrix[0,1]
                                        #if match > match_tol:
                                        inner_prod_matrix = get_similarity_matrix(pta, [pp, params_max_L[n_found]], noise_param_dict=nparam_dict_max_L[n_found])
                                        chi_squared = inner_prod_matrix[0,0] + inner_prod_matrix[1,1] - 2*inner_prod_matrix[0,1]
                                        #print(inner_prod_matrix[0,0] + inner_prod_matrix[1,1] - 2*inner_prod_matrix[0,1])
                                        #chi_squared = 2*param_SNR**2*(1-match_matrix[0,1])
                                        #print(chi_squared, params_max_L[n_found]['0_log10_fgw'], params[3])
                                        if chi_squared < chisq_tol:
                                            n_added += 1
                                            sample_dict[n_found] = np.append(sample_dict[n_found], np.array([list(params),]), axis=0)
                                            source_on_idxs[n_found].append(i)
                                            flagged_ij.append((i,j))
                    print("Samples added to source = ",n_added)
                    n_found += 1
                else:
                    print("Not enough SNR in found source:", param_SNR)
            else:
                found_all = True
        
        return sample_dict, source_on_idxs, params_max_L


    else:
        print("Not understood separation method: {0}".format(separation_method))
    
################################################################################
#
#MATCH CALCULATION ROUTINES
#
################################################################################
def get_similarity_matrix(pta, params_list, noise_param_dict=None):

    if noise_param_dict is None:
        print('No noise dictionary provided!...')
    else:
        pta.set_default_params(noise_param_dict)

    #print(pta.summary())

    phiinvs = pta.get_phiinv([], logdet=False)
    TNTs = pta.get_TNT([])
    Ts = pta.get_basis()
    Nvecs = pta.get_ndiag([])
    Nmats = [ Fe_statistic.make_Nmat(phiinv, TNT, Nvec, T) for phiinv, TNT, Nvec, T in zip(phiinvs, TNTs, Nvecs, Ts)]

    n_sources = len(params_list)

    res_model = [pta.get_delay(x) for x in params_list]

    S = np.zeros((n_sources,n_sources))
    for idx, (psr, Nmat, TNT, phiinv, T) in enumerate(zip(pta.pulsars, Nmats,
                                                          TNTs, phiinvs, Ts)):
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)
        
        for i in range(n_sources):
            for j in range(n_sources):
                delay_i = res_model[i][idx]
                delay_j = res_model[j][idx]
                #print(delay_i)
                #print(Nmat)
                #print(Nmat, T, Sigma)
                S[i,j] += Fe_statistic.innerProduct_rr(delay_i, delay_j, Nmat, T, Sigma)
    return S

def get_match_matrix(pta, params_list, noise_param_dict=None):
    S = get_similarity_matrix(pta, params_list, noise_param_dict=noise_param_dict)

    M = np.zeros(S.shape)
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            M[i,j] = S[i,j]/np.sqrt(S[i,i]*S[j,j])
    return M

################################################################################
#
#FUNCTION TO EASILY SET UP A LIST OF PTA OBJECTS
#
################################################################################
def get_ptas(pulsars, vary_white_noise=True, include_equad_ecorr=False, wn_backend_selection=False, noisedict_file=None, include_rn=True, include_per_psr_rn=False, vary_rn=True, vary_per_psr_rn=False, include_gwb=True, max_n_source=1, efac_start=1.0, rn_amp_prior='uniform', rn_log_amp_range=[-18,-11], per_psr_rn_amp_prior='uniform', per_psr_rn_log_amp_range=[-18,-11], rn_params=[-13.0,1.0], gwb_amp_prior='uniform', gwb_log_amp_range=[-18,-11], n_comp_common=30, n_comp_per_psr_rn=30, vary_gwb_gamma=True, vary_rn_gamma=True, cw_amp_prior='uniform', cw_log_amp_range=[-18,-11], cw_f_range=[3.5e-9,1e-7], include_psr_term=False, prior_recovery=False):
    #setting up base model
    if vary_white_noise:
        efac = parameter.Uniform(0.01, 10.0)
    else:
        efac = parameter.Constant(efac_start) 

    if include_equad_ecorr:
        equad = parameter.Constant()
        ecorr = parameter.Constant()

    if wn_backend_selection:
        selection = selections.Selection(selections.by_backend)
        ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
        if include_equad_ecorr:
            eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)
    else:
        ef = white_signals.MeasurementNoise(efac=efac)
        if include_equad_ecorr:
            eq = white_signals.EquadNoise(log10_equad=equad)
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr)

    #eq = white_signals.EquadNoise(log10_equad=equad)
    tm = gp_signals.TimingModel(use_svd=True)

    base_model = ef + tm
    if include_equad_ecorr:
        base_model = base_model + eq + ec

    if include_per_psr_rn:
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)

        if vary_per_psr_rn:
            if per_psr_rn_amp_prior == 'uniform':
                log10_A = parameter.LinearExp(per_psr_rn_log_amp_range[0], per_psr_rn_log_amp_range[1])
            elif per_psr_rn_amp_prior == 'log-uniform':
                log10_A = parameter.Uniform(per_psr_rn_log_amp_range[0], per_psr_rn_log_amp_range[1])

            gamma = parameter.Uniform(0, 7)
        else:
            log10_A = parameter.Constant()
            gamma = parameter.Constant()
        
        pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        per_psr_rn = gp_signals.FourierBasisGP(pl, components=n_comp_per_psr_rn, Tspan=Tspan)
        
        base_model = base_model + per_psr_rn
    
    #adding red noise if included
    if include_rn:
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)
        
        if vary_rn:
            #rn = ext_models.common_red_noise_block(prior='uniform', Tspan=Tspan, name='com_rn')
            amp_name = 'com_rn_log10_A'
            if rn_amp_prior == 'uniform':
                log10_Arn = parameter.LinearExp(rn_log_amp_range[0], rn_log_amp_range[1])(amp_name)
            elif rn_amp_prior == 'log-uniform':
                log10_Arn = parameter.Uniform(rn_log_amp_range[0], rn_log_amp_range[1])(amp_name)
            gam_name = 'com_rn_gamma'
            if vary_rn_gamma:
                gamma_rn = parameter.Uniform(0, 7)(gam_name)
            else:
                gamma_val = 13.0/3
                gamma_rn = parameter.Constant(gamma_val)(gam_name)
            pl = utils.powerlaw(log10_A=log10_Arn, gamma=gamma_rn)
            rn = gp_signals.FourierBasisGP(spectrum=pl, coefficients=False, components=n_comp_common, Tspan=Tspan,
                                           modes=None, name='com_rn')
        else:
            log10_A = parameter.Constant(rn_params[0])
            gamma = parameter.Constant(rn_params[1])
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=n_comp_common, Tspan=Tspan)
        
        #base_model += rn 

    #make base models including GWB
    if include_gwb:
        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)
        amp_name = 'gw_log10_A'
        if gwb_amp_prior == 'uniform':
            log10_Agw = parameter.LinearExp(gwb_log_amp_range[0], gwb_log_amp_range[1])(amp_name)
        elif gwb_amp_prior == 'log-uniform':
            log10_Agw = parameter.Uniform(gwb_log_amp_range[0], gwb_log_amp_range[1])(amp_name)
        
        gam_name = 'gw_gamma'
        if vary_gwb_gamma:
            gamma_gw = parameter.Uniform(0, 7)(gam_name)
        else:
            gamma_val = 13.0/3
            gamma_gw = parameter.Constant(gamma_val)(gam_name)

        cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        gwb = gp_signals.FourierBasisCommonGP(cpl, utils.hd_orf(), coefficients=False,
                                              components=n_comp_common, Tspan=Tspan,
                                              modes=None, name='gw')

        #base_model_gwb = base_model + gwb

    #setting up the pta object
    cws = []
    for i in range(max_n_source):
        log10_fgw = parameter.Uniform(np.log10(cw_f_range[0]), np.log10(cw_f_range[1]))(str(i)+'_'+'log10_fgw')
        log10_mc = parameter.Constant(np.log10(5e9))(str(i)+'_'+'log10_mc')
        cos_gwtheta = parameter.Uniform(-1, 1)(str(i)+'_'+'cos_gwtheta')
        gwphi = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'gwphi')
        phase0 = parameter.Uniform(0, 2*np.pi)(str(i)+'_'+'phase0')
        psi = parameter.Uniform(0, np.pi)(str(i)+'_'+'psi')
        cos_inc = parameter.Uniform(-1, 1)(str(i)+'_'+'cos_inc')
        if cw_amp_prior == 'log-uniform':
            log10_h = parameter.Uniform(cw_log_amp_range[0], cw_log_amp_range[1])(str(i)+'_'+'log10_h')
        elif cw_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(cw_log_amp_range[0], cw_log_amp_range[1])(str(i)+'_'+'log10_h')
        else:
            print("CW amplitude prior of {0} not available".format(cw_amp_prior))
        cw_wf = deterministic.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
                     log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0,
                     psi=psi, cos_inc=cos_inc, tref=53000*86400)
        cws.append(deterministic.CWSignal(cw_wf, psrTerm=include_psr_term, name='cw'+str(i)))
    
    gwb_options = [False,]
    if include_gwb:
        gwb_options.append(True)
    
    rn_options = [False,]
    if include_rn:
        rn_options.append(True)

    ptas = []
    for n_source in range(max_n_source+1):
        gwb_sub_ptas = []
        for gwb_o in gwb_options:
            rn_sub_ptas = []
            for rn_o in rn_options:
                #setting up the proper model
                s = base_model
                if gwb_o:
                    s += gwb
                if rn_o:
                    s += rn
                for i in range(n_source):
                    s = s + cws[i]

                model = []
                for p in pulsars:
                    model.append(s(p))
                
                #set the likelihood to unity if we are in prior recovery mode
                if prior_recovery:
                    rn_sub_ptas.append(get_prior_recovery_pta(signal_base.PTA(model)))
                elif noisedict_file is not None:
                    with open(noisedict_file, 'r') as fp:
                        noisedict = json.load(fp)
                    pta = signal_base.PTA(model)
                    pta.set_default_params(noisedict)
                    rn_sub_ptas.append(pta)
                else:
                    rn_sub_ptas.append(signal_base.PTA(model))

            gwb_sub_ptas.append(rn_sub_ptas)

        ptas.append(gwb_sub_ptas)

    return ptas

################################################################################
#
#SOME HELPER FUNCTIONS
#
################################################################################
def get_gwb_on(samples, j, i, max_n_source, num_noise_params):
    return int(samples[j,i,max_n_source*7+1+num_noise_params]!=0.0)

def get_rn_on(samples, j, i, max_n_source, num_per_psr_params):
    return int(samples[j,i,max_n_source*7+1+num_per_psr_params]!=0.0)

def get_n_source(samples, j, i):
    return int(samples[j,i,0])

def strip_samples(sample, n_source, rn_on, gwb_on, max_n_source, num_per_psr_params, num_noise_params, num_params):
    if rn_on==1 and gwb_on==1:
        return np.delete(sample[1:], range(n_source*7,max_n_source*7))
    elif rn_on==1 and gwb_on==0:
        return np.delete(sample[1:], list(range(n_source*7,max_n_source*7))+list(range(max_n_source*7+num_noise_params,num_params-1)))
    elif rn_on==0 and gwb_on==1:
        return np.delete(sample[1:], list(range(n_source*7,max_n_source*7))+list(range(max_n_source*7+num_per_psr_params,max_n_source*7+num_noise_params)))
    elif rn_on==0 and gwb_on==0:
        return np.delete(sample[1:], list(range(n_source*7,max_n_source*7))+list(range(max_n_source*7+num_per_psr_params,num_params-1)))
