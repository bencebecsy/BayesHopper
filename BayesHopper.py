################################################################################
#
#BayesHopper -- **Bayes**ian **H**yp**o**thesis testing for **p**ulsar timing arrays with **per**iodic signals
#
#Bence Bécsy (bencebecsy@montana.edu) -- 2019
################################################################################

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import enterprise
import enterprise.signals.parameter as parameter
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import utils

import enterprise_cw_funcs_from_git as models
from enterprise_extensions import models as ext_models

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################

def run_ptmcmc(N, T_max, n_chain, pulsars, max_n_source=1, n_source_prior='flat', n_source_start='random', RJ_weight=0,
               regular_weight=3, noise_jump_weight=3, PT_swap_weight=1, T_ladder = None,
               Fe_proposal_weight=0, fe_file=None, Fe_pdet=0.5, Fe_alpha=0.1, draw_from_prior_weight=0,
               de_weight=0, prior_recovery=False, cw_amp_prior='uniform', gwb_amp_prior='uniform', rn_amp_prior='uniform',
               gwb_log_amp_range=[-18,-11], rn_log_amp_range=[-18,-11], cw_log_amp_range=[-18,-11],
               vary_white_noise=False, efac_start=1.0,
               include_gwb=False, gwb_switch_weight=0, include_psr_term=False,
               include_rn=False, vary_rn=False, rn_params=[-13.0,1.0], jupyter_notebook=False,
               gwb_on_prior=0.5):

    #setting up base model
    if vary_white_noise:
        efac = parameter.Uniform(0.01, 10.0)
        #equad = parameter.Uniform(-8.5, -5)
    else:
        efac = parameter.Constant(efac_start) 
        #equad = parameter.Constant(wn_params[1])
    
    ef = white_signals.MeasurementNoise(efac=efac)
    #eq = white_signals.EquadNoise(log10_equad=equad)
    tm = gp_signals.TimingModel(use_svd=True)

    base_model = ef + tm
    
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
            gamma_rn = parameter.Uniform(0, 7)(gam_name)
            pl = utils.powerlaw(log10_A=log10_Arn, gamma=gamma_rn)
            rn = gp_signals.FourierBasisGP(spectrum=pl, coefficients=False, components=30, Tspan=Tspan,
                                           modes=None, name='com_rn')
        else:
            log10_A = parameter.Constant(rn_params[0])
            gamma = parameter.Constant(rn_params[1])
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
            rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
        
        base_model += rn 

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
        gamma_val = 13.0/3
        gamma_gw = parameter.Constant(gamma_val)(gam_name)

        cpl = utils.powerlaw(log10_A=log10_Agw, gamma=gamma_gw)
        gwb = gp_signals.FourierBasisCommonGP(cpl, utils.hd_orf(), coefficients=False,
                                              components=30, Tspan=Tspan,
                                              modes=None, name='gw')

        base_model_gwb = base_model + gwb

    #setting up the pta object
    cws = []
    for i in range(max_n_source):
        log10_fgw = parameter.Uniform(np.log10(3.5e-9), -7)(str(i)+'_'+'log10_fgw')
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
        cw_wf = models.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc,
                     log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0,
                     psi=psi, cos_inc=cos_inc, tref=53000*86400)
        cws.append(models.CWSignal(cw_wf, psrTerm=include_psr_term, name='cw'+str(i)))
    
    ptas = []
    for n_source in range(max_n_source+1):
        PTA = []
        s = base_model
        for i in range(n_source):
            s = s + cws[i]

        model = []
        for p in pulsars:
            model.append(s(p))
        
        #set the likelihood to unity if we are in prior recovery mode
        if prior_recovery:
            PTA.append(get_prior_recovery_pta(signal_base.PTA(model)))
        else:
            PTA.append(signal_base.PTA(model))

        if include_gwb:
            s_gwb = base_model_gwb
            for i in range(n_source):
                s_gwb = s_gwb + cws[i]
        
            model_gwb = []
            for p in pulsars:
                model_gwb.append(s_gwb(p))
            
            #set the likelihood to unity if we are in prior recovery mode
            if prior_recovery:
                PTA.append(get_prior_recovery_pta(signal_base.PTA(model_gwb)))
            else:
                PTA.append(signal_base.PTA(model_gwb))
        ptas.append(PTA)

    print(ptas)
    for i, PTA in enumerate(ptas):
        print(i)
        for j, pta in enumerate(PTA):
            print(j)
            print(ptas[i][j].params)
            #point_to_test = np.tile(np.array([0.0, 0.54, 1.0, -8.0, -13.39, 2.0, 0.5]),i+1)
            #print(PTA.summary())

    #getting the number of dimensions
    #ndim = len(pta.params)

    print("In Fe-proposal we will use p_det={0} and alpha={1}".format(Fe_pdet, Fe_alpha))

    #do n_global_first global proposal steps before starting any other step
    n_global_first = 0
    
    #fisher updating every n_fish_update step
    n_fish_update = 200 #50
    #print out status every n_status_update step
    n_status_update = 10
    #add current sample to de history file every n_de_history step
    n_de_history = 10

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
 
    #array to hold Differential Evolution history
    history_size = 1000    
    de_history = np.zeros((n_chain, history_size, n_source*7+1))
    #start DE after de_start_iter iterations
    de_start_iter = 100
       
    #printitng out the prior used on GWB on/off
    if include_gwb:
        print("Prior on GWB on/off: {0}%".format(gwb_on_prior*100))

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
        num_params += 1
    
    num_noise_params = 0
    if vary_white_noise:
        num_noise_params += len(pulsars)
    if vary_rn:
        num_noise_params += 2

    num_params += num_noise_params
    print(num_params)
    print(num_noise_params)

    samples = np.zeros((n_chain, N, num_params))

    #filling first sample with random draw
    for j in range(n_chain):
        if n_source_start is 'random':
            n_source = np.random.choice(max_n_source+1)
        else:
            n_source = n_source_start
        samples[j,0,0] = n_source
        #print(samples[0,0,:])
        print(n_source)
        if n_source!=0:
            samples[j,0,1:n_source*7+1] = np.hstack(p.sample() for p in ptas[n_source][0].params[:n_source*7])
        #not needed, because zeros are already there: samples[j,0,n_source*7+1:max_n_source*7+1] = np.zeros((max_n_source-n_source)*7)
        #print(samples[0,0,:])
        if vary_white_noise:
            samples[j,0,max_n_source*7+1:max_n_source*7+1+len(pulsars)] = np.ones(len(pulsars))*efac_start
        if vary_rn:
            samples[j,0,max_n_source*7+1+len(pulsars):max_n_source*7+1+num_noise_params] = np.array([ptas[n_source][0].params[n_source*7+num_noise_params-2].sample(), ptas[n_source][0].params[n_source*7+num_noise_params-1].sample()])
        if include_gwb:
            samples[j,0,max_n_source*7+1+num_noise_params] = ptas[n_source][1].params[n_source*7+num_noise_params].sample()
            #samples[j,0,max_n_source*7+1+num_noise_params] = 0.0 #start with GWB off
    print(samples[0,0,:])
    print(ptas[n_source][0].get_lnlikelihood(np.delete(samples[0,0,1:], range(n_source*7,max_n_source*7))))

    #setting up array for the fisher eigenvalues
    #one for cw parameters which we will keep updating
    eig = np.ones((n_chain, max_n_source, 7, 7))*0.1
    
    #one for GWB and common rn parameters, which we will keep updating
    if include_gwb:
        eig_gwb_rn = np.broadcast_to( np.array([[1.0,0,0], [0,0.3,0], [0,0,0.3]]), (n_chain, 3, 3)).copy()
    else:
        eig_gwb_rn = np.broadcast_to( np.array([[1.0,0], [0,0.3]]), (n_chain, 2, 2)).copy()

    #and one for white noise parameters, which we will not update
    eig_wn = np.broadcast_to(np.eye(len(pulsars))*0.1, (n_chain,len(pulsars), len(pulsars)) ).copy()
 
    #calculate wn eigenvectors
    for j in range(n_chain):
        #print('wn eigvec calculation')
        #print(n_source)
        if include_gwb:
            wn_eigvec = get_fisher_eigenvectors(np.delete(samples[j,0,1:], range(n_source*7,max_n_source*7)), ptas[n_source][1], T_chain=Ts[j], n_source=1, dim=len(pulsars), offset=n_source*7)
        else:
            wn_eigvec = get_fisher_eigenvectors(np.delete(samples[j,0,1:], range(n_source*7,max_n_source*7)), ptas[n_source][0], T_chain=Ts[j], n_source=1, dim=len(pulsars), offset=n_source*7)
        #print(wn_eigvec)
        eig_wn[j,:,:] = wn_eigvec[0,:,:]

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros(n_chain+2)
    a_no=np.zeros(n_chain+2)
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

    #set up probabilities of different proposals
    total_weight = (regular_weight + PT_swap_weight + Fe_proposal_weight + 
                    draw_from_prior_weight + de_weight + RJ_weight + gwb_switch_weight + noise_jump_weight)
    swap_probability = PT_swap_weight/total_weight
    fe_proposal_probability = Fe_proposal_weight/total_weight
    regular_probability = regular_weight/total_weight
    draw_from_prior_probability = draw_from_prior_weight/total_weight
    de_probability = de_weight/total_weight
    RJ_probability = RJ_weight/total_weight
    gwb_switch_probability = gwb_switch_weight/total_weight
    noise_jump_probability = noise_jump_weight/total_weight
    print("Percentage of steps doing different jumps:\nPT swaps: {0:.2f}%\nRJ moves: {5:.2f}%\nGWB-switches: {6:.2f}%\n\
Fe-proposals: {1:.2f}%\nJumps along Fisher eigendirections: {2:.2f}%\n\
Draw from prior: {3:.2f}%\nDifferential evolution jump: {4:.2f}%\nNoise jump: {7:.2f}%".format(swap_probability*100,
          fe_proposal_probability*100, regular_probability*100, draw_from_prior_probability*100,
          de_probability*100, RJ_probability*100, gwb_switch_probability*100, noise_jump_probability*100))

    for i in range(int(N-1)):
        #add current sample to DE history
        if i%n_de_history==0 and i>=de_start_iter and de_probability!=0:
            de_hist_index = int((i-de_start_iter)/n_de_history)%history_size
            de_history[:,de_hist_index,:] = samples[:,i,:]
        #print out run state every 10 iterations
        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            if jupyter_notebook:
                print('Progress: {0:2.2f}% '.format(i/N*100) +
                      'Acceptance fraction (RJ, swap, each chain): ({0:1.2f}, {1:1.2f}, '.format(acc_fraction[0], acc_fraction[1]) +
                      ', '.join(['{{{}:1.2f}}'.format(i) for i in range(n_chain)]).format(*acc_fraction[2:]) +
                      ')' + '\r',end='')
            else:
                print('Progress: {0:2.2f}% '.format(i/N*100) +
                      'Acceptance fraction (RJ, swap, each chain): ({0:1.2f}, {1:1.2f}, '.format(acc_fraction[0], acc_fraction[1]) +
                      ', '.join(['{{{}:1.2f}}'.format(i) for i in range(n_chain)]).format(*acc_fraction[2:]) + ')')
        #update our eigenvectors from the fisher matrix every n_fish_update iterations
        if i%n_fish_update==0 and i>=n_global_first:
            #only update T>1 chains every 10th time
            if i%(n_fish_update*10)==0:
                for j in range(n_chain):
                    n_source = int(np.copy(samples[j,i,0]))
                    if n_source!=0:
                        if include_gwb:
                            gwb_on = int(samples[j,i,max_n_source*7+1+num_noise_params]!=0.0)
                            eigvec_rn = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7)), ptas[n_source][gwb_on], T_chain=Ts[j], n_source=1, dim=3, offset=n_source*7+len(pulsars))
                        else:
                            gwb_on = 0
                            eigvec_rn = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7)), ptas[n_source][0], T_chain=Ts[j], n_source=1, dim=2, offset=n_source*7+len(pulsars))
                        eigenvectors = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7)), ptas[n_source][gwb_on], T_chain=Ts[j], n_source=n_source)
                        if np.all(eigenvectors):
                            eig[j,:n_source,:,:] = eigenvectors
                        if np.all(eigvec_rn):
                            eig_gwb_rn[j,:,:] = eigvec_rn[0,:,:]
                    else:
                        if include_gwb:
                            gwb_on = int(samples[j,i,max_n_source*7+1+num_noise_params]!=0.0)
                            eigvec_rn = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7)), ptas[n_source][gwb_on], T_chain=Ts[j], n_source=1, dim=3, offset=n_source*7+len(pulsars))
                        else:
                            eigvec_rn = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7)), ptas[n_source][0], T_chain=Ts[j], n_source=1, dim=2, offset=n_source*7+len(pulsars))
                        #check if eigenvector calculation was succesful
                        #if not, we just keep the initializes eig full of 0.1 values
                        if np.all(eigvec_rn):
                            eig_gwb_rn[j,:,:] = eigvec_rn[0,:,:]
            elif samples[0,i,0]!=0:
                n_source = int(np.copy(samples[0,i,0]))
                if include_gwb:
                    gwb_on = int(samples[0,i,max_n_source*7+1+num_noise_params]!=0.0)
                else:
                    gwb_on = 0
                eigenvectors = get_fisher_eigenvectors(np.delete(samples[0,i,1:], range(n_source*7,max_n_source*7)), ptas[n_source][gwb_on], T_chain=Ts[0], n_source=n_source)
                #check if eigenvector calculation was succesful
                #if not, we just keep the initializes eig full of 0.1 values              
                if np.all(eigenvectors):
                    eig[0,:n_source,:,:] = eigenvectors
        if i<n_global_first:
            do_fe_global_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, fe_file)
        else:
            #draw a random number to decide which jump to do
            jump_decide = np.random.uniform()
            #PT swap move
            if jump_decide<swap_probability:
                do_pt_swap(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_noise_params)
            #global proposal based on Fe-statistic
            elif jump_decide<swap_probability+fe_proposal_probability:
                do_fe_global_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, freqs, fe, inc_max, psi_max, phase0_max, h_max, vary_white_noise, include_gwb, num_noise_params, Fe_pdet, Fe_alpha)
            #draw from prior move
            elif jump_decide<swap_probability+fe_proposal_probability+draw_from_prior_probability:
                do_draw_from_prior_move(n_chain, n_source, pta, samples, i, Ts, a_yes, a_no)
            #do DE jump
            elif (jump_decide<swap_probability+fe_proposal_probability+
                 draw_from_prior_probability+de_probability and i>=de_start_iter):
                do_de_jump(n_chain, ndim, pta, samples, i, Ts, a_yes, a_no, de_history)
            #do RJ move
            elif (jump_decide<swap_probability+fe_proposal_probability+
                 draw_from_prior_probability+de_probability+RJ_probability):
                do_rj_move(n_chain, max_n_source, n_source_prior, ptas, samples, i, Ts, a_yes, a_no, freqs, fe, inc_max, psi_max, phase0_max, h_max, rj_record, vary_white_noise, include_gwb, num_noise_params)
            #do GWB switch move
            elif (jump_decide<swap_probability+fe_proposal_probability+
                 draw_from_prior_probability+de_probability+RJ_probability+gwb_switch_probability):
                gwb_switch_move(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, gwb_on_prior, gwb_log_amp_range)
            #do noise jump
            elif (jump_decide<swap_probability+fe_proposal_probability+
                 draw_from_prior_probability+de_probability+RJ_probability+gwb_switch_probability+noise_jump_probability):
                noise_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, eig_wn, include_gwb, num_noise_params, vary_white_noise)
            #regular step
            else:
                regular_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, eig, eig_gwb_rn, include_gwb, num_noise_params, vary_rn)
    
    acc_fraction = a_yes/(a_no+a_yes)
    return samples, acc_fraction, swap_record, rj_record

################################################################################
#
#GWB SWITCH (ON/OFF) MOVE
#
################################################################################
def gwb_switch_move(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb, num_noise_params, gwb_on_prior, gwb_log_amp_range):
    if not include_gwb:
       raise Exception("include_qwb must be True to use this move")
    for j in range(n_chain):
        n_source = int(np.copy(samples[j,i,0]))
        gwb_on = int(samples[j,i,max_n_source*7+1+num_noise_params]!=0.0)
        
        #turning off ---------------------------------------------------------------------------------------------------------
        if gwb_on==1:
            #if j==0: print("Turn off")
            samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            new_point = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            old_log_amp = np.copy(new_point[n_source*7+num_noise_params])
            #make a dummy enterprise parameter which we will use for getting the proposal density at the value of the old amplitude
            sampling_parameter = parameter.Uniform(gwb_log_amp_range[0], gwb_log_amp_range[1])('dummy')

            new_point[n_source*7+num_noise_params] = 0.0

            #change RN amplitude to help acceptance
            if np.random.uniform()<=0.5:
                new_point[n_source*7+num_noise_params-1] = ptas[n_source][0].params[n_source*7+num_noise_params-1].sample()

            #if j==0: print(samples_current, new_point)
            #if j==0: print(ptas[n_source][0].get_lnlikelihood(new_point)/Ts[j], ptas[n_source][0].get_lnprior(new_point), -ptas[n_source][1].get_lnlikelihood(samples_current)/Ts[j], -ptas[n_source][1].get_lnprior(samples_current))

            log_acc_ratio = ptas[n_source][0].get_lnlikelihood(new_point)/Ts[j]
            log_acc_ratio += ptas[n_source][0].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_source][1].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_source][1].get_lnprior(samples_current)

            acc_ratio = np.exp(log_acc_ratio)*sampling_parameter.get_pdf(old_log_amp)
            #apply on/off prior
            acc_ratio *= (1-gwb_on_prior)/gwb_on_prior
            #if j==0: print(acc_ratio)
            if np.random.uniform()<=acc_ratio:
                #if j==0: print('wooooow')
                samples[j,i+1,0] = n_source
                samples[j,i+1,1:n_source*7+1] = new_point[:n_source*7]
                #samples[j,i+1,max_n_source*7+1:max_n_source*7+1+len(ptas[n_source].pulsars)*2] = new_point[(n_source+1)*7:(n_source+1)*7+len(ptas[n_source].pulsars)*2]
                samples[j,i+1,max_n_source*7+1:] = new_point[n_source*7:]
                a_yes[0] += 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0] += 1
        #turning on ----------------------------------------------------------------------------------------------------------
        else:
            #if j==0: print("Turn on")
            samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            new_point = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            #new_point[n_source*7+num_noise_params] = ptas[0][1].params[num_noise_params].sample()
            #make a dummy enterprise parameter which we will use for drawing from a log-uniform A distribution
            sampling_parameter = parameter.Uniform(gwb_log_amp_range[0], gwb_log_amp_range[1])('dummy')
            new_log_amp = sampling_parameter.sample()
            new_point[n_source*7+num_noise_params] = new_log_amp

            #change RN amplitude to help acceptance
            #new_point[n_source*7+num_noise_params-1] = ptas[n_source][0].params[n_source*7+num_noise_params-1].sample()

            #if j==0: print(samples_current,new_point)
            #if j==0: print(ptas[n_source][1].get_lnlikelihood(new_point)/Ts[j], ptas[n_source][1].get_lnprior(new_point), -ptas[n_source][0].get_lnlikelihood(samples_current)/Ts[j], -ptas[n_source][0].get_lnprior(samples_current))

            log_acc_ratio = ptas[n_source][1].get_lnlikelihood(new_point)/Ts[j]
            log_acc_ratio += ptas[n_source][1].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_source][0].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_source][0].get_lnprior(samples_current)

            acc_ratio = np.exp(log_acc_ratio)/sampling_parameter.get_pdf(new_log_amp)
            #apply on/off prior
            acc_ratio *= gwb_on_prior/(1-gwb_on_prior)
            #if j==0: print(acc_ratio)
            if np.random.uniform()<=acc_ratio:
                #if j==0: print('yeeee')
                samples[j,i+1,0] = n_source
                samples[j,i+1,1:n_source*7+1] = new_point[:n_source*7]
                #samples[j,i+1,max_n_source*7+1:max_n_source*7+1+len(ptas[n_source].pulsars)*2] = new_point[(n_source+1)*7:(n_source+1)*7+len(ptas[n_source].pulsars)*2]
                samples[j,i+1,max_n_source*7+1:] = new_point[n_source*7:]
                a_yes[0] += 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0] += 1

################################################################################
#
#REVERSIBLE-JUMP (RJ, aka TRANS-DIMENSIONAL) MOVE
#
################################################################################
def do_rj_move(n_chain, max_n_source, n_source_prior, ptas, samples, i, Ts, a_yes, a_no, freqs, fe, inc_max, psi_max, phase0_max, h_max, rj_record, vary_white_noise, include_gwb, num_noise_params):
    for j in range(n_chain):
        n_source = int(np.copy(samples[j,i,0]))

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_source*7+1+num_noise_params]!=0.0)
        else:
            gwb_on = 0
        
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
            
            log_f_max = float(ptas[n_source][gwb_on].params[3]._typename.split('=')[2][:-1])
            log_f_min = float(ptas[n_source][gwb_on].params[3]._typename.split('=')[1].split(',')[0])

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
            log10_h = ptas[-1][gwb_on].params[4].sample()
            
            prior_ext = (ptas[-1][gwb_on].params[1].get_pdf(cos_inc) * ptas[-1][gwb_on].params[6].get_pdf(psi) *
                         ptas[-1][gwb_on].params[5].get_pdf(phase0) * ptas[-1][gwb_on].params[4].get_pdf(log10_h))
            
            #cos_inc = np.cos(inc_max[f_idx, hp_idx]) + 2*alpha*(np.random.uniform()-0.5)
            #psi = psi_max[f_idx, hp_idx] + 2*alpha*(np.random.uniform()-0.5)
            #phase0 = phase0_max[f_idx, hp_idx] + 2*alpha*(np.random.uniform()-0.5)
            #log10_h = np.log10(h_max[f_idx, hp_idx]) + 2*alpha*(np.random.uniform()-0.5)

            samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            #print(samples_current)
            
            new_point = np.delete(samples[j,i,1:], range((n_source+1)*7,max_n_source*7))
            new_source = np.array([np.cos(gw_theta), cos_inc, gw_phi, log_f_new, log10_h, phase0, psi])
            new_point[n_source*7:(n_source+1)*7] = new_source

            log_acc_ratio = ptas[(n_source+1)][gwb_on].get_lnlikelihood(new_point)/Ts[j]
            log_acc_ratio += ptas[(n_source+1)][gwb_on].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_source][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_source][gwb_on].get_lnprior(samples_current)

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
                samples[j,i+1,0] = n_source+1
                samples[j,i+1,1:(n_source+1)*7+1] = new_point[:(n_source+1)*7]
                #samples[j,i+1,max_n_source*7+1:max_n_source*7+1+len(ptas[n_source].pulsars)*2] = new_point[(n_source+1)*7:(n_source+1)*7+len(ptas[n_source].pulsars)*2]
                samples[j,i+1,max_n_source*7+1:] = new_point[(n_source+1)*7:]
                a_yes[0] += 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0] += 1

           
        elif n_source==max_n_source or (direction_decide>add_prob and n_source!=0):   #removing a signal----------------------------------------------------------
            if j==0: rj_record.append(-1)
            #choose which source to remove
            remove_index = np.random.randint(n_source)
            
            samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            new_point = np.delete(samples_current, range(remove_index*7,(remove_index+1)*7))
            
            log_acc_ratio = ptas[(n_source-1)][gwb_on].get_lnlikelihood(new_point)/Ts[j]
            log_acc_ratio += ptas[(n_source-1)][gwb_on].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_source][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_source][gwb_on].get_lnprior(samples_current)
            
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

            prior_ext = (ptas[-1][gwb_on].params[1].get_pdf(cos_inc) * ptas[-1][gwb_on].params[6].get_pdf(psi) *
                         ptas[-1][gwb_on].params[5].get_pdf(phase0) * ptas[-1][gwb_on].params[4].get_pdf(log10_h))

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
                samples[j,i+1,1:(n_source-1)*7+1] = new_point[:(n_source-1)*7]
                #samples[j,i+1,max_n_source*7+1:max_n_source*7+1+len(ptas[n_source].pulsars)*2] = new_point[(n_source-1)*7:(n_source-1)*7+len(ptas[n_source].pulsars)*2]                
                samples[j,i+1,max_n_source*7+1:] = new_point[(n_source-1)*7:]
                a_yes[0] += 1
            else:
                samples[j,i+1,:] = samples[j,i,:]
                a_no[0] += 1

################################################################################
#
#DIFFERENTIAL EVOLUTION PROPOSAL ----------------- OUT OF USE
#
################################################################################

def do_de_jump(n_chain, n_source, pta, samples, i, Ts, a_yes, a_no, de_history):
    de_indices = np.random.choice(de_history.shape[1], size=2, replace=False)

    ndim = 7*n_source

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
        
        log_acc_ratio = pta.get_lnlikelihood(new_point[:])/Ts[j]
        log_acc_ratio += pta.get_lnprior(new_point[:])
        log_acc_ratio += -pta.get_lnlikelihood(samples[j,i,:])/Ts[j]
        log_acc_ratio += -pta.get_lnprior(samples[j,i,:])

        acc_ratio = np.exp(log_acc_ratio)
        if np.random.uniform()<=acc_ratio:
            for k in range(ndim):
                samples[j,i+1,k] = new_point[k]
            a_yes[j+2]+=1
        else:
            for k in range(ndim):
                samples[j,i+1,k] = samples[j,i,k]
            a_no[j+2]+=1


################################################################################
#
#DRAW FROM PRIOR MOVE ------------ OUT OF USE
#
################################################################################

def do_draw_from_prior_move(n_chain, n_source, pta, samples, i, Ts, a_yes, a_no):
    ndim = n_source*7
    for j in range(n_chain):
        #make a rendom draw from the prior
        new_point = np.hstack(p.sample() for p in pta.params)

        #calculate acceptance ratio
        log_acc_ratio = pta.get_lnlikelihood(new_point[:])/Ts[j]
        log_acc_ratio += pta.get_lnprior(new_point[:])
        log_acc_ratio += -pta.get_lnlikelihood(samples[j,i,1:])/Ts[j]
        log_acc_ratio += -pta.get_lnprior(samples[j,i,1:])
        
        acc_ratio = np.exp(log_acc_ratio)
        samples[j,i+1,0] = n_source
        if np.random.uniform()<=acc_ratio:
            for k in range(ndim):
                samples[j,i+1,k+1] = new_point[k]
            a_yes[j+2]+=1
        else:
            for k in range(ndim):
                samples[j,i+1,k+1] = samples[j,i,k+1]
            a_no[j+2]+=1

################################################################################
#
#GLOBAL PROPOSAL BASED ON FE-STATISTIC
#
################################################################################

def do_fe_global_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, freqs, fe, inc_max, psi_max, phase0_max, h_max, vary_white_noise, include_gwb, num_noise_params, Fe_pdet, Fe_alpha):
    #ndim = n_source*7
       
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
        n_source = int(np.copy(samples[j,i,0]))
        if n_source==0:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1
            #print("No source to vary!")
            continue

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_source*7+1+num_noise_params]!=0.0)
        else:
            gwb_on = 0

        log_f_max = float(ptas[n_source][gwb_on].params[3]._typename.split('=')[2][:-1])
        log_f_min = float(ptas[n_source][gwb_on].params[3]._typename.split('=')[1].split(',')[0])

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
            log10_h = ptas[-1][gwb_on].params[4].sample()

        #choose randomly which source to change
        source_select = np.random.randint(n_source)
        #print(source_select)
        #print(samples[j,i,:])
        samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))# [j,i,1:n_source*7+1])
        #print(samples_current)
        new_point = np.copy(samples_current)
        new_point[source_select*7:(source_select+1)*7] = np.array([np.cos(gw_theta), cos_inc, gw_phi, log_f_new,
                                                                               log10_h, phase0, psi])
        #print(new_point)
        if fe_new_point>fe_limit:
            fe_new_point=fe_limit        
        
        log_acc_ratio = ptas[n_source][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_acc_ratio += ptas[n_source][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_source][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_source][gwb_on].get_lnprior(samples_current)

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
        
        hastings_extra_factor=1.0
        for k, old_param_fe, new_param, new_param_fe in zip([1,4,5,6], old_params_fe, new_params, new_params_fe):
            old_param = samples[j,i,1+k+source_select*7]
            #check if deterministic top-hat hits any boundaries
            #get prior boundaries
            upper_pb = float(ptas[n_source][gwb_on].params[k]._typename.split('=')[2][:-1])
            lower_pb = float(ptas[n_source][gwb_on].params[k]._typename.split('=')[1].split(',')[0])

            #for new params
            upper_diff_new = upper_pb - new_param_fe
            lower_diff_new = new_param_fe - lower_pb
            #print('-'*10)
            #print(new_param_fe)
            #print(upper_diff_new, lower_diff_new)
            if np.abs(upper_diff_new)<alpha:
                prior_det_new = 1/(alpha+upper_diff_new)
                #print("new hit upper")
            elif np.abs(lower_diff_new)<alpha:
                prior_det_new = 1/(alpha+lower_diff_new)
                #print("new hit lower")
            else:
                prior_det_new = 1/(2*alpha)

            #for old params
            upper_diff_old = upper_pb - old_param_fe
            lower_diff_old = old_param_fe - lower_pb
            #print(upper_diff_old, lower_diff_old)
            if np.abs(upper_diff_old)<alpha:
                prior_det_old = 1/(alpha+upper_diff_old)
                #print("old hit upper")
            elif np.abs(lower_diff_old)<alpha:
                prior_det_old = 1/(alpha+lower_diff_old)
                #print("old hit lower")
            else:
                prior_det_old = 1/(2*alpha)

            
            #True if the ith sample was at a place where we could jump with a deterministic jump
            #False otherwise            
            det_old = np.abs(old_param-old_param_fe)<alpha
            det_new = np.abs(new_param-new_param_fe)<alpha
            #get priors for old and new points
            prior_old = ptas[n_source][gwb_on].params[k].get_pdf(old_param)
            prior_new = ptas[n_source][gwb_on].params[k].get_pdf(new_param)
            
            #probability that it was put there as deterministic given that it's in a deterministic place
            #p_det_indet_old = p_det/(p_det + (1-p_det)*prior_old/prior_det_old )
            #print("p(det| in det)= ", p_det_indet_old)
            if det_new and not det_old: #from non-det to det
                hastings_extra_factor *= prior_old / ( (1-p_det)*prior_new + p_det*prior_det_new ) 
            elif not det_new and det_old: #from det to non-det
                hastings_extra_factor *= ( (1-p_det)*prior_old + p_det*prior_det_old ) / prior_new
            elif det_new and det_old: #from det to det
                hastings_extra_factor *= ( (1-p_det)*prior_old + p_det*prior_det_old ) / ( (1-p_det)*prior_new + p_det*prior_det_new )
            elif not det_new and not det_old: #from non-det to non-det
                hastings_extra_factor *= prior_old / prior_new

        acc_ratio = np.exp(log_acc_ratio)*(fe_old_point/fe_new_point)*hastings_extra_factor
        #not needed (most likely): samples[j,i+1,n_source*7+1:] = np.zeros((max_n_source-n_source)*7)
        if np.random.uniform()<=acc_ratio:
            samples[j,i+1,0] = n_source
            samples[j,i+1,1:n_source*7+1] = new_point[:n_source*7]
            #samples[j,i+1,max_n_source*7+1:max_n_source*7+1+len(ptas[n_source].pulsars)*2] = new_point[n_source*7:n_source*7+len(ptas[n_source].pulsars)*2]
            samples[j,i+1,max_n_source*7+1:] = new_point[n_source*7:]
            a_yes[j+2]+=1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1
    

################################################################################
#
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN CW, GWB AND RN PARAMETERS)
#
################################################################################

def regular_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, eig, eig_gwb_rn, include_gwb, num_noise_params, vary_rn):
    for j in range(n_chain):
        n_source = int(np.copy(samples[j,i,0]))

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_source*7+1+num_noise_params]!=0.0)
        else:
            gwb_on = 0

        samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
        
        #decide if moving in CW source parameters or GWB/RN parameters
        #case #1: we can vary both
        if n_source!=0 and (gwb_on==1 or vary_rn):
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'CW'
            else:
                what_to_vary = 'GWB'
        #case #2: we can only vary CW
        elif n_source!=0:
            what_to_vary = 'CW'
        #case #3: we can only vary GWB
        elif gwb_on==1 or vary_rn:
            what_to_vary = 'GWB'
        #case #4: nothing to vary
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1
            #print("Nothing to vary!")
            continue
        
        if what_to_vary == 'CW':
            source_select = np.random.randint(n_source)
            jump_select = np.random.randint(7)
            jump_1source = eig[j,source_select,jump_select,:]
            jump = np.array([jump_1source[int(i-source_select*7)] if i>=source_select*7 and i<(source_select+1)*7 else 0.0 for i in range(samples_current.size)])
            #print('cw')
            #print(jump)
        elif what_to_vary == 'GWB':
            if include_gwb:
                jump_select = np.random.randint(3)
            else:
                jump_select = np.random.randint(2)
            jump_gwb = eig_gwb_rn[j,jump_select,:]
            if gwb_on==0 and include_gwb:
                jump_gwb[-1] = 0
            if include_gwb:
                jump = np.array([jump_gwb[int(i-n_source*7-len(ptas[n_source][gwb_on].pulsars))] if i>=n_source*7+len(ptas[n_source][gwb_on].pulsars) and i<n_source*7+num_noise_params+1 else 0.0 for i in range(samples_current.size)])
            else:
                jump = np.array([jump_gwb[int(i-n_source*7-len(ptas[n_source][gwb_on].pulsars))] if i>=n_source*7+len(ptas[n_source][gwb_on].pulsars) and i<n_source*7+num_noise_params else 0.0 for i in range(samples_current.size)])
            #if j==0: print('gwb+rn')
            #if j==0: print(i)
            #if j==0: print(jump)
        
        new_point = samples_current + jump*np.random.normal()
        #Try draw from prior in RN sometimes
        #draw_prior_fraction = 0.25
        #if np.random.uniform()<=draw_prior_fraction:
        #    new_point[n_source*7+len(ptas[n_source][gwb_on].pulsars):n_source*7+len(ptas[n_source][gwb_on].pulsars)+2] = np.hstack(p.sample() for p in ptas[n_source][gwb_on].params[n_source*7+len(ptas[n_source][gwb_on].pulsars):n_source*7+num_noise_params])
        #    if j==0: print(new_point)

        log_acc_ratio = ptas[n_source][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_acc_ratio += ptas[n_source][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_source][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_source][gwb_on].get_lnprior(samples_current)

        acc_ratio = np.exp(log_acc_ratio)
        #if j==0: print(acc_ratio)
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("ohh jeez")
            samples[j,i+1,0] = n_source
            samples[j,i+1,1:n_source*7+1] = new_point[:n_source*7]
            #samples[j,i+1,max_n_source*7+1:max_n_source*7+1+len(ptas[n_source].pulsars)*2] = new_point[n_source*7:n_source*7+len(ptas[n_source].pulsars)*2]
            samples[j,i+1,max_n_source*7+1:] = new_point[n_source*7:]
            a_yes[j+2]+=1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1

################################################################################
#
#NOISE MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS IN WHITE NOISE PARAMETERS)
#
################################################################################

def noise_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, eig_wn, include_gwb, num_noise_params, vary_white_noise):
    for j in range(n_chain):
        n_source = int(np.copy(samples[j,i,0]))

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_source*7+1+num_noise_params]!=0.0)
        else:
            gwb_on = 0

        samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
        
        #do the wn jump
        jump_select = np.random.randint(len(ptas[n_source][gwb_on].pulsars))
        #print(jump_select)
        jump_wn = eig_wn[j,jump_select,:]
        jump = np.array([jump_wn[int(i-n_source*7)] if i>=n_source*7 and i<n_source*7+len(ptas[n_source][gwb_on].pulsars) else 0.0 for i in range(samples_current.size)])
        #if j==0: print('noise')
        #if j==0: print(jump)

        new_point = samples_current + jump*np.random.normal()

        log_acc_ratio = ptas[n_source][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_acc_ratio += ptas[n_source][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_source][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_source][gwb_on].get_lnprior(samples_current)

        acc_ratio = np.exp(log_acc_ratio)
        #if j==0: print(acc_ratio)
        if np.random.uniform()<=acc_ratio:
            #if j==0: print("Ohhhh")
            samples[j,i+1,0] = n_source
            samples[j,i+1,1:n_source*7+1] = new_point[:n_source*7]
            #samples[j,i+1,max_n_source*7+1:max_n_source*7+1+len(ptas[n_source].pulsars)*2] = new_point[n_source*7:n_source*7+len(ptas[n_source].pulsars)*2]
            samples[j,i+1,max_n_source*7+1:] = new_point[n_source*7:]
            a_yes[j+2]+=1
        else:
            samples[j,i+1,:] = samples[j,i,:]
            a_no[j+2]+=1


################################################################################
#
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, swap_record, vary_white_noise, include_gwb, num_noise_params):
    swap_chain = np.random.randint(n_chain-1)

    n_source1 = int(np.copy(samples[swap_chain,i,0]))
    n_source2 = int(np.copy(samples[swap_chain+1,i,0]))

    if include_gwb:
        gwb_on1 = int(samples[swap_chain,i,max_n_source*7+1+num_noise_params]!=0.0)
        gwb_on2 = int(samples[swap_chain+1,i,max_n_source*7+1+num_noise_params]!=0.0)
    else:
        gwb_on1 = 0
        gwb_on2 = 0
    
    samples_current1 = np.delete(samples[swap_chain,i,1:], range(n_source1*7,max_n_source*7))
    samples_current2 = np.delete(samples[swap_chain+1,i,1:], range(n_source2*7,max_n_source*7)) 

    log_acc_ratio = -ptas[n_source1][gwb_on1].get_lnlikelihood(samples_current1) / Ts[swap_chain]
    log_acc_ratio += -ptas[n_source2][gwb_on2].get_lnlikelihood(samples_current2) / Ts[swap_chain+1]
    log_acc_ratio += ptas[n_source2][gwb_on2].get_lnlikelihood(samples_current2) / Ts[swap_chain]
    log_acc_ratio += ptas[n_source1][gwb_on1].get_lnlikelihood(samples_current1) / Ts[swap_chain+1]

    acc_ratio = np.exp(log_acc_ratio)
    if np.random.uniform()<=acc_ratio:
        for j in range(n_chain):
            if j==swap_chain:
                samples[j,i+1,:] = samples[j+1,i,:]
            elif j==swap_chain+1:
                samples[j,i+1,:] = samples[j-1,i,:]
            else:
                samples[j,i+1,:] = samples[j,i,:]
        a_yes[1]+=1
        swap_record.append(swap_chain)
    else:
        for j in range(n_chain):
            samples[j,i+1,:] = samples[j,i,:]
        a_no[1]+=1

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

def transdim_postprocess(samples, separation_method='freq', f_tol=0.05, max_n_source=4, status_every=1000):
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
    elif separation_method=='match':
        print("Not implemented yet")
    else:
        print("Not understood separation method: {0}".format(separation_method))
    
    return sample_dict, source_on_idxs


