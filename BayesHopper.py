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

################################################################################
#
#MAIN MCMC ENGINE
#
################################################################################

def run_ptmcmc(N, T_max, n_chain, pulsars, max_n_source=1, RJ_weight=0,
               regular_weight=3, PT_swap_weight=1,
               Fe_proposal_weight=0, fe_file=None, draw_from_prior_weight=0,
               de_weight=0, prior_recovery=False, cw_amp_prior='uniform',
               vary_white_noise=False, wn_params=[1.04,-7],
               include_gwb=False, gwb_switch_weight=0, include_psr_term=False):
    #make sure that we always vary white noise if GWB is included
    if include_gwb:
        vary_white_noise = True
    #setting up base model
    if vary_white_noise:
        efac = parameter.Uniform(0.01, 10.0)
        equad = parameter.Uniform(-8.5, -5)
    else:
        efac = parameter.Constant(wn_params[0]) 
        equad = parameter.Constant(wn_params[1])
    
    ef = white_signals.MeasurementNoise(efac=efac)
    eq = white_signals.EquadNoise(log10_equad=equad)
    tm = gp_signals.TimingModel(use_svd=True)

    base_model = ef + eq + tm
    
    #make base models including GWB
    if include_gwb:
        # find the maximum time span to set GW frequency sampling
        tmin = [p.toas.min() for p in pulsars]
        tmax = [p.toas.max() for p in pulsars]
        Tspan = np.max(tmax) - np.min(tmin)
        # GW parameters (initialize with names here to use parameters in common across pulsars)
        log10_A_gw = parameter.LinearExp(-18,-12)('log10_A_gw')
        gamma_gw = parameter.Constant(4.33)('gamma_gw')
        # gwb (no spatial correlations)
        cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
        gwb = gp_signals.FourierBasisGP(spectrum=cpl, components=30, Tspan=Tspan, name='gw')
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
            log10_h = parameter.Uniform(-18, -11)(str(i)+'_'+'log10_h')
        elif cw_amp_prior == 'uniform':
            log10_h = parameter.LinearExp(-18, -11)(str(i)+'_'+'log10_h')
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

    #do n_global_first global proposal steps before starting any other step
    n_global_first = 0
    
    #fisher updating every n_fish_update step
    n_fish_update = 200 #50
    #print out status every n_status_update step
    n_status_update = 10
    #add current sample to de history file every n_de_history step
    n_de_history = 10

    #array to hold Differential Evolution history
    history_size = 1000    
    de_history = np.zeros((n_chain, history_size, n_source*7+1))
    #start DE after de_start_iter iterations
    de_start_iter = 100

    #setting up temperature ladder (geometric spacing)
    c = T_max**(1.0/(n_chain-1))
    Ts = c**np.arange(n_chain)
    print("Using {0} temperature chains with a geometric spacing of {1:.3f}.\
 Temperature ladder is:\n".format(n_chain,c),Ts)

    #setting up array for the fisher eigenvalues
    eig = np.ones((n_chain, max_n_source, 7, 7))*0.1
    wn_1psr_eig = np.array([[0.1,0],[0,0.1]])
    eig_wn = np.broadcast_to(wn_1psr_eig,(n_chain,len(pulsars), 2, 2))
    eig_gwb = 0.1

    #setting up array for the samples
    if include_gwb:
        samples = np.zeros((n_chain, N, max_n_source*7+1+len(pulsars)*2+1))
    elif vary_white_noise:
        samples = np.zeros((n_chain, N, max_n_source*7+1+len(pulsars)*2))
    else:
        samples = np.zeros((n_chain, N, max_n_source*7+1))
    
    #filling first sample with random draw
    for j in range(n_chain):
        n_source = np.random.choice(max_n_source+1)
        samples[j,0,0] = n_source
        #print(samples[0,0,:])
        print(n_source)
        if n_source!=0:
            samples[j,0,1:n_source*7+1] = np.hstack(p.sample() for p in ptas[n_source][0].params[:n_source*7])
        #not needed, because zeros are already there: samples[j,0,n_source*7+1:max_n_source*7+1] = np.zeros((max_n_source-n_source)*7)
        #print(samples[0,0,:])
        if vary_white_noise:
            #samples[j,0,max_n_source*7+1:max_n_source*7+1+len(pulsars)*2] = np.hstack(p.sample() for p in ptas[n_source].params[n_source*7:n_source*7+len(pulsars)*2])
            #starting from user provided wn parameters (possibly from a prefit)
            samples[j,0,max_n_source*7+1:max_n_source*7+1+len(pulsars)*2:2] = np.ones(len(pulsars))*wn_params[0]
            samples[j,0,max_n_source*7+1+1:max_n_source*7+1+len(pulsars)*2+1:2] = np.ones(len(pulsars))*wn_params[1]
        if include_gwb:
            if np.random.uniform()<0.5:
                samples[j,0,max_n_source*7+1+len(pulsars)*2] = 0.0
            else:
                samples[j,0,max_n_source*7+1+len(pulsars)*2] = ptas[n_source][1].params[n_source*7+len(pulsars)*2].sample()
        #samples[j,0,1:] = np.array([0.5, -0.5, 0.5403, 0.8776, 4.5, 3.5, -8.0969, -7.3979, -13.4133, -12.8381, 1.0, 0.5, 1.0, 0.5])
        #samples[j,0,1:] = np.array([0.0, 0.54, 1.0, -8.0, -13.39, 2.0, 0.5])
    print(samples[0,0,:])

    #setting up arrays to record acceptance and swaps
    a_yes=np.zeros(n_chain+2)
    a_no=np.zeros(n_chain+2)
    swap_record = []
    rj_record = []

    #set up probabilities of different proposals
    total_weight = (regular_weight + PT_swap_weight + Fe_proposal_weight + 
                    draw_from_prior_weight + de_weight + RJ_weight + gwb_switch_weight)
    swap_probability = PT_swap_weight/total_weight
    fe_proposal_probability = Fe_proposal_weight/total_weight
    regular_probability = regular_weight/total_weight
    draw_from_prior_probability = draw_from_prior_weight/total_weight
    de_probability = de_weight/total_weight
    RJ_probability = RJ_weight/total_weight
    gwb_switch_probability = gwb_switch_weight/total_weight
    print("Percentage of steps doing different jumps:\nPT swaps: {0:.2f}%\nRJ moves: {5:.2f}%\nGWB-switches: {6:.2f}%\n\
Fe-proposals: {1:.2f}%\nJumps along Fisher eigendirections: {2:.2f}%\n\
Draw from prior: {3:.2f}%\nDifferential evolution jump: {4:.2f}%".format(swap_probability*100,
          fe_proposal_probability*100, regular_probability*100, draw_from_prior_probability*100,
          de_probability*100, RJ_probability*100, gwb_switch_probability*100))

    for i in range(int(N-1)):
        #add current sample to DE history
        if i%n_de_history==0 and i>=de_start_iter and de_probability!=0:
            de_hist_index = int((i-de_start_iter)/n_de_history)%history_size
            de_history[:,de_hist_index,:] = samples[:,i,:]
        #print out run state every 10 iterations
        if i%n_status_update==0:
            acc_fraction = a_yes/(a_no+a_yes)
            print('Progress: {0:2.2f}% '.format(i/N*100) +
                  'Acceptance fraction (RJ, swap, each chain): ({0:1.2f}, {1:1.2f}, '.format(acc_fraction[0], acc_fraction[1]) +
                  ', '.join(['{{{}:1.2f}}'.format(i) for i in range(n_chain)]).format(*acc_fraction[2:]) +
                  ')' + '\r',end='')
        #update our eigenvectors from the fisher matrix every n_fish_update iterations
        if i%n_fish_update==0 and i>=n_global_first:
            #only update T>1 chains every 10th time
            if i%(n_fish_update*10)==0:
                for j in range(n_chain):
                    n_source = int(np.copy(samples[j,i,0]))
                    if n_source!=0:
                        if include_gwb:
                            gwb_on = int(samples[j,i,max_n_source*7+1+len(pulsars)*2]!=0.0)
                        else:
                            gwb_on = 0
                        eigenvectors = get_fisher_eigenvectors(np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7)), ptas[n_source][gwb_on], T_chain=Ts[j], n_source=n_source)
                    else:
                        continue
                    #check if eigenvector calculation was succesful
                    #if not, we just keep the initializes eig full of 0.1 values
                    if np.all(eigenvectors):
                        eig[j,:n_source,:,:] = eigenvectors
            elif samples[0,i,0]!=0:
                n_source = int(np.copy(samples[0,i,0]))
                if include_gwb:
                    gwb_on = int(samples[0,i,max_n_source*7+1+len(pulsars)*2]!=0.0)
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
                do_pt_swap(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, swap_record, vary_white_noise, include_gwb)
            #global proposal based on Fe-statistic
            elif jump_decide<swap_probability+fe_proposal_probability:
                do_fe_global_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, fe_file, vary_white_noise, include_gwb)
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
                do_rj_move(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, fe_file, rj_record, vary_white_noise, include_gwb)
            #do GWB switch move
            elif (jump_decide<swap_probability+fe_proposal_probability+
                 draw_from_prior_probability+de_probability+RJ_probability+gwb_switch_probability):
                gwb_switch_move(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb)
            #regular step
            else:
                regular_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, eig, eig_wn, eig_gwb, vary_white_noise, include_gwb)
    
    acc_fraction = a_yes/(a_no+a_yes)
    return samples, acc_fraction, swap_record, rj_record

################################################################################
#
#GWB SWITCH (ON/OFF) MOVE
#
################################################################################
def gwb_switch_move(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, vary_white_noise, include_gwb):
    if not include_gwb:
       raise Exception("include_qwb must be True to use this move")
    for j in range(n_chain):
        n_source = int(np.copy(samples[j,i,0]))
        gwb_on = int(samples[j,i,max_n_source*7+1+len(ptas[-1][0].pulsars)*2]!=0.0)
        
        #turning off ---------------------------------------------------------------------------------------------------------
        if gwb_on==1:
            #print("Turn off")
            samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            new_point = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            new_point[n_source*7+len(ptas[n_source][gwb_on].pulsars)*2] = 0.0
            #print(samples_current, new_point)

            log_acc_ratio = ptas[n_source][0].get_lnlikelihood(new_point)/Ts[j]
            log_acc_ratio += ptas[n_source][0].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_source][1].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_source][1].get_lnprior(samples_current)

            acc_ratio = np.exp(log_acc_ratio)
            
            if np.random.uniform()<=acc_ratio:
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
            #print("Turn on")
            samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            new_point = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
            new_point[n_source*7+len(ptas[n_source][gwb_on].pulsars)*2] = ptas[0][1].params[len(ptas[n_source][gwb_on].pulsars)*2].sample()
            #print(samples_current,new_point)

            log_acc_ratio = ptas[n_source][1].get_lnlikelihood(new_point)/Ts[j]
            log_acc_ratio += ptas[n_source][1].get_lnprior(new_point)
            log_acc_ratio += -ptas[n_source][0].get_lnlikelihood(samples_current)/Ts[j]
            log_acc_ratio += -ptas[n_source][0].get_lnprior(samples_current)

            acc_ratio = np.exp(log_acc_ratio)

            if np.random.uniform()<=acc_ratio:
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
def do_rj_move(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, fe_file, rj_record, vary_white_noise, include_gwb):
    for j in range(n_chain):
        n_source = int(np.copy(samples[j,i,0]))

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_source*7+1+len(ptas[-1][0].pulsars)*2]!=0.0)
        else:
            gwb_on = 0
        
        add_prob = 0.5 #flat prior on n_source-->same propability of addind and removing
        #decide if we add or remove a signal
        direction_decide = np.random.uniform()
        if n_source==0 or (direction_decide<add_prob and n_source!=max_n_source): #adding a signal------------------------------------------------------
            if j==0: rj_record.append(1)
            if fe_file==None:
                raise Exception("Fe-statistics data file is needed for Fe global propsals")
            npzfile = np.load(fe_file)
            freqs = npzfile['freqs']
            fe = npzfile['fe']
            inc_max = npzfile['inc_max']
            psi_max = npzfile['psi_max']
            phase0_max = npzfile['phase0_max']
            h_max = npzfile['h_max']
   
            #alpha = 0.1
 
            #set limit used for rejection sampling below
            fe_limit = np.max(fe)
            #if the max is too high, cap it at Fe=200 (Neil's trick to not to be too restrictive)
            #if fe_limit>200:
            #    fe_limit=200
    
            accepted = False
            while accepted==False:
                log_f_new = ptas[-1][gwb_on].params[3].sample()
                f_idx = (np.abs(np.log10(freqs) - log_f_new)).argmin()

                gw_theta = np.arccos(ptas[-1][gwb_on].params[0].sample())
                gw_phi = ptas[-1][gwb_on].params[2].sample()
                hp_idx = hp.ang2pix(hp.get_nside(fe), gw_theta, gw_phi)

                fe_new_point = fe[f_idx, hp_idx]
                if np.random.uniform()<(fe_new_point/fe_limit):
                    accepted = True

            cos_inc = ptas[-1][gwb_on].params[1].sample()
            psi = ptas[-1][gwb_on].params[6].sample()
            phase0 = ptas[-1][gwb_on].params[5].sample()
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
            norm = np.sum(fe)*healpy_pixel_area*log10f_resolution
            #short-term hacky solution
            norm *= 399/400


            #normalization
            fe_new_point_normalized = fe_new_point/norm

            acc_ratio = np.exp(log_acc_ratio)/prior_ext/fe_new_point_normalized
            #correction close to edge based on eqs. (40) and (41) of Sambridge et al. Geophys J. Int. (2006) 167, 528-542
            if n_source==0:
                acc_ratio *= 0.5
            elif n_source==max_n_source-1:
                acc_ratio *= 2.0
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
            if fe_file==None:
                raise Exception("Fe-statistics data file is needed for Fe global propsals")
            npzfile = np.load(fe_file)
            freqs = npzfile['freqs']
            fe = npzfile['fe']
            inc_max = npzfile['inc_max']
            psi_max = npzfile['psi_max']
            phase0_max = npzfile['phase0_max']
            h_max = npzfile['h_max']

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
            norm = np.sum(fe)*healpy_pixel_area*log10f_resolution
            #short-term hacky solution
            norm *= 399/400
            
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
            elif n_source==max_n_source:
                acc_ratio *= 0.5
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
#DIFFERENTIAL EVOLUTION PROPOSAL
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
#DRAW FROM PRIOR MOVE
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

def do_fe_global_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, fe_file, vary_white_noise, include_gwb):    
    if fe_file==None:
        raise Exception("Fe-statistics data file is needed for Fe global propsals")
    npzfile = np.load(fe_file)
    freqs = npzfile['freqs']
    fe = npzfile['fe']
    inc_max = npzfile['inc_max']
    psi_max = npzfile['psi_max']
    phase0_max = npzfile['phase0_max']
    h_max = npzfile['h_max']

    #ndim = n_source*7
       
    #set probability of deterministic vs flat proposal in extrinsic parameters
    p_det = 0.5
    #set width of deterministic proposal
    alpha = 0.1

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
            gwb_on = int(samples[j,i,max_n_source*7+1+len(ptas[-1][0].pulsars)*2]!=0.0)
        else:
            gwb_on = 0

        accepted = False
        while accepted==False:
            f_new = 10**(ptas[-1][gwb_on].params[3].sample())
            f_idx = (np.abs(freqs - f_new)).argmin()

            gw_theta = np.arccos(ptas[-1][gwb_on].params[0].sample())
            gw_phi = ptas[-1][gwb_on].params[2].sample()
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
            cos_inc = ptas[-1][gwb_on].params[1].sample()
            psi = ptas[-1][gwb_on].params[6].sample()
            phase0 = ptas[-1][gwb_on].params[5].sample()
            log10_h = ptas[-1][gwb_on].params[4].sample()

        #choose randomly which source to change
        source_select = np.random.randint(n_source)
        #print(source_select)
        #print(samples[j,i,:])
        samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))# [j,i,1:n_source*7+1])
        #print(samples_current)
        new_point = np.copy(samples_current)
        new_point[source_select*7:(source_select+1)*7] = np.array([np.cos(gw_theta), cos_inc, gw_phi, np.log10(f_new),
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
        prior_ranges = [2.0, 7.0, 2.0*np.pi, np.pi]
        
        new_params = [cos_inc, log10_h, phase0, psi]
        new_params_fe = [np.cos(inc_max[f_idx, hp_idx]), np.log10(h_max[f_idx, hp_idx]),
                        phase0_max[f_idx, hp_idx], psi_max[f_idx, hp_idx]]
        
        hastings_extra_factor=1.0
        for k, prior_range, old_param_fe, new_param, new_param_fe in zip([1,4,5,6], prior_ranges, old_params_fe, new_params, new_params_fe):
            old_param = samples[j,i,1+k+source_select*7]
            #True if the ith sample was at a place where we could jump with a deterministic jump
            #False otherwise            
            det_old = np.abs(old_param-old_param_fe)<alpha
            det_new = np.abs(new_param-new_param_fe)<alpha
            if det_new and not det_old: #from non-det to det
                hastings_extra_factor *= 1.0/( p_det/(1-p_det)*prior_range/(2*alpha) + 1 )
            elif not det_new and det_old: #from det to non-det
                hastings_extra_factor *= p_det/(1-p_det)*prior_range/(2*alpha) + 1

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
#REGULAR MCMC JUMP ROUTINE (JUMPING ALONG EIGENDIRECTIONS)
#
################################################################################

def regular_jump(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, eig, eig_wn, eig_gwb, vary_white_noise, include_gwb):
    for j in range(n_chain):
        n_source = int(np.copy(samples[j,i,0]))

        if include_gwb:
            gwb_on = int(samples[j,i,max_n_source*7+1+len(ptas[-1][0].pulsars)*2]!=0.0)
        else:
            gwb_on = 0

        samples_current = np.delete(samples[j,i,1:], range(n_source*7,max_n_source*7))
        
        #decide if moving in source parameters or white noise parameters
        #case #1: we can vary all of them
        if n_source!=0 and vary_white_noise==True and gwb_on==1:
            vary_decide = np.random.uniform()
            if vary_decide <= 1/3.0:
                what_to_vary = 'CW'
            elif vary_decide <=2.0/3.0:
                what_to_vary = 'WN'
            else:
                what_to_vary = 'GWB'
        #case #2: we can't vary GWB
        elif n_source!=0 and vary_white_noise==True:
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'CW'
            else:
                what_to_vary = 'WN'
        #case #3: we can't vary WN
        elif n_source!=0 and gwb_on==1:
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'CW'
            else:
                what_to_vary = 'GWB'
        #case #4: we can't vary CW
        elif vary_white_noise==True and gwb_on==1:
            vary_decide = np.random.uniform()
            if vary_decide <= 0.5:
                what_to_vary = 'WN'
            else:
                what_to_vary = 'GWB'
        elif n_source!=0:
            what_to_vary = 'CW'
        elif vary_white_noise==True:
            what_to_vary = 'WN'
        elif gwb_on==1:
            what_to_vary = 'GWB'
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
        elif what_to_vary == 'WN':
            pulsar_select = np.random.randint(len(ptas[n_source][gwb_on].pulsars))
            jump_select = np.random.randint(2) #we have two wn parameters, ecorr and efac
            jump_1psr = eig_wn[j,pulsar_select,jump_select,:]
            jump = np.array([jump_1psr[int(i-n_source*7-pulsar_select*2)] if i>=n_source*7+pulsar_select*2 and i<n_source*7+(pulsar_select+1)*2 else 0.0 for i in range(samples_current.size)])
        elif what_to_vary == 'GWB':
            jump = np.array([eig_gwb if i==n_source*7+len(ptas[n_source][gwb_on].pulsars)*2 else 0.0 for i in range(samples_current.size)])
        
        new_point = samples_current + jump*np.random.normal()

        log_acc_ratio = ptas[n_source][gwb_on].get_lnlikelihood(new_point)/Ts[j]
        log_acc_ratio += ptas[n_source][gwb_on].get_lnprior(new_point)
        log_acc_ratio += -ptas[n_source][gwb_on].get_lnlikelihood(samples_current)/Ts[j]
        log_acc_ratio += -ptas[n_source][gwb_on].get_lnprior(samples_current)

        acc_ratio = np.exp(log_acc_ratio)
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
#PARALLEL TEMPERING SWAP JUMP ROUTINE
#
################################################################################
def do_pt_swap(n_chain, max_n_source, ptas, samples, i, Ts, a_yes, a_no, swap_record, vary_white_noise, include_gwb):
    swap_chain = np.random.randint(n_chain-1)

    n_source1 = int(np.copy(samples[swap_chain,i,0]))
    n_source2 = int(np.copy(samples[swap_chain+1,i,0]))

    if include_gwb:
        gwb_on1 = int(samples[swap_chain,i,max_n_source*7+1+len(ptas[-1][0].pulsars)*2]!=0.0)
        gwb_on2 = int(samples[swap_chain+1,i,max_n_source*7+1+len(ptas[-1][0].pulsars)*2]!=0.0)
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
def get_fisher_eigenvectors(params, pta, T_chain=1, epsilon=1e-4, n_source=1):
    #get dimension and set up an array for the fisher matrix    
    #dim = int(params.shape[0]/n_source)
    #hardcoding 7 instead, because we can have white noise parameters, which we don'T want to vary here
    dim = 7
    fisher = np.zeros((n_source,dim,dim))
    eig = []

    #print(params)

    #lnlikelihood at specified point
    nn = pta.get_lnlikelihood(params)
    
    
    for k in range(n_source):
        #print(k)
        #calculate diagonal elements
        for i in range(dim):
            #create parameter vectors with +-epsilon in the ith component
            paramsPP = np.copy(params)
            paramsMM = np.copy(params)
            paramsPP[i+k*dim] += 2*epsilon
            paramsMM[i+k*dim] -= 2*epsilon
            #print(paramsPP)
            
            #lnlikelihood at +-epsilon positions
            pp = pta.get_lnlikelihood(paramsPP)
            mm = pta.get_lnlikelihood(paramsMM)

            #print(pp, nn, mm)
            
            #calculate diagonal elements of the Hessian from a central finite element scheme
            #note the minus sign compared to the regular Hessian
            fisher[k,i,i] = -(pp - 2.0*nn + mm)/(4.0*epsilon*epsilon)

        #calculate off-diagonal elements
        for i in range(dim):
            for j in range(i+1,dim):
                #create parameter vectors with ++, --, +-, -+ epsilon in the ith and jth component
                paramsPP = np.copy(params)
                paramsMM = np.copy(params)
                paramsPM = np.copy(params)
                paramsMP = np.copy(params)

                paramsPP[i+k*dim] += epsilon
                paramsPP[j+k*dim] += epsilon
                paramsMM[i+k*dim] -= epsilon
                paramsMM[j+k*dim] -= epsilon
                paramsPM[i+k*dim] += epsilon
                paramsPM[j+k*dim] -= epsilon
                paramsMP[i+k*dim] -= epsilon
                paramsMP[j+k*dim] += epsilon

                #lnlikelihood at those positions
                pp = pta.get_lnlikelihood(paramsPP)
                mm = pta.get_lnlikelihood(paramsMM)
                pm = pta.get_lnlikelihood(paramsPM)
                mp = pta.get_lnlikelihood(paramsMP)

                #calculate off-diagonal elements of the Hessian from a central finite element scheme
                #note the minus sign compared to the regular Hessian
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
            eig_limit = 100.0    
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

def transdim_postprocess(samples, separation_method='freq', f_tol=0.05, max_n_source=4):
    N = samples.shape[0]
    
    if separation_method=='freq':
        freqs = []
        sample_dict = {}
        for i in range(N):
            for j,f in enumerate(samples[i,4:max_n_source*7:7]):
                if not np.isnan(f):
                    new = True
                    for idx, freq in enumerate(freqs):
                        if np.abs(f-freq)<f_tol:
                            new = False
                            freq_idx = idx
                    if new:
                        freqs.append(f)
                        sample_dict[len(freqs)-1] = np.array([list(samples[i,1+7*j:1+7*(j+1)]),])
                    else:
                        sample_dict[freq_idx] = np.append(sample_dict[freq_idx], np.array([list(samples[i,1+7*j:1+7*(j+1)]),]), axis=0)                 
    elif separation_method=='match':
        print("Not implemented yet")
    else:
        print("Not understood separation method: {0}".format(separation_method))
    
    return sample_dict


