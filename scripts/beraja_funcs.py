'''
This is a script with useful functions used by other scripts
I am not sure if this is the best work flow. But it
keeps things very clean in the scripts.
'''

import numpy as np
import quantecon as qe
from numba import njit, prange, vectorize

class borrower():

    def __init__(self,
                 sigma = 2,            # inverse intertemporal elasticity
                 r = 0.03,             # rate on assets
                 beta = 0.95,          # discount rate
                 rmmin = 0.05,         # mortgage rate min
                 rmmax = 0.06,         # mortgage rate max
                 rmsize = 2,           # number of rms
                 gamma = 0.8,          # ltv ratio
                 mu = 0.025,           # house price growth (which is equal to income growth)
                 xmin= -1.025,         # inverse accumulated equity min
                 xmax = 0.45,
                 xsize = 60,
                 amin = 0,
                 amax = 1,
                 asize = 64,
                 ymin = -0.5,
                 ymax = 0.5,
                 ysize = 46,
                 sigmay = 0.1,
                 sigmap = 0.065,
                 yshocksize = 3,
                 pshocksize = 3,
                 Fnodes = np.array([0.105, .052]),
                 probF = np.array([0.875, 0.125]),
                 ):

        # == assigning paramters to "self" == #

        (self.sigma, self.r, self.beta, self.rmmin,
        self.rmmax, self.rmsize, self.gamma, self.mu,
        self.xmin, self.xmax, self.xsize, self.amin,
        self.amax, self.asize, self.ymin, self.ymax,
        self.ysize, self.sigmay, self.sigmap, self.yshocksize,
        self.pshocksize, self.Fnodes, self.probF) = (sigma, r, beta, rmmin, rmmax,
        rmsize, gamma, mu, xmin, xmax, xsize, amin, amax, asize, ymin, ymax,
        ysize, sigmay, sigmap, yshocksize, pshocksize, Fnodes, probF)

        # == getting grids == #

        rmnodes = self.rmnodes = np.linspace(rmmin, rmmax, rmsize)
        ynodes = np.linspace(ymin, ymax, ysize)

        # xgrid
        xnodes = np.linspace(xmin, xmax, xsize)

        # agrid
        # this is not evenly spaced. More nodes at the lower a values
        anodes = np.empty(asize)
        for i in range(asize):
            anodes[i]=(1.0/(asize-1))*(1.0*i-1.0)
        for i in range(asize):
            anodes[i]=np.exp(np.log(amax-amin+1)*anodes[i])+amin-1.0

        self.anodes = anodes


        # == getting grids and probabilities for shocks == #

        mc_y = qe.tauchen(0, sigmay, n=yshocksize)
        self.probyshock = probyshock = mc_y.P[0, :]
        self.yshocknodes = yshocknodes = mc_y.state_values
        self.probyshock_cum = probyshock_cum = np.cumsum(self.probyshock)

        mc_p = qe.tauchen(0, sigmap, n=pshocksize)
        self.probpshock = probpshock = mc_p.P[0, :]
        self.pshocknodes = pshocknodes = mc_p.state_values
        self.probpshock_cum = probpshock_cum = np.cumsum(self.probpshock)

        # defining the location of the the x value closest to 0
        # (used when constructing refinance value function)
        self.xreset = np.argmin(np.abs(xnodes))

        # == creating vectors to find closest match after a shock for a, x and y == #
        # These are index values for a given shock and a given level of the variable
        # For example, for a given shock and a given asset value, where is the closest
        # recorded asset value in my grid which corresponds to the resulting asset value
        # from the equation

        xnearest = np.empty((xsize, pshocksize), dtype=int)
        for i in range(xsize):
            for j in range(pshocksize):
                xnearest[i, j] = int(np.argmin(np.abs((xnodes[i] - mu - pshocknodes[j]) - xnodes)))

        anearest = np.empty((asize, pshocksize), dtype=int)
        for i in range(asize):
            for j in range(pshocksize):
                anearest[i, j] = int(np.argmin(np.abs((anodes[i]*np.exp(-mu-pshocknodes[j])) - anodes)))

        ynearest = np.empty((ysize, pshocksize, yshocksize), dtype=int)
        for i in range(ysize):
            for j in range(pshocksize):
                for k in range(yshocksize):
                    ynearest[i,j,k] = int(np.argmin(np.abs((ynodes[i]+yshocknodes[k]-pshocknodes[j]) - ynodes)))

        self.xnearest, self.anearest, self.ynearest = xnearest, anearest, ynearest

        # "unlogging" x and y nodes
        self.xnodes = np.exp(xnodes)
        self.ynodes = np.exp(ynodes)


    def unpack_params(self):

        # returns all relevant objects on call
        return (self.sigma, self.r, self.beta, self.rmmin,
                self.rmmax, self.rmsize, self.gamma, self.mu,
                self.xmin, self.xmax, self.xsize, self.amin,
                self.amax, self.asize, self.ymin, self.ymax,
                self.ysize, self.sigmay, self.sigmap, self.yshocksize,
                self.pshocksize, self.Fnodes, self.probF,
                self.yshocknodes, self.pshocknodes,
                self.rmnodes, self.xnodes, self.anodes,
                self.ynodes, self.probyshock, self.probyshock_cum,
                self.probpshock, self.probpshock_cum, self.xreset,
                self.xnearest, self.anearest, self.ynearest)

def operator_factory(agent):

    (sigma, r, beta, rmmin, rmmax, rmsize,
     gamma, mu, xmin, xmax, xsize, amin,
     amax, asize, ymin, ymax, ysize, sigmay,
     sigmap, yshocksize, pshocksize, Fnodes,
     probF, yshocknodes, pshocknodes, rmnodes,
     xnodes, anodes, ynodes, probyshock,
     probyshock_cum, probpshock, probpshock_cum,
     xreset, xnearest, anearest, ynearest) = agent.unpack_params()


    @vectorize
    def u(c, sigma):
        '''
        CRRA utility
        '''
        if c < 1e-10:
            return -np.inf
        elif sigma == 1:
            return np.log(c)
        else:
            return (c**(1 - sigma))/(1 - sigma)


    @njit(parallel = True)
    def T(vold, vrefi, vnorefi, vrefi_out, vnorefi_out, pol_refi, pol_norefi):
        '''
        bellman operator
        '''
        for a_i in prange(asize):
            for y_i in prange(ysize):
                for x_i in prange(xsize):
                    for r_0i, r_0 in enumerate(rmnodes):
                        for r_1i, r_1 in enumerate(rmnodes):
                            for f_i, f in enumerate(Fnodes):

                                # getting node values for parallelized loops
                                a = anodes[a_i]; x = xnodes[x_i]; y = ynodes[y_i]

                                # == Refinancing value function == #

                                # getting income if hh decides to refinance
                                inc_ref = (a * (1.0 + r) +
                                          y -
                                          gamma * r_1 +
                                          gamma * (1.0 - x) -
                                          f)

                                # getting highest feasible asset choice location on our asset grid
                                idx_ref = np.searchsorted(anodes, inc_ref)

                                max_sofar_refi = -1e10

                                # looping over feasible asset choices
                                for a_1 in range(idx_ref):
                                    a_next = anodes[a_1]

                                    util = u(inc_ref - a_next, sigma)

                                    # expectations of future value
                                    e = 0.0
                                    for ps_i, prob_ps in enumerate(probpshock):
                                        for ys_i, prob_ys in enumerate(probyshock):
                                            for fs_i, prob_fs in enumerate(probF):
                                                e += (
                                                    vold[anearest[a_1, ps_i],
                                                         ynearest[y_i, ps_i, ys_i],
                                                         xnearest[xreset, ps_i],
                                                         r_1i, r_1i, fs_i] *
                                                    prob_ps * prob_ys * prob_fs
                                                )

                                    val_refi = util + beta * np.exp(mu * (1 - sigma)) * e

                                    if val_refi > max_sofar_refi:
                                        max_sofar_refi = val_refi
                                        a_refi = a_1

                                # == No Refinance == #

                                # getting income if hh decides not to refinance
                                inc_noref = (
                                    a * (1.0 + r) +
                                    y -
                                    gamma * r_0 * x
                                )

                                # getting highest feasible asset choice location on our asset grid
                                idx_noref = np.searchsorted(anodes, inc_noref)

                                max_sofar_norefi = -1e10

                                # looping over feasible asset choices
                                for a_1 in range(idx_noref):

                                    a_next = anodes[a_1]

                                    util = u(inc_noref - a_next, sigma)

                                    # expected future value
                                    e = 0.0
                                    for ps_i, prob_ps in enumerate(probpshock):
                                        for ys_i, prob_ys in enumerate(probyshock):
                                            for fs_i, prob_fs in enumerate(probF):
                                                    e += (
                                                        vold[anearest[a_1, ps_i],
                                                             ynearest[y_i, ps_i, ys_i],
                                                             xnearest[x_i, ps_i],
                                                             r_0i, r_1i, fs_i] *
                                                        prob_ps * prob_ys * prob_fs
                                                    )

                                    val_norefi = util + beta * np.exp(mu * (1 - sigma)) * e

                                    if val_norefi > max_sofar_norefi:
                                        max_sofar_norefi = val_norefi
                                        a_norefi = a_1

                                # == allocating values and asset allocations to arrays == #

                                vrefi_out[a_i, y_i, x_i, r_0i, r_1i, f_i] = max_sofar_refi
                                vnorefi_out[a_i, y_i, x_i, r_0i, r_1i, f_i] = max_sofar_norefi

                                pol_refi[a_i, y_i, x_i, r_0i, r_1i, f_i] = a_refi
                                pol_norefi[a_i, y_i, x_i, r_0i, r_1i, f_i] = a_norefi






    @njit
    def get_policies(v_refi, v_norefi, pol_refi, pol_norefi):
        '''
        Getting asset saving and consumption
        policies from value functions
        '''

        # initialize empty "choice" frames
        refichoice = np.empty_like(v_refi)
        achoice = np.empty_like(pol_refi)
        cchoice = np.empty_like(pol_refi)
        policy_matrix = np.empty_like(pol_refi)


        for a_i, a in enumerate(anodes):
            for y_i, y in enumerate(ynodes):
                for x_i, x in enumerate(xnodes):
                    for r_0i, r_0 in enumerate(rmnodes): # old rate
                        for r_1i, r_1 in enumerate(rmnodes): # new rate
                            for f_i, f in enumerate(Fnodes):

                                # == check which value function is higher == #

                                if v_refi[a_i, y_i, x_i, r_0i, r_1i, f_i] > v_norefi[a_i, y_i, x_i, r_0i, r_1i, f_i]:
                                    # refinance
                                    refichoice[a_i, y_i, x_i, r_0i, r_1i, f_i] = 1.0
                                    achoice[a_i, y_i, x_i, r_0i, r_1i, f_i] = pol_refi[a_i, y_i, x_i, r_0i, r_1i, f_i]
                                    cchoice[a_i, y_i, x_i, r_0i, r_1i, f_i] = ((1.0 + r) * a  + y -
                                                                 (gamma * r_1) + gamma * (1 - x) -
                                                                 f - anodes[achoice[a_i, y_i, x_i, r_0i, r_1i, f_i]])
                                else:
                                    # doesnt refinance
                                    refichoice[a_i, y_i, x_i, r_0i, r_1i, f_i] = 0.0
                                    achoice[a_i, y_i, x_i, r_0i, r_1i, f_i] = pol_norefi[a_i, y_i, x_i, r_0i, r_1i, f_i]
                                    cchoice[a_i, y_i, x_i, r_0i, r_1i, f_i] = ((1.0 + r) * a + y -
                                                                 gamma * r_0 * x -
                                                                anodes[achoice[a_i, y_i, x_i, r_0i, r_1i, f_i]])

        return refichoice, achoice, cchoice


    @njit
    def vfi(T, tol=1e-4, max_iter=1000):
        '''
        value function iterator
        '''

        v_in = np.empty((asize,ysize,xsize,rmsize,rmsize,len(Fnodes)))
        vrefi_in = np.empty_like(v_in)
        vnorefi_in = np.empty_like(v_in)
        pol_refi = np.empty_like(v_in, dtype=np.int_)
        pol_norefi = np.empty_like(v_in, dtype=np.int_)
        vrefi_out = np.empty_like(v_in)
        vnorefi_out = np.empty_like(v_in)

        # Set up loop
        i = 0
        error = tol + 1

        while i < max_iter and error > tol:
            T(v_in, vrefi_in, vnorefi_in, vrefi_out, vnorefi_out, pol_refi, pol_norefi)
            i += 1
            error = max(
                np.max(np.abs(vrefi_in - vrefi_out)),
                np.max(np.abs(vnorefi_in - vnorefi_out))
            )
            vrefi_in, vnorefi_in = vrefi_out, vnorefi_out
            v_out = np.maximum(vrefi_in, vnorefi_in)
            v_in = v_out

        return v_out, vrefi_out, vnorefi_out, pol_refi, pol_norefi, error

    return T, get_policies, vfi
