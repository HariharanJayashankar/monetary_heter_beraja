'''
Goals of the notebook:

- [ ] Replicate figure IX - Switching threshold for different refinancing rates
    - This is done right after calculating the value functions for borrowers
- [ ] Replicate figure X - IRFs of consmption and refinance propensity to a mortgage rate decline
'''

import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from numba import njit, prange, vectorize
from beraja_funcs import * # importing helper functions

# make this 'full_run' if you want to go with full grid sizes
param_set = 'full_run'

if param_set == 'quick_run':
    xgrid_size = 30
    agrid_size = 30
    ygrid_size = 30
elif param_set == 'full_run':
    xgrid_size = 60
    agrid_size = 64
    ygrid_size = 46

# == initialize agent and operators we need == #
hh = borrower(xsize = xgrid_size,
              ysize = ygrid_size,
              asize = agrid_size)
T, get_policies, vfi = operator_factory(hh)

# == getting the value function and choices == #
v_star, vrefi_out, vnorefi_out, pol_refi, pol_norefi, error = vfi(T, tol=1e-4, max_iter=1000)
refichoice, achoice, cchoice = get_policies(vrefi_out, vnorefi_out, pol_refi, pol_norefi)

# == plotting figure IX == #
f_val=0
a_val=1
y_val=4
f, ax = plt.subplots(figsize = (9, 6))
ax.plot(1 - hh.xnodes, refichoice[a_val,y_val,:,0,0,f_val], label = 'Low R')
ax.plot(1 - hh.xnodes, refichoice[a_val,y_val,:,1,1,f_val], label = 'High R', linestyle = '--')
ax.plot(1 - hh.xnodes, refichoice[a_val,y_val,:,1,0,f_val], label = 'High to Low R', linestyle = '-.', color = 'yellow')
ax.set_title('Figure IX')
ax.legend()
plt.show()
f.savefig('./product/replication_files/figure_9.pdf')
