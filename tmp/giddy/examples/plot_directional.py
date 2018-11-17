"""
=========================
Directional LISA Plotting
=========================

A directional LISA plot can be used to examine the spatial dynamics of a process.
By placing a local indicator of spatial association (LISA) in a dynamic context, insights on directional biases,
co-movements, space-time hot-spots (and cold-spots) can be generated.

In this example, we use data on state per-capita incomes for the lower 48 US states that we will first process:
"""

import libpysal as lps
import numpy as np
from giddy.directional import Rose
f = open(lps.examples.get_path('spi_download.csv'), 'r')
lines = f.readlines()
f.close()


lines = [line.strip().split(",") for line in lines]
names = [line[2] for line in lines[1:-5]]
data = np.array([list(map(int, line[3:])) for line in lines[1:-5]])


#############################
# We can omit the BEA regions and focus only on the lower 48 states
# and place incomes on relative terms:

sids  = range(60)
out = ['"United States 3/"',
      '"Alaska 3/"',
      '"District of Columbia"',
      '"Hawaii 3/"',
      '"New England"','"Mideast"',
       '"Great Lakes"',
       '"Plains"',
       '"Southeast"',
       '"Southwest"',
       '"Rocky Mountain"',
       '"Far West 3/"']

snames = [name for name in names if name not in out]
sids = [names.index(name) for name in snames]
states = data[sids,:]
us = data[0]
years = np.arange(1969, 2009)
rel = states/(us*1.)
Y = rel[:, [0, -1]]

###############################################################################
# Spatial Weights
# ---------------
#
# We will use a simple contiguity structure to define neighbors. The file
# states48.gal encodes the adjacency structure of the 48 states. We read this in
# and row-normalize the weights:

gal = lps.open(lps.examples.get_path('states48.gal'))
w = gal.read()
w.transform = 'r'

##########################################
# Visualization
# ==============
#
# The Rose class creates a circular histogram that can be used to examine the distribution
# of LISA Vectors across segments of the histogram:



r4 = Rose(Y, w, k=4)



##########################################
# LISA Vectors
# ------------
#
# The Rose class contains methods to carry out inference on the circular distribution of the LISA vectors. The first approach is based on a two-sided alternative where the null is that the distribution of the vectors across the segments reflects independence in the movements of the focal unit and its spatial lag. Inference is based on random spatial permutations under the null.



r4.plot_vectors() # lisa vectors




##########################################
# LISA Vectors Origin Standardized
# ================================
#
# As the LISA vectors combine the locations of a give LISA statistic in two different time periods, it can be useful
# to standardize the vectors to look for directional biases in the movements:


r4.plot_origin() # origin standardized


##########################################
# LISA Plot
# =========
#
# The Rose class contains methods to carry out inference on the circular distribution of the LISA vectors. The first approach is based on a two-sided alternative where the null is that the distribution of the vectors across the segments reflects independence in the movements of the focal unit and its spatial lag. Inference is based on random spatial permutations under the null.



r4.plot() # Polar

##########################################
# Conditional LISA Plot (Focal)
# =============================
#
# Here we condition on the relative starting income of the focal units:


r4.plot(attribute=Y[:,0]) # condition on starting relative income

##########################################
# Conditional LISA Plot (Spatial Lag)
# ===================================
#
# Here we condition on the relative starting income of the
# neighboring units:



r4.plot(attribute=r4.lag[:,0]) # condition on lag of starting relative income

##########################################
# Inference
# ==========
#
# The Rose class contains methods to carry out inference on the circular distribution of the LISA vectors. The first approach is based on a two-sided alternative where the null is that the distribution of the vectors across the segments reflects independence in the movements of the focal unit and its spatial lag. Inference is based on random spatial permutations under the null.


print(r4.cuts)
print(r4.counts)
np.random.seed(1234)
r4.permute(permutations=999)
print(r4.p)


################################################
# Here all the four sector counts are significantly different from their expectation under the null.
# A directional test can also be implemented. Here the direction of the departure from the null due to positive co-movement of a focal unit and its spatial lag over the time period results in two  two general cases. For sectors in the positive quadrants (I and III), the observed counts are considered extreme if they are larger than expectation, while for the negative quadrants (II, IV) the observed counts are considered extreme if they are small than the expected counts under the null.



r4.permute(alternative='positive', permutations=999)
print(r4.p)

######################################
# The expected values are:


print(r4.expected_perm)

######################################
# Finally, a directional alternative reflecting negative association between the movement of the focal unit and its lag has the complimentary interpretation to the positive alternative: lower counts in I and III, and higher counts in II and IV relative to the null.


r4.permute(alternative='negative', permutations=999)
print(r4.p)
