# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from mapping import Map_Projection, quantile_map, equal_interval_map, fisher_jenks_map, classless_map
import pysal as ps
import numpy as np
from matplotlib.pyplot import * # needed for later when running as a saved script

# <codecell>

nc = Map_Projection(ps.examples.get_path("sids2.shp"))
dbf = ps.open(ps.examples.get_path("sids2.dbf"))
sidr79 = np.array(dbf.by_col("SIDR74"))

# <codecell>

sidr79_q5 = quantile_map(nc,sidr79, k=5, title = "SIDR79 Quantiles k=5")

# <codecell>

sidr79_ei5 = equal_interval_map(nc,sidr79, k=5, title = "SIDR79 Equal Intervals k=5")

# <codecell>

sidr79_ei7 = equal_interval_map(nc,sidr79, k=7, title = "SIDR79 Equal Intervals k=7")

# <codecell>

maps = {}
for k in range(3,9):
    maps[('q',k)] = quantile_map(nc,sidr79, k=k, title = "SIDR79 Quantiles k=%d"%k)
    maps[('ei',k)] = equal_interval_map(nc,sidr79, k=k, title = "SIDR79 Equal Intervals: k=%d"%k)

# <codecell>

clm = classless_map(nc, sidr79, title='SIDR79 Classless')

# <codecell>

south = Map_Projection("data/south.shp")
dbf = ps.open("data/south.dbf")
HR90 = np.array(dbf.by_col("HR90"))

# <codecell>

hr90_clm = classless_map(south, HR90, title="HR90 Classless")

# <codecell>

hr90_q5 = quantile_map(south, HR90, k=5, title="HR90 Quantiles k=5")

# <codecell>

hr90_fjs5 = fisher_jenks_map(south, HR90, k=5, sampled=True, title="HR90 Fisher-Jenks Sampled k=5")

# <codecell>

hr90_fj5 = fisher_jenks_map(south, HR90, k=5, sampled=False, title="HR90 Fisher-Jenks k=5")

# <codecell>

hr90_fj5.tss

# <codecell>

hr90_fjs5.tss

# <codecell>

hr90_ei5 = equal_interval_map(south, HR90, k=5)

# <codecell>

hr90_ei5.tss

# <codecell>

hr90_q5.tss

# <codecell>

hr90_tss = HR90.var() * len(HR90)

# <codecell>

hr90_tss

# <headingcell level=2>

# Comparison of Classification Schemes

# <codecell>

tss = np.zeros((5,4))
i = 0
for k in range(4,9):
    tss[i,0] =  1 - equal_interval_map(south, HR90, k=k, title="EI k=%d"%k).tss / hr90_tss
    tss[i,1] = 1 - quantile_map(south, HR90, k=k, title="Quantiles k=%d"%k).tss / hr90_tss
    tss[i,2] = 1 - fisher_jenks_map(south, HR90, k=k, sampled=True, title="FJ Sampled k=%d"%k).tss / hr90_tss
    tss[i,3] = 1 - fisher_jenks_map(south, HR90, k=k, sampled=False, title="FJ k=%d"%k).tss / hr90_tss
    i += 1

# <codecell>


xs = range(4,9)
ax = subplot(111)
plot(xs, tss[:,0], label = 'EI')
plot(xs, tss[:,1], label = 'Quantiles')
plot(xs, tss[:,2], label = 'Fisher-Jenks Sampled')
plot(xs, tss[:,3], label = 'Fisher-Jenks')
ylabel('Fit')
xlabel('k')
xa = ax.get_xaxis()
xa.set_major_locator(MaxNLocator(integer=True))
title('Classification Fit')
legend(loc=4)

# <markdowncell>

# $Fit = 1 - \sum_c\sum_{i \in c} (y_i - \bar{y}_c)^2   /   \sum_i (y_i - \bar{y})^2  $

