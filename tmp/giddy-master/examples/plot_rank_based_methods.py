"""
===================
Rank based Methods 
===================

This notebook introduces two classic nonparametric statistics of exchange mobility and their spatial extensions. We will demonstrate the usage of these methods by an empirical study for understanding [regional exchange mobility pattern in US](#Regional-exchange-mobility-pattern-in-US-1929-2009). The dataset is the per capita incomes observed annually from 1929 to 2010 for the lower 48 US states.
 
 * [Kendall's $\tau$](#Kendall's-$\tau$)
     * Classic measures:
         * [Classic Kendall's $\tau$](#Classic-Kendall's-$\tau$)
         * [Local Kendall's $\tau$](#Local-Kendall's-$\tau$)
     * Spatial extensions:
         * [Spatial Kendall's $\tau$](#Spatial-Kendall's-$\tau$)
         * [Inter- and Intra-regional decomposition of Kendall's $\tau$](#Inter--and-Intra-regional-decomposition-of-Kendall's-$\tau$)
         * [Local indicator of mobility association-LIMA](#Local-indicator-of-mobility-association-LIMA)
 * [$\Theta$ statistic of exchange mobility](#$\Theta$-statistic-of-exchange-mobility)
"""

# ## Regional exchange mobility pattern in US 1929-2009
# 
# Firstly we load in the US dataset:

# In[1]:


import libpysal
import libpysal.api as ps
import libpysal.io.geotable as pdio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

import giddy
from splot import mapping as maps

shp_link = libpysal.examples.get_path('us48.shp')
data_table = pdio.read_files(shp_link)
income_table = pd.read_csv(libpysal.examples.get_path("usjoin.csv"))
complete_table = data_table.merge(income_table,left_on='STATE_NAME',right_on='Name')
complete_table.head()


"""
We will visualize the spatial distributions of per capita incomes in US states across 1929 to 2009 to obtain a first impression of the dynamics.

"""

index_year = range(1929,2010,15)
fig, axes = plt.subplots(nrows=2, ncols=3,figsize = (15,7))
for i in range(2):
    for j in range(3):
        ax = axes[i,j]
        maps.geoplot(complete_table, col=str(index_year[i*3+j]),ax=ax,classi="Quantiles")
        ax.set_title('Per Capita Income %s Quintiles'%str(index_year[i*3+j]))
plt.tight_layout()


# In[3]:


years = range(1929,2010)
names = income_table['Name']
pci = income_table.drop(['Name','STATE_FIPS'], 1).as_matrix().T
rpci= (pci.T / pci.mean(axis=1)).T
order1929 = np.argsort(rpci[0,:])
order2009 = np.argsort(rpci[-1,:])
names1929 = names[order1929[::-1]]
names2009 = names[order2009[::-1]]
first_last = np.vstack((names[order1929[::-1]],names[order2009[::-1]]))
rcParams['figure.figsize'] = 15,10
p = plt.plot(years,rpci)
for i in range(48):
    plt.text(1915,1.91-(i*0.041), names1929[i],fontsize=12)
    plt.text(2010.5,1.91-(i*0.041), names2009[i],fontsize=12)
plt.xlim((years[0], years[-1]))
plt.ylim((0, 1.94))
plt.ylabel(r"$y_{i,t}/\bar{y}_t$",fontsize=14)
plt.xlabel('Years',fontsize=12)
plt.title('Relative per capita incomes of 48 US states',fontsize=18)


# The above figure displays the trajectories of relative per capita incomes of 48 US states. It is quite obvious that states were swapping positions across 1929-2009. We will demonstrate how to quantify the exchange mobility as well as how to assess the regional and local contribution to the overall exchange mobility. We will ultilize [BEA regions](https://www.bea.gov/regional/docs/regions.cfm) and base on it for constructing the block weight matrix. 
# 
# BEA regional scheme divide US states into 8 regions:
# * New England Region
# * Mideast Region
# * Great Lakes Region
# * Plains Region
# * Southeast Region
# * Southwest Region
# * Rocky Mountain Region
# * Far West Region
# 
# As the dataset does not contain information regarding BEA regions, we manually input the regional information:

# In[4]:


BEA_regions = ["New England Region","Mideast Region","Great Lakes Region","Plains Region","Southeast Region","Southwest Region","Rocky Mountain Region","Far West Region"]
BEA_regions_abbr = ["NENG","MEST","GLAK","PLNS","SEST","SWST","RKMT","FWST"]
BEA = pd.DataFrame({ 'Region code' : np.arange(1,9,1), 'BEA region' : BEA_regions,'BEA abbr':BEA_regions_abbr})
BEA


# In[5]:


region_code = list(np.repeat(1,6))+list(np.repeat(2,6))+list(np.repeat(3,5))+list(np.repeat(4,7))+list(np.repeat(5,12))+list(np.repeat(6,4))+list(np.repeat(7,5))+list(np.repeat(8,6))
state_code = ['09','23','25','33','44','50','10','11','24','34','36','42','17','18','26','39','55','19','20','27','29','31','38','46','01','05','12','13','21','22','28','37','45','47','51','54','04','35','40','48','08','16','30','49','56','02','06','15','32','41','53']
state_region = pd.DataFrame({'Region code':region_code,"State code":state_code})
state_region_all = state_region.merge(BEA,left_on='Region code',right_on='Region code')
complete_table = complete_table.merge(state_region_all,left_on='STATE_FIPS_x',right_on='State code')
complete_table.head()


# The BEA regions are visualized below:

# In[6]:


maps.plot_choropleth(shp_link, complete_table["BEA region"], "unique_values", title="BEA Regions")


# ## Kendall's $\tau$
# 
# Kendall’s $\tau$ statistic is based on a comparison of the number of pairs of $n$ observations that have concordant ranks between two variables. For measuring exchange mobility in **giddy**, the two variables in question are the values of an attribute measured at two points in time over $n$ spatial units. This classic measure of rank correlation indicates how much relative stability there has been in the map pattern over the two periods. Spatial decomposition of Kendall’s $\tau$ could be classified into three spatial scales: global spatial decomposition , inter- and intra-regional decomposition and local spatial decomposition. More details will be given latter.

# ### Classic Kendall's $\tau$
# 
# Kendall's $\tau$ statistic is a global measure of exchange mobility. For $n$ spatial units over two periods, it is formally defined as follows:
# 
# $$\tau = \frac{c-d}{(n(n-1))/2}$$
# 
# where $c$ is the number of concordant pairs (two spatial units which do not exchange ranks over two periods), and $d$ is the number of discordant pairs (two spatial units which exchange ranks over two periods). $-1 \leq \tau \leq 1$. Smaller $\tau$ indicates higher exchange mobility.

# In giddy, class $Tau$ requires two inputs: a cross-section of income values at one period ($x$) and a cross-section of income values at another period ($y$):
# 
# ```python
# giddy.rank.Tau(self, x, y)
# ```
# 
# We will construct a $Tau$ instance by specifying the incomes in two periods. Here, we look at the global exchange mobility of US states between 1929 and 2009.

# In[7]:




# In[8]:


tau = giddy.rank.Tau(complete_table["1929"],complete_table["2009"])
tau


# In[9]:


tau.concordant


# In[10]:


tau.discordant


# There are 856 concordant pairs of US states between 1929 and 2009, and 271 discordant pairs.

# In[11]:


tau.tau


# In[12]:


tau.tau_p


# The observed Kendall's $\tau$ statistic is 0.519 and its p-value is $1.974 \times 10^{-7}$. Therefore, we will reject the null hypothesis of no assocation between 1929 and 2009 at the $5\%$ significance level.

# ### Spatial Kendall's $\tau$
# 
# The spatial Kendall's $\tau$ decomposes all pairs into those that are spatial neighbors and those that are not, and examines whether the rank correlation is different between the two sets (Rey, 2014). 
# 
# $$\tau_w = \frac{\iota'(W\circ S)\iota}{\iota'W \iota}$$
# 
# $W$ is the spatial weight matrix, $S$ is the concordance matrix and $\iota$ is the $(n,1)$ unity vector. The null hypothesis is the spatial randomness of rank exchanges. The inference of $\tau_w$ could be conducted based on random spatial permutation of incomes at two periods. 

# ```python
# giddy.rank.SpatialTau(self, x, y, w, permutations=0)
# ```
# For illustration, we turn back to the case of incomes in US states over 1929-2009:

# In[13]:




# In[14]:


w = ps.block_weights(complete_table["BEA region"])
np.random.seed(12345)
tau_w = giddy.rank.SpatialTau(complete_table["1929"],complete_table["2009"],w,999) 


# In[15]:


tau_w.concordant


# In[16]:


tau_w.concordant_spatial


# In[17]:


tau_w.discordant


# In[18]:


tau_w.discordant_spatial


# Out of 856 concordant pairs of spatial units, 103 belong to the same region (and are considered neighbors); out of 271 discordant pairs of spatial units, 41 belong to the same region.

# In[19]:


tau_w.tau_spatial


# In[20]:


tau_w.tau_spatial_psim


# The estimate of spatial Kendall's $\tau$ is 0.431 and its p-value is 0.001 which is much smaller than the significance level $0.05$. Therefore, we reject the null of spatial randomness of exchange mobility. The fact that $\tau_w=0.431$  is smaller than the global average $\tau=0.519$ implies that globally a significant number of rank exchanges happened between states within the same region though we do not know the specific region or regions hosting these rank exchanges. A more thorough decomposition of $\tau$ such as inter- and intra-regional indicators and local indicators will provide insights on this issue.

# ### Inter- and Intra-regional decomposition of Kendall's $\tau$
# 
# A meso-level view on the exchange mobility pattern is provided by inter- and intra-regional decomposition of Kendall's $\tau$. This decomposition can shed light on specific regions hosting most rank exchanges. More precisely, insteading of examining the concordance relationship between any two neighboring spatial units in the whole study area, for a specific region A, we examine the concordance relationship between any two spatial units within region A (neighbors), resulting in the intraregional concordance statistic for A; or we could examine the concordance relationship between any spatial unit in region A and any spatial unit in region B (nonneighbors), resulting in the interregional concordance statistic for A and B. If there are k regions, there will be k intraregional concordance statistics and $(k-1)^2$ interregional concordance statistics, we could organize them into a $(k,k)$ matrix where the diagonal elements are intraregional concordance statistics and nondiagnoal elements are interregional concordance statistics.
# 
# Formally, this inter- and intra-regional concordance statistic matrix is defined as follows (Rey, 2016):
# 
# $$T=\frac{P(H \circ S)P'}{P H P'}$$
# 
# $P$ is a $(k,n)$ binary matrix where $p_{j,i}=1$ if spatial unit $i$ is in region $j$ and $p_{j,i}=0$ otherwise. $H$ is a $(n,n)$ matrix with 0 on diagnoal and 1 on other places. $\circ$ is the Hadamard product. Inference could be based on random spatial permutation of incomes at two periods, similar to spatial $\tau$. 

# To obtain an estimate for the inter- and intra-regional indicator matrix, we use the $Tau\_Regional$ class:
# ```python
# giddy.rank.Tau_Regional(self, x, y, regime, permutations=0)
# ```
# Here, $regime$ is an 1-dimensional array of size n. Each element is the id of which region an spatial unit belongs to.

# In[21]:




# Similar to before, we go back to the case of incomes in US states over 1929-2009:

# In[22]:


np.random.seed(12345)
tau_w = giddy.rank.Tau_Regional(complete_table["1929"],complete_table["2009"],complete_table["BEA region"],999) 
tau_w


# In[23]:


tau_w.tau_reg


# The attribute $tau\_reg$ gives the inter- and intra-regional concordance statistic matrix. Higher values represents lower exchange mobility. Obviously there are some negative values indicating high exchange mobility. Attribute $tau\_reg\_pvalues$ gives pvalues for all inter- and intra-regional concordance statistics: 

# In[24]:


tau_w.tau_reg_pvalues


# We can manipulate these two attribute to obtain significant inter- and intra-regional statistics only (at the $5\%$ significance level):

# In[25]:


tau_w.tau_reg * (tau_w.tau_reg_pvalues<0.05)


# The table below displays the inter- and intra-regional decomposition matrix of Kendall's $\tau$ for US states over 1929-2009 based on the 8 BEA regions. Bold numbers indicate significance at the $5\%$ significance level. The negative and significant intra-Southeast concordance statistic ($-0.486$) indicates that the rank exchanges within Southeast region is significantly more frequent than those between states within and out of Southeast region.

# | Region        | New England| Mideast|Great Lakes|Plains|Southeast|Southwest|Rocky Mountain|Far West|
# |:-------------:|:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
# | New England  | 0.667|  0.5 | 0.3|0.417|  0.2856|0.5 | 0.792|  0.875|
# | Mideast      | 0.5 |  0.4|0.52|0.267| **-0.486**|0.52| 0.533| 0.6 |
# | Great Lakes |  0.3 |  0.52 |  0  |  0.4 |  **0.886**| 0.76 | **0.933**|1.|
# |Plains| 0.417| 0.267|  0.4 |  0.867|  0.476|**0.833**| **0.861**| **0.917**|
# |Southeast|0.286|**-0.486**|**0.886**| 0.476| **-0.143**|0.429| 0.690| 0.143|
# |Southwest| 0.5 |0.52 |0.76|**0.833**| 0.429|0.8|**0.067**|0.1|
# |Rocky Mountain|0.792| 0.533| **0.933**|**0.861**| 0.69|**0.067**| 0.545|0.333|
# |Far West|0.875|0.6| 1.| **0.917**|0.143|0.1 |0.333| 0|

# ### Local Kendall's $\tau$
# 
# Local Kendall's $\tau$ is a local decomposition of classic Kendall's $\tau$ which provides an indication of the contribution of spatial unit $r$’s rank changes to the overall level of exchange mobility (Rey, 2016). Focusing on spatial unit $r$, we formally define it as follows:
# $$\tau_{r} = \frac{c_r - d_r}{n-1}$$
# 
# where $c_r$ is the number of spatial units (except $r$) which are concordant with $r$ and $d_r$ is the number of spatial units which are discordant with $r$. Similar to classic Kendall's $\tau$, local $\tau$ takes values on $[-1,1]$. The larger the value, the lower the exchange mobility for $r$.

# ```python
# giddy.rank.Tau_Local(self, x, y)
# ```

# In[28]:




# We create a $Tau\_Local$ instance for US dynamics 1929-2009:

# In[29]:


tau_r = giddy.rank.Tau_Local(complete_table["1929"],complete_table["2009"])
tau_r


# In[48]:


pd.DataFrame({"STATE_NAME":complete_table['STATE_NAME'].tolist(),"$\\tau_r$":tau_r.tau_local}).head()


# Therefore, local concordance measure produces a negative value for North Dakota (-0.0213) indicating that North Dakota exchanged ranks with a lot of states across 1929-2000. On the contrary, the local $\tau$ statistic is quite high for Washington (0.617) highlighting a high stability of Washington.

# ### Local indicator of mobility association-LIMA
# 
# To reveal of the role of space in shaping the exchange mobility pattern for each spatial unit, two spatial variants of local Kendall's $\tau$ could be utilized: neighbor set LIMA and neighborhood set LIMA (Rey, 2016). The latter is also the result of a decomposition of local Kendall's $\tau$ (into neighboring and nonneighboring parts) as well as a decompostion of spatial Kendall's $\tau$ (into its local components).

# #### Neighbor set LIMA
# 
# Instead of examining the concordance relationship between a focal spatial unit $r$ and all the other units as what local $\tau$ does, neighbor set LIMA focuses on the concordance relationship between a focal spatial unit $r$ and its neighbors only. It is formally defined as follows:
# 
# $$\tilde{\tau}_{r} = \frac{\sum_b w_{r,b} s_{r,b}}{\sum_b w_{r,b}}$$

# ```python
# giddy.rank.Tau_Local_Neighbor(self, x, y, w, permutations=0)
# ```

# In[49]:




# In[53]:


tau_wr = giddy.rank.Tau_Local_Neighbor(complete_table["1929"],complete_table["2009"],w,999) 
tau_wr


# In[54]:


tau_wr.tau_ln


# To visualize the spatial distribution of neighbor set LIMA:

# In[72]:


maps.plot_choropleth(shp_link, tau_wr.tau_ln, "equal_interval",title="Neighbor set LIMA for US states 1929-2009")


# Therefore, Arizona, North Dakota, and Missouri exchanged ranks with most of their neighbors over 1929-2009 while California, Virginia etc. barely exchanged ranks with their neighbors.

# Let see whether neighbor set LIMA statistics are siginificant for these "extreme" states:

# In[55]:


tau_wr.tau_ln_pvalues


# In[81]:


sig_wr = tau_wr.tau_ln * (tau_wr.tau_ln_pvalues<0.05)
sig_wr


# In[82]:


maps.plot_choropleth(shp_link, sig_wr, "unique_values",title="Significant Neighbor set LIMA for US states 1929-2009")


# Thus, Arizona and Missouri have significant and negative neighbor set LIMA values, and can be considered as hotspots of rank exchanges. This means that Arizona (or Missouri) tended to exchange ranks with its neighbors than with others over 1929-2009. On the contrary, Virgina has significant and large positive neighbor set LIMA value indicating that it tended to exchange ranks with its nonneighbors than with 

# #### Neighborhood set LIMA
# 
# Neighborhood set LIMA extends neighbor set LIMA $\tilde{\tau}_{r}$ to consider the concordance relationships between any two spatial units in the subset which is composed of the focal unit $r$ and its neighbors.

# ```python
# giddy.rank.Tau_Local_Neighborhood(self, x, y, w, permutations=0)
# ```

# In[51]:




# In[57]:


tau_wwr = giddy.rank.Tau_Local_Neighborhood(complete_table["1929"],complete_table["2009"],w,999) 
tau_wwr


# In[58]:


tau_wwr.tau_lnhood


# In[59]:


tau_wwr.tau_lnhood_pvalues


# In[60]:


tau_wwr.tau_lnhood * (tau_wwr.tau_lnhood_pvalues<0.05)


# ## $\Theta$ statistic of exchange mobility

# ## Next steps
# 
# * theta statistic

# ## References
# * Rey, Sergio J., and Myrna L. Sastré-Gutiérrez. 2010. “[Interregional Inequality Dynamics in Mexico](http://www.tandfonline.com/doi/abs/10.1080/17421772.2010.493955).” Spatial Economic Analysis 5 (3). Taylor & Francis: 277–98.
# * Rey, Sergio J. 2014. “[Fast Algorithms for a Space-Time Concordance Measure](https://link.springer.com/article/10.1007/s00180-013-0461-2).” Computational Statistics 29 (3-4). Springer: 799–811.
# * Rey, Sergio J. 2016. “[Space--Time Patterns of Rank Concordance: Local Indicators of Mobility Association with Application to Spatial Income Inequality Dynamics](http://www.tandfonline.com/doi/abs/10.1080/24694452.2016.1151336?journalCode=raag21).” Annals of the Association of American Geographers. Association of American Geographers 106 (4): 788–803.
