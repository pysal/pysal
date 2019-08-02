# Constructing data for illustration of directional LISA analytics.
# Data is for the 48 lower US states over the period 1969-2009 and
# includes per capita income normalized to the national average.

# Load comma delimited data file in and convert to a numpy array

import pysal.lib
from pysal.explore.giddy.directional import Rose
import matplotlib.pyplot as plt
file_path = pysal.lib.examples.get_path("spi_download.csv")
f=open(file_path,'r')
lines=f.readlines()
f.close()
lines=[line.strip().split(",") for line in lines]
names=[line[2] for line in lines[1:-5]]
data=np.array([list(map(int,line[3:])) for line in lines[1:-5]])

# Bottom of the file has regional data which we don't need for this
# example so we will subset only those records that match a state name

sids=list(range(60))
out=['"United States 3/"',
     '"Alaska 3/"',
     '"District of Columbia"',
     '"Hawaii 3/"',
     '"New England"',
     '"Mideast"',
     '"Great Lakes"',
     '"Plains"',
     '"Southeast"',
     '"Southwest"',
     '"Rocky Mountain"',
     '"Far West 3/"']
snames=[name for name in names if name not in out]
sids=[names.index(name) for name in snames]
states=data[sids,:]
us=data[0]
years=np.arange(1969,2009)

# Now we convert state incomes to express them relative to the national
# average

rel=states/(us*1.)

# Create our contiguity matrix from an external GAL file and row
# standardize the resulting weights

gal=pysal.lib.io.open(pysal.lib.examples.get_path('states48.gal'))
w=gal.read()
w.transform='r'

# Take the first and last year of our income data as the interval to do
# the directional directional analysis

Y=rel[:,[0,-1]]

# Set the random seed generator which is used in the permutation based
# inference for the rose diagram so that we can replicate our example
# results

np.random.seed(100)

# Call the rose function to construct the directional histogram for the
# dynamic LISA statistics. We will use four circular sectors for our
# histogram

r4=Rose(Y,w,k=4)

# What are the cut-offs for our histogram - in radians

r4.cuts
# array([0.        , 1.57079633, 3.14159265, 4.71238898, 6.28318531])

# How many vectors fell in each sector

r4.counts
# array([32,  5,  9,  2])

# We can test whether these counts are different than what would be
# expected if there was no association between the movement of the
# focal unit and its spatial lag.

# To do so we call the `permute` method of the object

r4.permute()

# and then inspect the `p` attibute:

r4.p
# array([0.04, 0.  , 0.02, 0.  ])

# Repeat the exercise but now for 8 rather than 4 sectors

r8 = Rose(Y, w, k=8)
r8.counts
# array([19, 13,  3,  2,  7,  2,  1,  1])
r8.permute()
r8.p
# array([0.86, 0.08, 0.16, 0.  , 0.02, 0.2 , 0.56, 0.  ])

# The default is a two-sided alternative. There is an option for a
# directional alternative reflecting positive co-movement of the focal
# series with its spatial lag. In this case the number of vectors in
# quadrants I and III should be much larger than expected, while the
# counts of vectors falling in quadrants II and IV should be much lower
# than expected.

r8.permute(alternative='positive')
r8.p
# array([0.51, 0.04, 0.28, 0.02, 0.01, 0.14, 0.57, 0.03])

# Finally, there is a second directional alternative for examining the
# hypothesis that the focal unit and its lag move in opposite directions.

r8.permute(alternative='negative')
r8.p
# array([0.69, 0.99, 0.92, 1.  , 1.  , 0.97, 0.74, 1.  ])

# We can call the plot method to visualize directional LISAs as a
# rose diagram conditional on the starting relative income:

fig1, _ = r8.plot(attribute=Y[:,0])
plt.show(fig1)
