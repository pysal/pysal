import pysal as ps
import numpy as np

rook = np.loadtxt("ROOK_HR80.txt")
queen = np.loadtxt("QUEEN_HR80.txt")

common = (rook==1) * (queen==1)
queen_only = (queen==1) * (rook==0)
rook_only = (rook==1) * (queen==0)
common_nonsig = (rook==0) * (queen==0)

data = np.vstack((rook, queen, common,  rook_only, queen_only, common_nonsig)).T
