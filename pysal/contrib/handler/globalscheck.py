g1 = globals().keys()
from registry import *
g2 = globals().keys()
additions = [k for k in g2 if k not in g1]

print(additions)
