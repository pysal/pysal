"""
shared_perimeter_weights -- calculate shared perimeters weights....

wij = l_ij/P_i
wji = l_ij/P_j

l_ij = length of shared border i and j
P_j = perimeter of j

"""
__author__ = "Charles R Schmidt <schmidtc@gmail.com>"
__all__ = ["spw_from_shapefile"]


import pysal
import shapely.geometry

def spw_from_shapefile(shapefile, idVariable=None):
    polygons = pysal.open(shapefile,'r').read()
    polygons = map(shapely.geometry.asShape,polygons)
    perimeters = [p.length for p in polygons]
    Wsrc = pysal.rook_from_shapefile(shapefile)
    new_weights = {}
    for i in Wsrc.neighbors:
        a = polygons[i]
        p = perimeters[i]
        new_weights[i] = [a.intersection(polygons[j]).length/p for j in Wsrc.neighbors[i]]
    return pysal.W(Wsrc.neighbors,new_weights)

if __name__=='__main__':
    fname = pysal.examples.get_path('stl_hom.shp')
    W = spw_from_shapefile(fname)

