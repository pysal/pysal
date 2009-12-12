from pysal import weights
# should give us everything we need to do weights

w_rook,observation_ids=weights.rook_from_shapefile("file.shp", id_variable_name)

w_queen,observation_ids=weights.queen_from_shapefile('file.shp', id_variable_name)

w_regime=weights.regime(regime_variable, observation_ids)



# set operations
# members of same regime and contiguous non-regime units
w_hybrid=w_regime.union(w_queen)

# members of same regime that are contiguous
w_regime_contiguities = w_regime.intersection(w_queen)

# contiguous but not in same regime = border connections
set_ids=set(observation_ids)
w_border = w_queen.intersection(set_ids.difference(w_regime))


# subsetting example
w_us_county_rook=weights.rook_from_shapefile("uscounties.shp")
# california_ids is an orded sequence - subsetting pulls out only cal-cal
# neighbor joins/weights and sets the ids of the resulting w to match
# that of california_ids
w_california_rook=w_us_county.subset(california_ids)


# aggregation
w_us_states_rook=w_us_county.aggregate(state_ids, observation_ids)

w_lattice = weights.lat2gal(10,10)

w_idist=weights.inverse_distance(data,p=2,row_standardization=True)

w_nn=weights.nearest_neighbors(data,k=2,p=2)

w_thresh=weights.threshold(data,threshold,p=2)

w_shared=weights.shared_boundary(shapefile)
# have to think about exterior polygons


# lag api
# observation_ids are necessary to ensure alignment of elements of y with
# weights
w_lattice.lag(y,observation_ids)


numpy_array_w,observation_ids = w_lattice.full(observation_ids=None)
numpy_array_w,observation_ids = w_lattice.full(observation_ids=observation_ids)


# saving
gal = pysal.open('my_lattice.gal','w')
gal.write(w_lattice)
gal.close()

gwt = pysal.open('my_lattice.gwt','w')
gwt.write(w_lattice)
gwt.close()

# reading
w_lattice=pysal.open('my_lattice.gal','r').read()

