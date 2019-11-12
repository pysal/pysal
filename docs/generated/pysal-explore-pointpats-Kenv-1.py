import pysal.lib as ps
from pointpats import Kenv, PoissonPointProcess, Window
from pysal.lib.cg import shapely_ext
va = ps.io.open(ps.examples.get_path("vautm17n.shp"))
polys = [shp for shp in va]
state = shapely_ext.cascaded_union(polys)
pp = PoissonPointProcess(Window(state.parts), 100, 1, asPP=True).realizations[0]
csrs = PoissonPointProcess(pp.window, 100, 100, asPP=True)
kenv = Kenv(pp, realizations=csrs)
kenv.plot()
