Overall, there were 1580 commits that closed 355 issues, together with 226 pull requests since our last release on 2017-11-03.

## Changes by package

### libpysal:
* weights.distance.KNN.from_dataframe ignoring radius  [(#116)](https://github.com/pysal/libpysal/issues/116)
* Always make spherical KDTrees if radius is passed [(#117)](https://github.com/pysal/libpysal/pull/117)
* [ENH] should `weights.util.get_ids()` also accept a geodataframe? [(#97)](https://github.com/pysal/libpysal/issues/97)
* enh: add doctests to travis (#2) [(#112)](https://github.com/pysal/libpysal/pull/112)
* sphinx docs need updating [(#49)](https://github.com/pysal/libpysal/issues/49)
* Add notebooks for subpackage contract [(#108)](https://github.com/pysal/libpysal/issues/108)
* Api docs complete [(#110)](https://github.com/pysal/libpysal/pull/110)
* Doctests and start of documentation for libpysal [(#109)](https://github.com/pysal/libpysal/pull/109)
* Add dependencies to requirements_plus.txt for test_db [(#107)](https://github.com/pysal/libpysal/pull/107)
* Weights/util/get ids gdf [(#101)](https://github.com/pysal/libpysal/pull/101)
* missing adjustments to lower case module names [(#106)](https://github.com/pysal/libpysal/pull/106)
* Rel.4.0.0 [(#105)](https://github.com/pysal/libpysal/pull/105)
* REL: 3.0.8 [(#104)](https://github.com/pysal/libpysal/pull/104)
* error importing v3.0.7 [(#100)](https://github.com/pysal/libpysal/issues/100)
* Lower case module names [(#98)](https://github.com/pysal/libpysal/pull/98)
* remove function regime_weights [(#96)](https://github.com/pysal/libpysal/pull/96)
* depreciating regime_weights in the new release? [(#94)](https://github.com/pysal/libpysal/issues/94)
* inconsistency in api? [(#93)](https://github.com/pysal/libpysal/issues/93)
* Ensure consistency in `from .module import *` in components of libpysal [(#95)](https://github.com/pysal/libpysal/pull/95)
* [WIP] cleanup [(#88)](https://github.com/pysal/libpysal/pull/88)
* docstrings for attributes are defined in properties [(#87)](https://github.com/pysal/libpysal/pull/87)
* docstrings in W class need editing [(#64)](https://github.com/pysal/libpysal/issues/64)
* version name as __version__ [(#92)](https://github.com/pysal/libpysal/pull/92)
* remove `del` statements and modify alphashape __all__ [(#89)](https://github.com/pysal/libpysal/pull/89)
* libpysal/libpysal/cg/__init__.py not importing `rtree` [(#90)](https://github.com/pysal/libpysal/issues/90)
* including rtree in imports [(#91)](https://github.com/pysal/libpysal/pull/91)
* BUG:  test_weights_IO.py is using pysal and hard-coded paths [(#85)](https://github.com/pysal/libpysal/issues/85)
* fix hardcoded swm test [(#86)](https://github.com/pysal/libpysal/pull/86)
* check for spatial index if nonplanar neighbors [(#84)](https://github.com/pysal/libpysal/pull/84)
* nonplanar_neighbors fails when sindex is not constructed.  [(#63)](https://github.com/pysal/libpysal/issues/63)
* increment version number and add bugfixes, api changes [(#79)](https://github.com/pysal/libpysal/pull/79)
* Spherebug [(#82)](https://github.com/pysal/libpysal/pull/82)
* only warn once for islands/disconnected components [(#83)](https://github.com/pysal/libpysal/pull/83)
* only warn on disconnected components if there are no islands [(#81)](https://github.com/pysal/libpysal/issues/81)
* LEP: Stuff/use pysal/network stuff to provide queen weights on linestring dataframes [(#59)](https://github.com/pysal/libpysal/issues/59)
* swm fix not ported forward from pysal.  [(#66)](https://github.com/pysal/libpysal/issues/66)
* import scipy syntax typo in the new issue template [(#68)](https://github.com/pysal/libpysal/issues/68)
* deletion of extra spaces in warning message [(#78)](https://github.com/pysal/libpysal/pull/78)
* Nightli.es build permissions [(#77)](https://github.com/pysal/libpysal/issues/77)
* name of geometry column is hardcoded in nonplanar_neighbors [(#75)](https://github.com/pysal/libpysal/issues/75)
* changed geometry column name from a str to an attribute [(#76)](https://github.com/pysal/libpysal/pull/76)
* Missing example file  [(#71)](https://github.com/pysal/libpysal/issues/71)
* if numba isn't present, libpysal warns every time imported [(#73)](https://github.com/pysal/libpysal/issues/73)
* add check for disconnected components [(#65)](https://github.com/pysal/libpysal/pull/65)
* clean up for release [(#74)](https://github.com/pysal/libpysal/pull/74)
* update for new examples [(#72)](https://github.com/pysal/libpysal/pull/72)
* Swm [(#70)](https://github.com/pysal/libpysal/pull/70)
* Remaining concerns left unfixed in #61 [(#62)](https://github.com/pysal/libpysal/pull/62)
* [WIP] Alpha shapes (2D) code [(#58)](https://github.com/pysal/libpysal/pull/58)
* [WIP]: add linestring/multilinestring functionality [(#61)](https://github.com/pysal/libpysal/pull/61)
* Fuzzy contiguity [(#57)](https://github.com/pysal/libpysal/pull/57)
* add berlin example [(#56)](https://github.com/pysal/libpysal/pull/56)
* force UTF8 encoding for the long description read [(#55)](https://github.com/pysal/libpysal/pull/55)
* add guerry example dataset [(#45)](https://github.com/pysal/libpysal/pull/45)
* update georgia shapefile [(#53)](https://github.com/pysal/libpysal/pull/53)
* fix typo add `non_planar_joins` instead of `non_planar_neighbor` attr… [(#54)](https://github.com/pysal/libpysal/pull/54)
* add voronoi to the API [(#46)](https://github.com/pysal/libpysal/pull/46)
* ENH: Neighbor detection for nonplanar enforced polygon collections. [(#51)](https://github.com/pysal/libpysal/pull/51)
* Missing example used by gwr tests [(#43)](https://github.com/pysal/libpysal/pull/43)
* Wplot [(#50)](https://github.com/pysal/libpysal/pull/50)
* close the door on 2 for libpysal [(#44)](https://github.com/pysal/libpysal/pull/44)
* To networkx argument name changed [(#40)](https://github.com/pysal/libpysal/issues/40)
* bump micro version [(#42)](https://github.com/pysal/libpysal/pull/42)
* fix networkx adapters [(#41)](https://github.com/pysal/libpysal/pull/41)
* minor version bump for release [(#39)](https://github.com/pysal/libpysal/pull/39)
* Plot weights [(#38)](https://github.com/pysal/libpysal/pull/38)
* forward port of legacy fix #1028 [(#37)](https://github.com/pysal/libpysal/pull/37)
* Adding Voronoi generator for cg as well as Voronoi weights from 2-d points [(#36)](https://github.com/pysal/libpysal/pull/36)
* bump version for release [(#34)](https://github.com/pysal/libpysal/pull/34)
* attach_islands correction: incorporate pr #32 [(#33)](https://github.com/pysal/libpysal/pull/33)
* change data type of contiguity W.neighbors value from set to list (consistent with other weights) [(#32)](https://github.com/pysal/libpysal/pull/32)
* add a function to attach the nearest neighbor to island [(#30)](https://github.com/pysal/libpysal/pull/30)
* fix id2i lookup for string names and make better names [(#31)](https://github.com/pysal/libpysal/pull/31)
* two modules “Wsets.py” and "util.py" depend on each other [(#26)](https://github.com/pysal/libpysal/issues/26)
* add unittest for Wset.w_clip [(#29)](https://github.com/pysal/libpysal/pull/29)
* resolve circular import in Wsets and util [(#28)](https://github.com/pysal/libpysal/pull/28)
* update doctests in weights module to use libpysal instead of pysal [(#27)](https://github.com/pysal/libpysal/pull/27)
* bump stable date [(#25)](https://github.com/pysal/libpysal/pull/25)
* bump version for point release of lag cat fix [(#24)](https://github.com/pysal/libpysal/pull/24)

### esda:
* enh: updating travis build and rtd [(#40)](https://github.com/pysal/esda/pull/40)
* BUG: missing rtd file [(#39)](https://github.com/pysal/esda/pull/39)
* REL: 2.0.1 [(#38)](https://github.com/pysal/esda/pull/38)
* Prepping for a doc release [(#37)](https://github.com/pysal/esda/pull/37)
* docstrings are using pysal legacy [(#4)](https://github.com/pysal/esda/issues/4)
* add zenodo doi badge [(#36)](https://github.com/pysal/esda/pull/36)
* REL: 2.0.0 [(#34)](https://github.com/pysal/esda/pull/34)
* Changing esda setup to handle version programatically [(#33)](https://github.com/pysal/esda/pull/33)
* port legacy esda fix for 1013  [(#12)](https://github.com/pysal/esda/issues/12)
* notebook links broken [(#29)](https://github.com/pysal/esda/issues/29)
* include /tests in release [(#32)](https://github.com/pysal/esda/pull/32)
* Add tests to release [(#27)](https://github.com/pysal/esda/issues/27)
* Accounting for incoming API changes to `libpysal` and adding testing against `libpysal`'s master branch [(#26)](https://github.com/pysal/esda/pull/26)
* no `varnames` in `Moran_BV_matrix` [(#22)](https://github.com/pysal/esda/issues/22)
* add `.varnames` attribute to `Moran_BV` objects in `Moran_BV_Matrix` results [(#23)](https://github.com/pysal/esda/pull/23)
* Inconsistent metadata in setup.py [(#28)](https://github.com/pysal/esda/issues/28)
* Update license [(#30)](https://github.com/pysal/esda/pull/30)
* esda has no readme [(#14)](https://github.com/pysal/esda/issues/14)
* Readme added [(#25)](https://github.com/pysal/esda/pull/25)
* move api into __init__ and remove api [(#24)](https://github.com/pysal/esda/pull/24)
* `.z` attribute divided by standard deviation [(#21)](https://github.com/pysal/esda/pull/21)
* offer standardised and non-standardised `.z`, `.x` and `.y` attributes [(#20)](https://github.com/pysal/esda/issues/20)
* get sjsrey's changes into a release [(#16)](https://github.com/pysal/esda/pull/16)
* esda's namespace is broken [(#17)](https://github.com/pysal/esda/issues/17)
* update api.py [(#18)](https://github.com/pysal/esda/pull/18)
* Update docstrings to use libpysal not pysal [(#13)](https://github.com/pysal/esda/pull/13)
* esda needs an api.py module [(#9)](https://github.com/pysal/esda/issues/9)
* chore: Update setup for 3+ [(#15)](https://github.com/pysal/esda/pull/15)
* Master [(#10)](https://github.com/pysal/esda/pull/10)

### giddy:
* remove giddy.api in README.rst [(#66)](https://github.com/pysal/giddy/pull/66)
*  chore: update for libpysal lower case module name changes [(#65)](https://github.com/pysal/giddy/pull/65)
* remove api.py [(#62)](https://github.com/pysal/giddy/issues/62)
* set up travis dual testing against mapclassify and  esda [(#63)](https://github.com/pysal/giddy/issues/63)
* replace `libpysal.api` imports with new imports in `markov.py` and `d… [(#61)](https://github.com/pysal/giddy/pull/61)
* Remove api.py and account for changes in (incoming) API of mapclassify, esda, and libpysal [(#64)](https://github.com/pysal/giddy/pull/64)
* version giddy only in giddy/__ini__.py [(#60)](https://github.com/pysal/giddy/pull/60)
* remove duplicate makefile for sphinx build [(#59)](https://github.com/pysal/giddy/pull/59)
* add zenodo doi badge to README [(#58)](https://github.com/pysal/giddy/pull/58)
* add changelog for the release 1.2.0 [(#57)](https://github.com/pysal/giddy/pull/57)
* prepare for release 1.2.0 [(#56)](https://github.com/pysal/giddy/pull/56)
* set up dual travis tests for libpysal (pip and github) [(#55)](https://github.com/pysal/giddy/pull/55)
* ENH: Allow for more flexible specification of Spatial Markov [(#54)](https://github.com/pysal/giddy/pull/54)
* Update notebooks to rely on geopandas for mapping [(#52)](https://github.com/pysal/giddy/pull/52)
* ENH to docs [(#51)](https://github.com/pysal/giddy/pull/51)
* include /tests in the release and correct for the directional doctests [(#50)](https://github.com/pysal/giddy/pull/50)
* add doc building badge to README [(#49)](https://github.com/pysal/giddy/pull/49)
* Tests and documentation for `rose.plot()` and `rose.plot_vectors()` [(#47)](https://github.com/pysal/giddy/pull/47)
* A tentative version of giddy documentation website with sphinx  [(#48)](https://github.com/pysal/giddy/pull/48)
* encoding issue in  README.rst [(#45)](https://github.com/pysal/giddy/issues/45)
* force utf8 for the install description read [(#46)](https://github.com/pysal/giddy/pull/46)
* implement `rose.plot()` and `rose.plot_vectors()` method using `splot` [(#43)](https://github.com/pysal/giddy/pull/43)
* More on building doc webpages using sphinx [(#44)](https://github.com/pysal/giddy/pull/44)
* Gallery [(#42)](https://github.com/pysal/giddy/pull/42)
* new features for sphinx documentation website [(#41)](https://github.com/pysal/giddy/pull/41)
* typo - email notifications [(#40)](https://github.com/pysal/giddy/pull/40)
* fix for python 3 [(#38)](https://github.com/pysal/giddy/pull/38)
* first draft of sphinx gallery [(#39)](https://github.com/pysal/giddy/pull/39)
* add docstring for categorical spatial Markov [(#37)](https://github.com/pysal/giddy/pull/37)
* add changelog for the release 1.1.0 [(#36)](https://github.com/pysal/giddy/pull/36)
* prepare for release [(#35)](https://github.com/pysal/giddy/pull/35)
* code 2to3 [(#34)](https://github.com/pysal/giddy/pull/34)
* chore: update for python 3+ only [(#33)](https://github.com/pysal/giddy/pull/33)
* How to use the 'development' version [(#31)](https://github.com/pysal/giddy/issues/31)
* KeyError: 1 in spatial_lag.py [(#30)](https://github.com/pysal/giddy/issues/30)
* giddy needs an api.py module [(#26)](https://github.com/pysal/giddy/issues/26)
* add inequality to api [(#28)](https://github.com/pysal/giddy/pull/28)
* adding discretized Spatial_Markov [(#29)](https://github.com/pysal/giddy/pull/29)

### inequality:
* Change setup to handle version pragmatically [(#6)](https://github.com/pysal/inequality/pull/6)
* accounting for libpysal api changes in unittests [(#5)](https://github.com/pysal/inequality/pull/5)
* missing parenthesis in call to print [(#2)](https://github.com/pysal/inequality/issues/2)
* setting up dual testing [(#4)](https://github.com/pysal/inequality/pull/4)
* 2to3 for _indices.py [(#3)](https://github.com/pysal/inequality/pull/3)
* Initial setup [(#1)](https://github.com/pysal/inequality/pull/1)

### pointpats:
* add changelog for release 2.0.0 [(#21)](https://github.com/pysal/pointpats/pull/21)
* remove api.py & adjust notebooks and doctests for changes in libpysal [(#19)](https://github.com/pysal/pointpats/pull/19)
* version pointpats only in pointpats/__ini__.py [(#18)](https://github.com/pysal/pointpats/pull/18)
* include /tests  in the release [(#17)](https://github.com/pysal/pointpats/pull/17)
* configure dual testing [(#16)](https://github.com/pysal/pointpats/pull/16)
* install stable released libpysal for travis testing [(#15)](https://github.com/pysal/pointpats/pull/15)
* force UTF8 encoding for the long description read [(#14)](https://github.com/pysal/pointpats/pull/14)
* Prepare for the release  [(#13)](https://github.com/pysal/pointpats/pull/13)
* chore: libpysal is 3 only now so removing travis tests on python 2 [(#12)](https://github.com/pysal/pointpats/pull/12)
* try removing conversion and see if this passes [(#11)](https://github.com/pysal/pointpats/pull/11)

### spaghetti:
* refreshing documentation [(#124)](https://github.com/pysal/spaghetti/pull/124)
* option to add distance from point to snapped location [(#75)](https://github.com/pysal/spaghetti/issues/75)
* attempting pyproj_fix [(#122)](https://github.com/pysal/spaghetti/pull/122)
* [WIP] Add snap dist [(#123)](https://github.com/pysal/spaghetti/pull/123)
* travis CI build failing with `KeyError: 'PROJ_LIB'` [(#121)](https://github.com/pysal/spaghetti/issues/121)
* resolving obs_to_node question [(#120)](https://github.com/pysal/spaghetti/pull/120)
* why convert obs_to_node from defaultdict to list? [(#93)](https://github.com/pysal/spaghetti/issues/93)
* network.PointPatterns condense code chunk [(#74)](https://github.com/pysal/spaghetti/issues/74)
* condensing idvariable code chunk [(#119)](https://github.com/pysal/spaghetti/pull/119)
* Network Cross Nearest Neighbor [(#102)](https://github.com/pysal/spaghetti/issues/102)
* refreshing docs [(#117)](https://github.com/pysal/spaghetti/pull/117)
* shortest path look up from allneighborsdistances? [(#115)](https://github.com/pysal/spaghetti/issues/115)
* adding shortest path traceback for point patterns [(#116)](https://github.com/pysal/spaghetti/pull/116)
* ImportError: No module named 'boto3' [(#113)](https://github.com/pysal/spaghetti/issues/113)
* adding boto3 test req for current fiona bug [(#114)](https://github.com/pysal/spaghetti/pull/114)
* [WIP] cleanup_nearest_neighbor [(#112)](https://github.com/pysal/spaghetti/pull/112)
* duplicate neighbor distance functions? [(#91)](https://github.com/pysal/spaghetti/issues/91)
* network.allneighbordistances documentation not accurate [(#111)](https://github.com/pysal/spaghetti/issues/111)
* [WIP] General package maintenance [(#109)](https://github.com/pysal/spaghetti/pull/109)
* new badges [(#96)](https://github.com/pysal/spaghetti/issues/96)
* tools/ [(#99)](https://github.com/pysal/spaghetti/issues/99)
* updating thumbnails in docs [(#108)](https://github.com/pysal/spaghetti/pull/108)
* [WIP] updating docs, badges, tools, etc. [(#107)](https://github.com/pysal/spaghetti/pull/107)
* initializing new sphinx docs based on submodule_template [(#98)](https://github.com/pysal/spaghetti/pull/98)
* new labels for issues [(#105)](https://github.com/pysal/spaghetti/issues/105)
* populating sphinx docs [(#37)](https://github.com/pysal/spaghetti/issues/37)
* tests for analysis and util [(#44)](https://github.com/pysal/spaghetti/issues/44)
* NetworkF [(#94)](https://github.com/pysal/spaghetti/issues/94)
* rename functions to be more pythonic [(#104)](https://github.com/pysal/spaghetti/issues/104)
* add poisson distribution to tests [(#106)](https://github.com/pysal/spaghetti/issues/106)
* initial sphix docs attempt [(#67)](https://github.com/pysal/spaghetti/pull/67)
* bumping version to 1.1.0 [(#97)](https://github.com/pysal/spaghetti/pull/97)
* adding in new tests for utils.py [(#95)](https://github.com/pysal/spaghetti/pull/95)
* add flag for util.generatetree() [(#92)](https://github.com/pysal/spaghetti/issues/92)
* [completed atm] - docstrings cleanup [(#89)](https://github.com/pysal/spaghetti/pull/89)
* clean docstrings [(#77)](https://github.com/pysal/spaghetti/issues/77)
* adding MANIFEST.in [(#88)](https://github.com/pysal/spaghetti/pull/88)
* clearing Facility_Location.ipynb [(#87)](https://github.com/pysal/spaghetti/pull/87)
* removing typo in Facility_Location [(#86)](https://github.com/pysal/spaghetti/pull/86)
* clearing Facility_Location.ipynb [(#85)](https://github.com/pysal/spaghetti/pull/85)
* updating Facility_Location.ipynb for typos [(#84)](https://github.com/pysal/spaghetti/pull/84)
* adding Facility_Location.ipynb [(#83)](https://github.com/pysal/spaghetti/pull/83)
* new notebook ideas [(#48)](https://github.com/pysal/spaghetti/issues/48)
* adding windows functionality for 'last updated' [(#82)](https://github.com/pysal/spaghetti/pull/82)
* ensure nearest nodes are returned as np.array() [(#73)](https://github.com/pysal/spaghetti/pull/73)
* snapping trouble when the initial node in KDtree is the nearest [(#72)](https://github.com/pysal/spaghetti/issues/72)
* add Github version badge [(#80)](https://github.com/pysal/spaghetti/issues/80)
* add open issues badge [(#79)](https://github.com/pysal/spaghetti/issues/79)
* update notebooks as per pysal/pysal#1057 [(#81)](https://github.com/pysal/spaghetti/issues/81)
* [Complete/Needs Review] updating `in_shp` parameter in spaghetti.Network [(#69)](https://github.com/pysal/spaghetti/pull/69)
* [ENH] geopandas.GeoDataFrame for PointPattern [(#28)](https://github.com/pysal/spaghetti/issues/28)
* update in_shp kwarg in spaghetti.Network [(#68)](https://github.com/pysal/spaghetti/issues/68)
* removing undeclared edge_time attribute [(#65)](https://github.com/pysal/spaghetti/pull/65)
* update README.txt [(#33)](https://github.com/pysal/spaghetti/issues/33)
* [ENH] Add badges [(#31)](https://github.com/pysal/spaghetti/issues/31)
* Publish on Zenodo [(#36)](https://github.com/pysal/spaghetti/issues/36)
* Some errors in node_distance_matrix() [(#64)](https://github.com/pysal/spaghetti/issues/64)
* declare SMALL as np.finfo(float).eps [(#63)](https://github.com/pysal/spaghetti/pull/63)
* smallest numpy epsilon float? [(#61)](https://github.com/pysal/spaghetti/issues/61)
* [WIP] Prep for next pypi [(#60)](https://github.com/pysal/spaghetti/pull/60)
* mimic other pysal submodules for api removal [(#50)](https://github.com/pysal/spaghetti/issues/50)
* PEP8 compliant [(#38)](https://github.com/pysal/spaghetti/issues/38)
* update notebooks [(#43)](https://github.com/pysal/spaghetti/issues/43)
* api testing [(#59)](https://github.com/pysal/spaghetti/issues/59)
* DRY version documentation [(#53)](https://github.com/pysal/spaghetti/issues/53)
* configure travis dual testing for "Allowed Failures" to work [(#58)](https://github.com/pysal/spaghetti/pull/58)
* adding geopandas for dual travis testing [(#56)](https://github.com/pysal/spaghetti/pull/56)
* New tests required for new in_shp parameter option [(#30)](https://github.com/pysal/spaghetti/issues/30)
* fix or remove code_health badge [(#54)](https://github.com/pysal/spaghetti/issues/54)
* add .landscape.yml for code health [(#51)](https://github.com/pysal/spaghetti/issues/51)
* removing code health badge from README [(#55)](https://github.com/pysal/spaghetti/pull/55)
* adding .landscape.yml for code health [(#52)](https://github.com/pysal/spaghetti/pull/52)
* adding additional pip install instructions/options [(#35)](https://github.com/pysal/spaghetti/pull/35)
* dual testing [(#45)](https://github.com/pysal/spaghetti/issues/45)
* change libpysal imports in tests [(#46)](https://github.com/pysal/spaghetti/issues/46)
* change weights variable name [(#47)](https://github.com/pysal/spaghetti/issues/47)
* update notebooks for reorg [(#1)](https://github.com/pysal/spaghetti/issues/1)
* spaghetti/analysis.py:182: RuntimeWarning [(#42)](https://github.com/pysal/spaghetti/issues/42)
* change xrange to range [(#40)](https://github.com/pysal/spaghetti/issues/40)
* summation error in computeobserved() of spaghetti.analysis.py [(#41)](https://github.com/pysal/spaghetti/issues/41)
* TypeError: can't pickle dict_keys objects [(#39)](https://github.com/pysal/spaghetti/issues/39)
* add CHANGELOG [(#34)](https://github.com/pysal/spaghetti/issues/34)
* update import scheme for new package name [(#5)](https://github.com/pysal/spaghetti/issues/5)
* Prepare a release of spaghetti for pypi [(#26)](https://github.com/pysal/spaghetti/issues/26)
* pip [(#32)](https://github.com/pysal/spaghetti/issues/32)
* preparing for pypi release [(#25)](https://github.com/pysal/spaghetti/pull/25)
* api.py tests [(#19)](https://github.com/pysal/spaghetti/issues/19)
* trailing comma not allowed with surrounding parenthesis [(#29)](https://github.com/pysal/spaghetti/issues/29)
* Necessity of __future__? [(#27)](https://github.com/pysal/spaghetti/issues/27)
* `spaghetti` currently only python 2.7.x compatible [(#21)](https://github.com/pysal/spaghetti/issues/21)
* Geopandas read [(#22)](https://github.com/pysal/spaghetti/pull/22)
* Py2topy3 [(#23)](https://github.com/pysal/spaghetti/pull/23)
* Spaghetti/update travis [(#24)](https://github.com/pysal/spaghetti/pull/24)
* Generalize the Network input API for libpysal/#59 [(#20)](https://github.com/pysal/spaghetti/issues/20)

### mapclassify:
* fix doctests (interactive examples in inline docstrings) [(#19)](https://github.com/pysal/mapclassify/pull/19)
* complete readthedocs configuration & add Slocum 2009 reference [(#17)](https://github.com/pysal/mapclassify/pull/17)
* prepping for a doc based release [(#15)](https://github.com/pysal/mapclassify/pull/15)
* new release on pypi [(#10)](https://github.com/pysal/mapclassify/issues/10)
* prepare for release 2.0.0 [(#13)](https://github.com/pysal/mapclassify/pull/13)
* Clean up for next pypi release [(#12)](https://github.com/pysal/mapclassify/pull/12)
* move notebooks outside of the package [(#11)](https://github.com/pysal/mapclassify/pull/11)
* ENH: move classifiers up into init [(#9)](https://github.com/pysal/mapclassify/pull/9)
* Moving to python 3+ [(#8)](https://github.com/pysal/mapclassify/pull/8)

### splot:
* merge Sprint with master branch [(#39)](https://github.com/pysal/splot/pull/39)
* Change documentation style [(#38)](https://github.com/pysal/splot/pull/38)
* add travis build badge to README.md [(#37)](https://github.com/pysal/splot/pull/37)
* fix current documentation for sprint [(#36)](https://github.com/pysal/splot/pull/36)
* `value_by_alpha` prototype [(#28)](https://github.com/pysal/splot/pull/28)
* Clean up of current code base [(#30)](https://github.com/pysal/splot/pull/30)
* Value By Alpha specification [(#24)](https://github.com/pysal/splot/issues/24)
* nonplanar example update [(#33)](https://github.com/pysal/splot/issues/33)
* add README.md [(#29)](https://github.com/pysal/splot/pull/29)
* issues in some docstrings for giddy [(#26)](https://github.com/pysal/splot/issues/26)
* debug `splot` documentation [(#25)](https://github.com/pysal/splot/pull/25)
* collection of cleanups for`splot.giddy`  [(#23)](https://github.com/pysal/splot/pull/23)
* created `esda.moran.Moran_Local_BV` visualisations [(#20)](https://github.com/pysal/splot/pull/20)
* add `esda.moran.Moran_BV` visualizations to `splot.esda` [(#18)](https://github.com/pysal/splot/pull/18)
* add `seaborn` and `matplotlib` to `install_requirements` in `setup.py` [(#19)](https://github.com/pysal/splot/pull/19)
* prototype `moran_scatterplot()`, `plot_moran_simulation()` and `plot_moran()` for `esda` [(#17)](https://github.com/pysal/splot/pull/17)
* include utility functions `shift_colormap` and `truncate_colormap` [(#15)](https://github.com/pysal/splot/pull/15)
* fix setup.py so files are installed with "pip install ." [(#16)](https://github.com/pysal/splot/pull/16)
* `plot_spatial_weights` including network joins for `non_planar_joins` [(#14)](https://github.com/pysal/splot/pull/14)
* adapting existing `esda` functionality to `splot.esda` namespace and allow `.plot()` method [(#13)](https://github.com/pysal/splot/pull/13)
* adding license [(#4)](https://github.com/pysal/splot/pull/4)
* add `giddy` dynamic LISA functionality under `splot.giddy` [(#11)](https://github.com/pysal/splot/pull/11)
* start sphinx html documentation [(#12)](https://github.com/pysal/splot/pull/12)
* add visualization option with significance to mplot [(#7)](https://github.com/pysal/splot/pull/7)
* Visualising Local Autocorrelation [(#8)](https://github.com/pysal/splot/pull/8)
* Copy new changes made to viz module into split [(#5)](https://github.com/pysal/splot/pull/5)
* run 2to3 for splot [(#6)](https://github.com/pysal/splot/pull/6)

### spreg:
* update docstrings for libpysal API changes [(#9)](https://github.com/pysal/spreg/pull/9)
* Merging in spanel & spreg2 code necessary for new spatial panel & GeoDaSpace [(#10)](https://github.com/pysal/spreg/pull/10)
* move to silence_warnings from current libpysal [(#7)](https://github.com/pysal/spreg/pull/7)
* add init to ensure tests are shipped [(#6)](https://github.com/pysal/spreg/pull/6)
* weights typechecking will only accept things from `pysal`.  [(#3)](https://github.com/pysal/spreg/issues/3)
* relax error checking in check_weights [(#4)](https://github.com/pysal/spreg/pull/4)
* simplify testing [(#5)](https://github.com/pysal/spreg/pull/5)
* Convert spreg to common subset 2,3 code [(#2)](https://github.com/pysal/spreg/pull/2)

### spglm:
* fix docstrings (as well as some within interactive examples) [(#14)](https://github.com/pysal/spglm/pull/14)
* Fix docs [(#17)](https://github.com/pysal/spglm/pull/17)
* Submodule [(#16)](https://github.com/pysal/spglm/pull/16)
* submodule_contract [(#13)](https://github.com/pysal/spglm/pull/13)
* Inconsistent metadata in setup.py [(#10)](https://github.com/pysal/spglm/issues/10)
* adapting spglm to new libpysal [(#12)](https://github.com/pysal/spglm/pull/12)
* move to using libpysal.io.open rather than just libpysal.open [(#11)](https://github.com/pysal/spglm/pull/11)
* Freeze [(#8)](https://github.com/pysal/spglm/pull/8)
* add tr_S attribute for use in GWR [(#7)](https://github.com/pysal/spglm/pull/7)
* remove pysal requirements [(#6)](https://github.com/pysal/spglm/pull/6)
* remove v 2.x tests from CI [(#5)](https://github.com/pysal/spglm/pull/5)
* houskeeping of new basefiles needed as a submodule [(#4)](https://github.com/pysal/spglm/pull/4)

### spint:
* adapting spint to newest version of libpysal [(#13)](https://github.com/pysal/spint/pull/13)
* Reorg [(#12)](https://github.com/pysal/spint/pull/12)
* spint has pysal as a dependency, should be libpysal [(#2)](https://github.com/pysal/spint/issues/2)
* api.py syntax error [(#10)](https://github.com/pysal/spint/issues/10)
* removed trailing api comma [(#11)](https://github.com/pysal/spint/pull/11)
* Freeze [(#8)](https://github.com/pysal/spint/pull/8)
* update dependencies [(#7)](https://github.com/pysal/spint/pull/7)
* remove v 2.x test from CI [(#6)](https://github.com/pysal/spint/pull/6)
* version bump [(#5)](https://github.com/pysal/spint/pull/5)
* Common subset [(#4)](https://github.com/pysal/spint/pull/4)
* houskeeping of basefiles needed for submodule  [(#3)](https://github.com/pysal/spint/pull/3)

### mgwr:
* add badges to README [(#35)](https://github.com/pysal/mgwr/pull/35)
* (ENH) prepare online docs [(#33)](https://github.com/pysal/mgwr/pull/33)
* Revert "move notebooks outside of the package folder and fix notebooks" [(#34)](https://github.com/pysal/mgwr/pull/34)
* move notebooks outside of the package folder and fix notebooks [(#32)](https://github.com/pysal/mgwr/pull/32)
* Georgia main example patch [(#31)](https://github.com/pysal/mgwr/pull/31)
* format gwr.py following PEP 8 style and fix docstrings [(#30)](https://github.com/pysal/mgwr/pull/30)
* use libpysal in docstrings and adapt to python 3 syntax [(#29)](https://github.com/pysal/mgwr/pull/29)
* Inconsistent metadata info on setup.py [(#25)](https://github.com/pysal/mgwr/issues/25)
* rebuild rights access? [(#28)](https://github.com/pysal/mgwr/issues/28)
* swap to libpysal.io.open [(#26)](https://github.com/pysal/mgwr/pull/26)
* adapting mgwr to newest libpysal [(#27)](https://github.com/pysal/mgwr/pull/27)
* change spreg import pattern [(#24)](https://github.com/pysal/mgwr/pull/24)
* change imports from spreg [(#22)](https://github.com/pysal/mgwr/issues/22)
* rework pickles in the tests [(#21)](https://github.com/pysal/mgwr/issues/21)
* Swap to use more portable types than pickles [(#23)](https://github.com/pysal/mgwr/pull/23)
* Output summary [(#17)](https://github.com/pysal/mgwr/issues/17)
* Adding summary output [(#18)](https://github.com/pysal/mgwr/pull/18)
* Freeze [(#20)](https://github.com/pysal/mgwr/pull/20)
* Allow user-set BW's for MGWR [(#7)](https://github.com/pysal/mgwr/issues/7)
* Set mgwr bw [(#15)](https://github.com/pysal/mgwr/pull/15)
* adds py27 to ci [(#16)](https://github.com/pysal/mgwr/pull/16)
* Gwr to mgwr [(#14)](https://github.com/pysal/mgwr/pull/14)
* Standard errors and t-vals [(#8)](https://github.com/pysal/mgwr/issues/8)
* Redundant calculation of Aj [(#10)](https://github.com/pysal/mgwr/issues/10)
* question about final fit in MGWR class?  [(#1)](https://github.com/pysal/mgwr/issues/1)
* change main directory from gwr to mgwr [(#13)](https://github.com/pysal/mgwr/pull/13)
* Update mgwr [(#12)](https://github.com/pysal/mgwr/pull/12)
* clean up MGWR [(#11)](https://github.com/pysal/mgwr/pull/11)
* Consolidate MGWR [(#5)](https://github.com/pysal/mgwr/pull/5)
* hat matrices [(#2)](https://github.com/pysal/mgwr/issues/2)

### spvcm:
* Test failures in effective size & geweke diagnostics [(#2)](https://github.com/pysal/spvcm/issues/2)
* update plotting and diagnostics for pandas deprecation [(#3)](https://github.com/pysal/spvcm/pull/3)



## Summary Statistics

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>package</th>
      <th>commits</th>
      <th>total issues</th>
      <th>pulls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>spaghetti</td>
      <td>282</td>
      <td>94</td>
      <td>34</td>
    </tr>
    <tr>
      <td>libpysal</td>
      <td>276</td>
      <td>79</td>
      <td>57</td>
    </tr>
    <tr>
      <td>splot</td>
      <td>247</td>
      <td>27</td>
      <td>21</td>
    </tr>
    <tr>
      <td>mgwr</td>
      <td>229</td>
      <td>30</td>
      <td>16</td>
    </tr>
    <tr>
      <td>giddy</td>
      <td>137</td>
      <td>38</td>
      <td>31</td>
    </tr>
    <tr>
      <td>esda</td>
      <td>80</td>
      <td>29</td>
      <td>19</td>
    </tr>
    <tr>
      <td>spglm</td>
      <td>70</td>
      <td>12</td>
      <td>9</td>
    </tr>
    <tr>
      <td>spint</td>
      <td>57</td>
      <td>11</td>
      <td>9</td>
    </tr>
    <tr>
      <td>spreg</td>
      <td>57</td>
      <td>8</td>
      <td>6</td>
    </tr>
    <tr>
      <td>mapclassify</td>
      <td>48</td>
      <td>9</td>
      <td>8</td>
    </tr>
    <tr>
      <td>pointpats</td>
      <td>40</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <td>inequality</td>
      <td>36</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <td>spvcm</td>
      <td>21</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>commits</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dani Arribas-Bel</th>
      <td>25</td>
    </tr>
    <tr>
      <th>Eli Knaap</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Hu Shao</th>
      <td>5</td>
    </tr>
    <tr>
      <th>James Gaboardi</th>
      <td>310</td>
    </tr>
    <tr>
      <th>Jsignell</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Levi John Wolf</th>
      <td>227</td>
    </tr>
    <tr>
      <th>Philip Kahn</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Serge Rey</th>
      <td>239</td>
    </tr>
    <tr>
      <th>Stefanie Lumnitz</th>
      <td>240</td>
    </tr>
    <tr>
      <th>Taylor Oshan</th>
      <td>239</td>
    </tr>
    <tr>
      <th>Thequackdaddy</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Wei Kang</th>
      <td>217</td>
    </tr>
    <tr>
      <th>Ziqi Li</th>
      <td>51</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dani Arribas-Bel</th>
      <th>Eli Knaap</th>
      <th>Hu Shao</th>
      <th>James Gaboardi</th>
      <th>Jsignell</th>
      <th>Levi John Wolf</th>
      <th>Philip Kahn</th>
      <th>Serge Rey</th>
      <th>Stefanie Lumnitz</th>
      <th>Taylor Oshan</th>
      <th>Thequackdaddy</th>
      <th>Wei Kang</th>
      <th>Ziqi Li</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>lib.libpysal</th>
      <td>19</td>
      <td>12</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>112</td>
      <td>0</td>
      <td>101</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>explore.esda</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>50</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <th>explore.giddy</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>18</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>106</td>
      <td>0</td>
    </tr>
    <tr>
      <th>explore.inequality</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
    </tr>
    <tr>
      <th>explore.pointpats</th>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>0</td>
    </tr>
    <tr>
      <th>explore.spaghetti</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>278</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>viz.mapclassify</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
    </tr>
    <tr>
      <th>viz.splot</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>7</td>
      <td>218</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>model.spreg</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>model.spglm</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>model.spint</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>model.mgwr</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>133</td>
      <td>0</td>
      <td>21</td>
      <td>51</td>
    </tr>
    <tr>
      <th>model.spvcm</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


