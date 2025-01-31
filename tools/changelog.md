Overall, there were 302 commits that closed 122 issues since our last release on 2024-07-31.


<a name="changes-by-package"></a>
## Changes by Package


<a name="libpysal-v4.12.1"></a>
### libpysal v4.12.1
* [#765:](https://github.com/pysal/libpysal/pull/765) ENH: ensure lag_spatial is compatible with both W and Graph 
* [#761:](https://github.com/pysal/libpysal/pull/761) Add exponential kernel to Graph 
* [#763:](https://github.com/pysal/libpysal/pull/763) allow continuous weights for knn graph 
* [#760:](https://github.com/pysal/libpysal/pull/760) Fix for Graph.describe() when the graph has a string index (#759) 
* [#759:](https://github.com/pysal/libpysal/issues/759) BUG: Graph.describe() does not work with non-integer index 


<a name="giddy-v2.3.6"></a>
### giddy v2.3.6
* [#228:](https://github.com/pysal/giddy/pull/228) update CI - minimum versions, naming, etc. 
* [#229:](https://github.com/pysal/giddy/issues/229) some linting adjustments [2024-07-15] 
* [#227:](https://github.com/pysal/giddy/issues/227) update chat from `gitter` to `discord` 
* [#226:](https://github.com/pysal/giddy/issues/226) drop 3.9 as minimally support Python version? 
* [#225:](https://github.com/pysal/giddy/issues/225) CI maint for standard naming, oldest dependencies, & Python 3.12 


<a name="inequality-v1.1.1"></a>
### inequality v1.1.1
* [#96:](https://github.com/pysal/inequality/pull/96) move conftest 
* [#97:](https://github.com/pysal/inequality/pull/97) remove print statements 
* [#95:](https://github.com/pysal/inequality/pull/95) Correct sphinx theme 
* [#94:](https://github.com/pysal/inequality/pull/94) Fix theme for doc build 
* [#93:](https://github.com/pysal/inequality/pull/93) Fix inconsistency in efficient gini 
* [#16:](https://github.com/pysal/inequality/issues/16) Inconsistent results with different input shape for Spatial_Gini 
* [#92:](https://github.com/pysal/inequality/pull/92) Documentation and landing page updates 
* [#15:](https://github.com/pysal/inequality/issues/15) TheilD within group inequality 
* [#91:](https://github.com/pysal/inequality/pull/91) [pre-commit.ci] pre-commit autoupdate 
* [#90:](https://github.com/pysal/inequality/pull/90) Theil doc 
* [#89:](https://github.com/pysal/inequality/pull/89) wolfson nb narrative 
* [#88:](https://github.com/pysal/inequality/pull/88) Polarization indices and new documentation 
* [#86:](https://github.com/pysal/inequality/pull/86) added narrative to theil nb 
* [#87:](https://github.com/pysal/inequality/pull/87) Gini polarization 
* [#85:](https://github.com/pysal/inequality/pull/85) [pre-commit.ci] pre-commit autoupdate 
* [#84:](https://github.com/pysal/inequality/pull/84) remove [Nijkamp & Poot (2013)] citation 
* [#36:](https://github.com/pysal/inequality/issues/36) Nijkamp & Poot (2013)? 
* [#82:](https://github.com/pysal/inequality/issues/82) add root-level `conftest.py` to skip doctest only in `_indices.py` 
* [#81:](https://github.com/pysal/inequality/issues/81) adjust tests for warnings 
* [#83:](https://github.com/pysal/inequality/pull/83) resolve all warnings from CI -- [2024-09-02] 
* [#80:](https://github.com/pysal/inequality/pull/80) add matplotlib as a requirement 
* [#79:](https://github.com/pysal/inequality/issues/79) add `matplotlib` as a dependency 
* [#75:](https://github.com/pysal/inequality/pull/75) Schutz inequality measures 
* [#78:](https://github.com/pysal/inequality/pull/78) Wolfson bipolarization index 
* [#74:](https://github.com/pysal/inequality/pull/74) Atkinson inequality measures 
* [#77:](https://github.com/pysal/inequality/pull/77) reup lint+format - rename CI env 
* [#76:](https://github.com/pysal/inequality/issues/76) rename CI envs 
* [#73:](https://github.com/pysal/inequality/issues/73) drop `black` -> adopt `ruff` for formatting 
* [#72:](https://github.com/pysal/inequality/pull/72) Pengram 
* [#69:](https://github.com/pysal/inequality/pull/69) deprecation of _indices 
* [#67:](https://github.com/pysal/inequality/issues/67) CI adjustment needed for `numpy>=2.1.0.dev0` 


<a name="pointpats-v2.5.1"></a>
### pointpats v2.5.1
* [#148:](https://github.com/pysal/pointpats/pull/148) TST: relax assertion to avoid floating point issues 
* [#147:](https://github.com/pysal/pointpats/pull/147) COMPAT: compatibility with numpy 
* [#145:](https://github.com/pysal/pointpats/pull/145) ENH: support geopandas objects in distance statistics 


<a name="segregation-v2.5.1"></a>
### segregation v2.5.1
* [#231:](https://github.com/pysal/segregation/pull/231) testing 
* [#230:](https://github.com/pysal/segregation/pull/230) old numpy nan 
* [#229:](https://github.com/pysal/segregation/pull/229) (bug) numpy 2.0 is not supporting np.NaN 



<a name="momepy-v0.9.1"></a>
### momepy v0.9.1
* [#661:](https://github.com/pysal/momepy/pull/661) ENH: do not fail with 3d nodes - `preprocess.remove_false_nodes()` 
* [#660:](https://github.com/pysal/momepy/pull/660) DOC: clear the installation instructions 
* [#658:](https://github.com/pysal/momepy/pull/658) ENH: add Streetscape class 
* [#654:](https://github.com/pysal/momepy/pull/654) CI: pin fiona 
* [#675:](https://github.com/pysal/momepy/pull/675) Get mean of actual values (do not imply missing == 0) in Streetscape 
* [#673:](https://github.com/pysal/momepy/issues/673) Streetscape assumes 0 when nan is given 
* [#674:](https://github.com/pysal/momepy/pull/674) BUG: fix corner case of empty intersection in streetscape 
* [#671:](https://github.com/pysal/momepy/pull/671) BUG: fix extraction of ids if there is only a single hit when retrieving point level data in Streetscape 
* [#670:](https://github.com/pysal/momepy/pull/670) DOC: more osmnx compat 
* [#669:](https://github.com/pysal/momepy/pull/669) DOC: user guide compat with osmnx 2.0 
* [#668:](https://github.com/pysal/momepy/pull/668) Bump codecov/codecov-action from 4 to 5 
* [#667:](https://github.com/pysal/momepy/pull/667) ENH: retain index of buildings and plots intersecting sightlines 
* [#666:](https://github.com/pysal/momepy/pull/666) handling edge cases in `preprocessing.FaceArtifacts` 
* [#665:](https://github.com/pysal/momepy/issues/665) all equivalent `"face_artifact_index"` leads to `LinAlgError` in `preprocessing.FaceArtifacts` 
* [#664:](https://github.com/pysal/momepy/issues/664) fail gracefully when no initial polygons generated – `preprocessing.FaceArtifacts` 
* [#657:](https://github.com/pysal/momepy/issues/657) remove_false_nodes throws an unexpected error 
* [#659:](https://github.com/pysal/momepy/issues/659) momepy.COINS: raise ValueError if empty geodataframe is passed 
* [#656:](https://github.com/pysal/momepy/pull/656) [pre-commit.ci] pre-commit autoupdate 
* [#655:](https://github.com/pysal/momepy/pull/655) Bump mamba-org/setup-micromamba from 1 to 2 
* [#653:](https://github.com/pysal/momepy/pull/653) Allow nodes as an input to graph construction method 'gdf_to_nx' 
* [#652:](https://github.com/pysal/momepy/pull/652) let COINS run even if there's overlapping geometry 


<a name="spreg-v1.8.1"></a>
### spreg v1.8.1
* [#170:](https://github.com/pysal/spreg/pull/170) Fixing GM_KPP in the presence of pandas DF 
* [#167:](https://github.com/pysal/spreg/pull/167) Updating DPG api listing 
* [#165:](https://github.com/pysal/spreg/pull/165) add spsearch to docs, rm sphinx-bibtex pin 
* [#166:](https://github.com/pysal/spreg/pull/166) Update tutorials.rst to include new notebooks 
* [#164:](https://github.com/pysal/spreg/pull/164) Update build docs using segregation's version 
* [#163:](https://github.com/pysal/spreg/pull/163) Spreg version 1.8 
* [#160:](https://github.com/pysal/spreg/pull/160) Adding spsearch.py 
* [#162:](https://github.com/pysal/spreg/pull/162) doc: Fix typo in DGP docs 
* [#159:](https://github.com/pysal/spreg/pull/159) Bump mamba-org/setup-micromamba from 1 to 2 
* [#158:](https://github.com/pysal/spreg/pull/158) Updating spreg to 1.7 
* [#156:](https://github.com/pysal/spreg/pull/156) `ruff` format repo 
* [#151:](https://github.com/pysal/spreg/issues/151) update `pre-commit` -- add `ruff` ; drop `black` 
* [#150:](https://github.com/pysal/spreg/issues/150) swap from `black` to `ruff` for formatting 
* [#154:](https://github.com/pysal/spreg/pull/154) update `environment.yml` & remove `.coveragerc` 
* [#149:](https://github.com/pysal/spreg/pull/149) build docs with 3.12 environment 
* [#153:](https://github.com/pysal/spreg/issues/153) purge `.coveragerc` - no longer needed 
* [#152:](https://github.com/pysal/spreg/issues/152) update `environment.yml` 
* [#148:](https://github.com/pysal/spreg/issues/148) Build docs failure 
* [#146:](https://github.com/pysal/spreg/issues/146) spatial diagnostics in OLS fail with Graph 
* [#147:](https://github.com/pysal/spreg/pull/147) Deprecating check_spat_diag 
* [#145:](https://github.com/pysal/spreg/pull/145) Updating to version 1.6.0 
* [#127:](https://github.com/pysal/spreg/issues/127) ENH: support pandas object as X, y 
* [#57:](https://github.com/pysal/spreg/issues/57) Docstring test failures 
* [#125:](https://github.com/pysal/spreg/issues/125) `scipy.sparse.csr` DeprecationWarning in `user_output.py` 
* [#135:](https://github.com/pysal/spreg/issues/135) 2 tests in CI failing [2024-04-23] 


<a name="tobler-v0.12.1"></a>
### tobler v0.12.1
* [#223:](https://github.com/pysal/tobler/pull/223) h3compat 
* [#222:](https://github.com/pysal/tobler/pull/222) CI: fetch examples ahead of time 
* [#221:](https://github.com/pysal/tobler/pull/221) COMPAT: fix compatibility with scipy 1.15 
* [#219:](https://github.com/pysal/tobler/pull/219) DOCS: use nbsphinx-link and include myst beta 
* [#220:](https://github.com/pysal/tobler/pull/220) codecov4 
* [#204:](https://github.com/pysal/tobler/issues/204) Site docs return 404 
* [#213:](https://github.com/pysal/tobler/issues/213) docs action is failing 
* [#206:](https://github.com/pysal/tobler/pull/206) Docs notebook links 
* [#214:](https://github.com/pysal/tobler/pull/214) Docs 
* [#205:](https://github.com/pysal/tobler/pull/205) infrastructure 
* [#218:](https://github.com/pysal/tobler/pull/218) COMPAT: compatibility with h3 v4 
* [#207:](https://github.com/pysal/tobler/issues/207) update `h3fy` for h3 api changes 
* [#216:](https://github.com/pysal/tobler/issues/216) cut new release of tobler? 
* [#217:](https://github.com/pysal/tobler/issues/217) area_interpolate() fills gap areas where target and source geopandas frames don't intersect with 0s instead of NaNs 


<a name="mapclassify-v2.8.1"></a>
### mapclassify v2.8.1
* [#229:](https://github.com/pysal/mapclassify/pull/229) fix nan handling in color array 
* [#230:](https://github.com/pysal/mapclassify/pull/230) pin max fiona in oldest 


<a name="splot-v1.1.7"></a>
### splot v1.1.7
* [#187:](https://github.com/pysal/splot/pull/187) BUG: fix scatter plots and regression lines in Moran plots 
* [#186:](https://github.com/pysal/splot/issues/186) BUG: OLS regression in moran is wrong 
* [#178:](https://github.com/pysal/splot/issues/178) BUG: plot_local_autocorrelation colors do not match between subplots 
* [#131:](https://github.com/pysal/splot/issues/131) Inconsistent colors for `moran_scatterplot` and `lisa_cluster`  when p values are small 
* [#185:](https://github.com/pysal/splot/pull/185) Consistent hot/cold spot colors for local moran 
* [#184:](https://github.com/pysal/splot/pull/184) (bug) support graph in moran's viz 


<a name="pysal-v25.01"></a>
### pysal v25.01
* [#1337:](https://github.com/pysal/pysal/pull/1337) removing `spvcm` from meta release -- archived - #1330 
* [#1330:](https://github.com/pysal/pysal/issues/1330) inclusion of `spvcm` in PySAL meta-release 
* [#1362:](https://github.com/pysal/pysal/pull/1362) 25.01rc1 
* [#1360:](https://github.com/pysal/pysal/pull/1360) Bump codecov/codecov-action from 4 to 5 
* [#1358:](https://github.com/pysal/pysal/pull/1358) Bump mamba-org/setup-micromamba from 1 to 2 
* [#1357:](https://github.com/pysal/pysal/pull/1357) pin fiona in oldest 
* [#1355:](https://github.com/pysal/pysal/pull/1355) MNT: New release and publish action 

<a name="contributors"></a>
## Contributors

Many thanks to all of the following individuals who contributed to this release:


 - Eli Knaap
 - James Gaboardi
 - Josiah Parry
 - Knaaptime
 - Krasen Samardzhiev
 - Martin Fleischmann
 - Pedro Amaral
 - Serge Rey
 - Wei Kang


