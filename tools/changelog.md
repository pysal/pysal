Overall, there were 352 commits that closed 192 issues since our last release on 2025-07-31.


<a name="changes-by-package"></a>
## Changes by Package


<a name="libpysal-v4.14.1"></a>
### libpysal v4.14.1
* [#824:](https://github.com/pysal/libpysal/pull/824) REGR: Revert #793 (sorting in sparse to arrays) 
* [#822:](https://github.com/pysal/libpysal/pull/822) REGR: fix regression in Graph.build_kernel 
* [#821:](https://github.com/pysal/libpysal/issues/821) REGR: regression in Graph.build_kernel in 4.14 
* [#823:](https://github.com/pysal/libpysal/pull/823) [pre-commit.ci] pre-commit autoupdate 
* [#820:](https://github.com/pysal/libpysal/pull/820) DOC: Fix stale documentation links 
* [#819:](https://github.com/pysal/libpysal/pull/819) Move `fast_point_in_polygon_algorithm` notebook and data files to documentation 
* [#60:](https://github.com/pysal/libpysal/issues/60) quadtree files 
* [#480:](https://github.com/pysal/libpysal/pull/480) max val in rtree.silhouette_coeff 
* [#260:](https://github.com/pysal/libpysal/pull/260) ENH: start of patsy branch for discussion 
* [#287:](https://github.com/pysal/libpysal/pull/287) [WIP] Extend KNN neighbor search beyond coincident sites 
* [#184:](https://github.com/pysal/libpysal/pull/184) BUG: Queen and Rook from_dataframe do not match docs 
* [#331:](https://github.com/pysal/libpysal/pull/331) Add ArcGIS API for Python Geometry Support to Weight Objects 
* [#455:](https://github.com/pysal/libpysal/pull/455) BUG: fix support for DataArray objects read through rioxarray to weights module 
* [#445:](https://github.com/pysal/libpysal/issues/445) Add support for `DataArray` objects read through `rioxarray` 
* [#436:](https://github.com/pysal/libpysal/pull/436) Update of the min_threshold_dist_from_shapefile function to support geopandas objects directly. 
* [#815:](https://github.com/pysal/libpysal/pull/815) ENH: add information about components and isolates to Graph.__repr__ 
* [#807:](https://github.com/pysal/libpysal/issues/807) Connectivity warnings on Graph 
* [#790:](https://github.com/pysal/libpysal/issues/790) Expectations in kernel weights 
* [#791:](https://github.com/pysal/libpysal/pull/791) ENH: Add decay and taper arguments to normalize kernel in distance-based weights 
* [#818:](https://github.com/pysal/libpysal/pull/818) TST: ensure EEA large rivers are downloaded prior running tests, numpy compat, plotting text fix 
* [#798:](https://github.com/pysal/libpysal/pull/798) Bug/gaussian 
* [#816:](https://github.com/pysal/libpysal/pull/816) ENH: support any 2-dimensional inputs in Graph.lag() 
* [#813:](https://github.com/pysal/libpysal/issues/813) Graph.lag should support DataFrames 
* [#810:](https://github.com/pysal/libpysal/pull/810) Bump actions/checkout from 5 to 6 
* [#814:](https://github.com/pysal/libpysal/pull/814) COMPAT: pandas 3.0 compatibillity in Graph 
* [#812:](https://github.com/pysal/libpysal/pull/812) CI: ensure pyproj is present in 313-min 
* [#811:](https://github.com/pysal/libpysal/pull/811) modernize macOS testing 
* [#809:](https://github.com/pysal/libpysal/pull/809) BUG: fix euality check on Graph 
* [#808:](https://github.com/pysal/libpysal/pull/808) [pre-commit.ci] pre-commit autoupdate 
* [#806:](https://github.com/pysal/libpysal/pull/806) CI: try fixing spopt reverse dependency testing 
* [#797:](https://github.com/pysal/libpysal/pull/797) add benchmarks of Graph using asv 
* [#802:](https://github.com/pysal/libpysal/pull/802) BUG: RNG pruning condition and loop bound 
* [#801:](https://github.com/pysal/libpysal/issues/801) Bug in the Delaunay prunning (RNG-2) 
* [#803:](https://github.com/pysal/libpysal/pull/803) Bump actions/github-script from 7 to 8 
* [#804:](https://github.com/pysal/libpysal/pull/804) Bump actions/setup-python from 5 to 6 
* [#799:](https://github.com/pysal/libpysal/pull/799) Bump actions/checkout from 4 to 5 


<a name="access-v1.1.10.post3"></a>
### access v1.1.10.post3
* [#99:](https://github.com/pysal/access/pull/99) Update release action for trusted publishing 
* [#97:](https://github.com/pysal/access/pull/97) initial attempt at Python 3.14 in CI 
* [#95:](https://github.com/pysal/access/pull/95) standard CI envs dir & naming conventions 
* [#96:](https://github.com/pysal/access/pull/96) modernize macOS testing 


<a name="esda-v2.8.1"></a>
### esda v2.8.1
* [#401:](https://github.com/pysal/esda/pull/401) support Python 3.14 in CI matrix 
* [#400:](https://github.com/pysal/esda/pull/400) adapted `check_array()` keyword - sklearn version 
* [#394:](https://github.com/pysal/esda/pull/394) pruning down warnings/errors in make html 
* [#398:](https://github.com/pysal/esda/pull/398) modernize macOS testing 
* [#397:](https://github.com/pysal/esda/pull/397) Bump actions/checkout from 5 to 6 
* [#395:](https://github.com/pysal/esda/issues/395) AttributeError in Local_Join_Counts 
* [#393:](https://github.com/pysal/esda/pull/393) fix docs build directory mismatch 
* [#396:](https://github.com/pysal/esda/pull/396) Fix docx issues 
* [#390:](https://github.com/pysal/esda/issues/390) Moran fourth moment missing sum() 
* [#392:](https://github.com/pysal/esda/issues/392) docs directory – `build` vs. `_build` 
* [#391:](https://github.com/pysal/esda/pull/391) correct fourth moment calculation for total randomization null 
* [#252:](https://github.com/pysal/esda/issues/252) correlogram 
* [#259:](https://github.com/pysal/esda/pull/259) add spatial correlogram function 
* [#205:](https://github.com/pysal/esda/pull/205) adding alternative option in local_moran and moral_local_rate 
* [#279:](https://github.com/pysal/esda/pull/279) add start of local partial moran statistics 
* [#281:](https://github.com/pysal/esda/pull/281) pseudo-p significance calculation  
* [#388:](https://github.com/pysal/esda/pull/388) Bump actions/github-script from 7 to 8 
* [#387:](https://github.com/pysal/esda/pull/387) Bump actions/setup-python from 5 to 6 
* [#386:](https://github.com/pysal/esda/pull/386) Bump actions/checkout from 4 to 5 


<a name="giddy-v2.3.8"></a>
### giddy v2.3.8
* [#246:](https://github.com/pysal/giddy/pull/246) fix warnings in the doctests for dynamic_lisa_rose  
* [#244:](https://github.com/pysal/giddy/issues/244) UserWarning: No data for colormapping provided via 'c'. 
* [#239:](https://github.com/pysal/giddy/pull/239) fix failing doctests, other actions maint, ensure docs build, etc 
* [#198:](https://github.com/pysal/giddy/issues/198) fail gracefully when `splot` not installed 
* [#243:](https://github.com/pysal/giddy/pull/243) `use_index` warning from `Queen.from_dataframe()` 
* [#232:](https://github.com/pysal/giddy/pull/232) [pre-commit.ci] pre-commit autoupdate 
* [#242:](https://github.com/pysal/giddy/pull/242) change .toml file to anticipate the deprecation of old crand scheme in esda 
* [#240:](https://github.com/pysal/giddy/pull/240) merge `main` into #232 
* [#235:](https://github.com/pysal/giddy/issues/235) `doctest` failures in `ubuntu-latest, ci/312-dev.yaml` 
* [#233:](https://github.com/pysal/giddy/issues/233) add `with: fetch-depth: 0` for checkout in `build_docs.yml` 


<a name="gwlearn-v0.1.1"></a>
### gwlearn v0.1.1
* [#73:](https://github.com/pysal/gwlearn/pull/73) COMPAT: pandas 3 compatibility 
* [#72:](https://github.com/pysal/gwlearn/pull/72) DOC: fix docs building 
* [#70:](https://github.com/pysal/gwlearn/pull/70) Fix strict type annotation in GWLogisticRegression 
* [#68:](https://github.com/pysal/gwlearn/pull/68) Enable internal metadata routing for geometry 
* [#67:](https://github.com/pysal/gwlearn/pull/67) GHA: build docs for PRs 
* [#66:](https://github.com/pysal/gwlearn/issues/66) Execute notebooks as part of the documentation build 
* [#65:](https://github.com/pysal/gwlearn/pull/65) GHA: ignore pre-commit-ci[bot] in release notes 
* [#64:](https://github.com/pysal/gwlearn/pull/64) DOC: various documentation enhancements 
* [#63:](https://github.com/pysal/gwlearn/pull/63) DOC: make notebooks a bit faster to execute 
* [#62:](https://github.com/pysal/gwlearn/pull/62) MAINT: infrastructure enhancements 
* [#61:](https://github.com/pysal/gwlearn/pull/61) DOC: automatised release update 
* [#59:](https://github.com/pysal/gwlearn/pull/59) DOC: automatically build docs for stable (on version) and latest (on push) 
* [#57:](https://github.com/pysal/gwlearn/pull/57) ENH: add GWGradientBoostingRegressor 
* [#48:](https://github.com/pysal/gwlearn/issues/48) non-linear regressors 
* [#60:](https://github.com/pysal/gwlearn/pull/60) TST: make sure that codecov tracks undersampling 
* [#45:](https://github.com/pysal/gwlearn/pull/45) Implement metadata routing and update fit methods 
* [#56:](https://github.com/pysal/gwlearn/pull/56) document and test metadata routing 
* [#51:](https://github.com/pysal/gwlearn/pull/51) DOC: add comparison with mgwr 
* [#55:](https://github.com/pysal/gwlearn/pull/55) API: move geometry back to fit, implement score and backbone for metadata routing 
* [#43:](https://github.com/pysal/gwlearn/issues/43) Implement score 
* [#54:](https://github.com/pysal/gwlearn/pull/54) ENH: implement fusion with the global model in prediction 
* [#42:](https://github.com/pysal/gwlearn/issues/42) Flexibility of prediction 
* [#52:](https://github.com/pysal/gwlearn/pull/52) ENH: prediction based on nearest model only or a custom bandwidth 
* [#49:](https://github.com/pysal/gwlearn/pull/49) ENH: add GWRandomForestRegressor 
* [#50:](https://github.com/pysal/gwlearn/pull/50) [pre-commit.ci] pre-commit autoupdate 
* [#47:](https://github.com/pysal/gwlearn/pull/47) ENH: support predict method on regressors 
* [#41:](https://github.com/pysal/gwlearn/issues/41) Support predict on regressors 
* [#46:](https://github.com/pysal/gwlearn/pull/46) MAINT: eliminate warnings from CI 
* [#44:](https://github.com/pysal/gwlearn/pull/44) clean up `n_jobs` warning 
* [#40:](https://github.com/pysal/gwlearn/pull/40) minor touch of docs 
* [#39:](https://github.com/pysal/gwlearn/pull/39) full type hints (checked by ty) 
* [#38:](https://github.com/pysal/gwlearn/pull/38) fix some type hints 
* [#37:](https://github.com/pysal/gwlearn/pull/37) beef up docstrings 
* [#36:](https://github.com/pysal/gwlearn/pull/36) fix API rendering 
* [#35:](https://github.com/pysal/gwlearn/pull/35) use immaterial theme 
* [#34:](https://github.com/pysal/gwlearn/pull/34) DOC: Add basic user guide 
* [#33:](https://github.com/pysal/gwlearn/pull/33) custom undersampling 
* [#32:](https://github.com/pysal/gwlearn/pull/32) API: do not compute performance metrics, return arrays for users  
* [#30:](https://github.com/pysal/gwlearn/pull/30) API: try to make sure that API for metrics is not confusing 
* [#31:](https://github.com/pysal/gwlearn/pull/31) CI: attempt testing on Python 3.14 
* [#29:](https://github.com/pysal/gwlearn/pull/29) CI: replace deprecated macos-13 with macos-15-intel 
* [#28:](https://github.com/pysal/gwlearn/pull/28) fix incorrect sorting 
* [#27:](https://github.com/pysal/gwlearn/pull/27) compute global metrics only for non-unique focal set 
* [#26:](https://github.com/pysal/gwlearn/pull/26) ensure bandwidth can be none with custom graph 


<a name="momepy-v0.11.0"></a>
### momepy v0.11.0
* [#720:](https://github.com/pysal/momepy/pull/720) DOC: expand on meaning of segment keyword in tessellation 
* [#714:](https://github.com/pysal/momepy/pull/714) Update and enhance pre-commit-config 
* [#717:](https://github.com/pysal/momepy/pull/717) bump oldest dependency versions as per spec000 – 2025-10 
* [#711:](https://github.com/pysal/momepy/pull/711) TYP: fix type hints for new numpy 
* [#696:](https://github.com/pysal/momepy/pull/696) DEP: deprecate functions moved to neatnet 
* [#727:](https://github.com/pysal/momepy/pull/727) DEPR: deprecate preprocessing tooling 
* [#693:](https://github.com/pysal/momepy/issues/693) Deprecate stuff moved to neatnet 
* [#726:](https://github.com/pysal/momepy/pull/726) COMPAT: streetscape compatibility with numpy 2.4 
* [#704:](https://github.com/pysal/momepy/pull/704) Adding metrics to strokes made by COINS 
* [#591:](https://github.com/pysal/momepy/issues/591) Adding metrics to strokes made by COINS 
* [#710:](https://github.com/pysal/momepy/pull/710) include numba in dev CI env, compat with dev libpysal 
* [#700:](https://github.com/pysal/momepy/pull/700) SciPy sparse array migration from sparse matrices 
* [#723:](https://github.com/pysal/momepy/issues/723) Inconcistency between mm.close_gaps and mm.extend_lines 
* [#716:](https://github.com/pysal/momepy/pull/716) initial attempt at Python 3.14 support 
* [#725:](https://github.com/pysal/momepy/pull/725) modernize macOS testing 
* [#724:](https://github.com/pysal/momepy/pull/724) Bump actions/checkout from 5 to 6 
* [#722:](https://github.com/pysal/momepy/pull/722) DOC: Add missing text to enclosed_tessellation 
* [#707:](https://github.com/pysal/momepy/pull/707) DEP: fix user guide 
* [#706:](https://github.com/pysal/momepy/pull/706) DEP: remove deprecated class API 
* [#718:](https://github.com/pysal/momepy/issues/718) Artefacts in output when tessellating islands with narrow channels 
* [#721:](https://github.com/pysal/momepy/issues/721) Coplanar error 
* [#719:](https://github.com/pysal/momepy/pull/719) Update `elements.py` – typo in `morphological_tessellation ` docstring 
* [#715:](https://github.com/pysal/momepy/issues/715) revisit spec000 minimal support versions – 2025-10 
* [#713:](https://github.com/pysal/momepy/pull/713) [pre-commit.ci] pre-commit autoupdate 
* [#681:](https://github.com/pysal/momepy/pull/681) fixes #680 
* [#683:](https://github.com/pysal/momepy/pull/683) Updating the Contributing file 
* [#708:](https://github.com/pysal/momepy/pull/708) Bump actions/setup-python from 5 to 6 
* [#709:](https://github.com/pysal/momepy/pull/709) Bump actions/github-script from 7 to 8 
* [#705:](https://github.com/pysal/momepy/pull/705) Bump actions/checkout from 4 to 5 



<a name="spreg-v1.8.5"></a>
### spreg v1.8.5
* [#187:](https://github.com/pysal/spreg/issues/187) ``TypeError`` in ``GM_Lag`` when ``slx_vars`` is a list and ``slx_lags > 0``. If ``slx_vars`` is not a list it defaults to ``"all"`` 
* [#189:](https://github.com/pysal/spreg/pull/189) Temporarily disabling 3.14 tests without numba 
* [#186:](https://github.com/pysal/spreg/issues/186) New feature: Different spatial weight matrices for lag and error terms 
* [#188:](https://github.com/pysal/spreg/pull/188) Fix bug in slx_vars in set_endog 
* [#185:](https://github.com/pysal/spreg/pull/185) try Python 3.14 with and without numba 
* [#181:](https://github.com/pysal/spreg/pull/181) modernize macOS testing 
* [#184:](https://github.com/pysal/spreg/pull/184) Bump actions/setup-python from 5 to 6 
* [#183:](https://github.com/pysal/spreg/pull/183) Bump actions/github-script from 7 to 8 
* [#182:](https://github.com/pysal/spreg/pull/182) Bump actions/checkout from 4 to 6 
* [#168:](https://github.com/pysal/spreg/pull/168) Bump mamba-org/setup-micromamba from 1 to 2 
* [#169:](https://github.com/pysal/spreg/pull/169) Bump codecov/codecov-action from 4 to 5 
* [#180:](https://github.com/pysal/spreg/issues/180) Conflicting pysal.spreg.GMM_Error results when using esda.moran.Moran 
* [#175:](https://github.com/pysal/spreg/pull/175) Increasing test coverage 


<a name="tobler-v0.13.0"></a>
### tobler v0.13.0
* [#246:](https://github.com/pysal/tobler/pull/246) CI: pin geopandas version in 3.14 env 
* [#241:](https://github.com/pysal/tobler/pull/241) Python 3.14 in CI matrix etc 
* [#243:](https://github.com/pysal/tobler/pull/243) chipping away more warnings in tests 
* [#245:](https://github.com/pysal/tobler/pull/245) TST: geopandas compat 
* [#242:](https://github.com/pysal/tobler/pull/242) control for some warnings when running tests 
* [#240:](https://github.com/pysal/tobler/pull/240) reup linting for `tobler` - part3 
* [#239:](https://github.com/pysal/tobler/pull/239) reup linting for `tobler` - part2 
* [#238:](https://github.com/pysal/tobler/pull/238) standardize reqs/deps 
* [#237:](https://github.com/pysal/tobler/issues/237) update pyproject.toml & deps 
* [#236:](https://github.com/pysal/tobler/pull/236) modernize macOS testing 
* [#235:](https://github.com/pysal/tobler/pull/235) reup linting for `tobler` -  part1 
* [#230:](https://github.com/pysal/tobler/pull/230) [maint] `ruff` review for `tobler`  
* [#231:](https://github.com/pysal/tobler/issues/231) review spec000 for `tobler` [2025-10] 
* [#232:](https://github.com/pysal/tobler/pull/232) spec000 maintenance – [2025-10] 
* [#210:](https://github.com/pysal/tobler/issues/210) add nightly upstream testing to matrix 
* [#229:](https://github.com/pysal/tobler/pull/229) Update README.md -- universal DOI 
* [#165:](https://github.com/pysal/tobler/issues/165) add doi to readme 
* [#227:](https://github.com/pysal/tobler/issues/227) Add polygon binary dasymetric mapping 



<a name="pysal-v26.01rc1"></a>
### pysal v26.01rc1
* [#1394:](https://github.com/pysal/pysal/issues/1394) Lazy Subpackage Loading for PySAL (PEP 562) 
* [#1386:](https://github.com/pysal/pysal/issues/1386) Modernize dynamic imports: Replace exec/eval with importlib and use stdlib cached_property 
* [#1399:](https://github.com/pysal/pysal/issues/1399) [Docs] Broken "User Guide" link on libpysal documentation homepage 
* [#1396:](https://github.com/pysal/pysal/pull/1396) add gwlearn to meta 
* [#1392:](https://github.com/pysal/pysal/issues/1392) Performance Optimization: Parallel Version Checking for 5.5x Import Speedup 
* [#1395:](https://github.com/pysal/pysal/pull/1395) ENH: implement lazy loading for subpackages using SPEC 1 
* [#1393:](https://github.com/pysal/pysal/pull/1393) feat: add parallel version checking for 5.5x speedup 
* [#1390:](https://github.com/pysal/pysal/issues/1390) NameError bug in _installed_version() and deprecated string formatting 
* [#1391:](https://github.com/pysal/pysal/pull/1391) fix: resolve NameError bug and modernize string formatting 
* [#1388:](https://github.com/pysal/pysal/issues/1388) Add test coverage for pysal.base and pysal.lib.common modules 
* [#1389:](https://github.com/pysal/pysal/pull/1389) test: add comprehensive test coverage for base and lib.common modules 
* [#1387:](https://github.com/pysal/pysal/pull/1387) refactor: modernize dynamic imports with importlib and stdlib cached_property (fixes issue #1386) 
* [#1384:](https://github.com/pysal/pysal/pull/1384) Docs: Migrated contributing guidelines from Wiki to CONTRIBUTING.md 
* [#1068:](https://github.com/pysal/pysal/issues/1068) PyPI packages ownership 
* [#1381:](https://github.com/pysal/pysal/pull/1381) modernize macOS testing 
* [#1379:](https://github.com/pysal/pysal/pull/1379) Bump actions/checkout from 5 to 6 
* [#1378:](https://github.com/pysal/pysal/pull/1378) Bump actions/setup-python from 5 to 6 
* [#1377:](https://github.com/pysal/pysal/pull/1377) Bump actions/checkout from 4 to 5 

<a name="contributors"></a>
## Contributors

Many thanks to all of the following individuals who contributed to this release:


 - Aksrivastava28
 - Ashish Raj
 - Clément Sebastiao
 - Dani Arribas-Bel
 - Dcodrut
 - Eli Knaap
 - Firepheonix
 - Germano Barcelos
 - James Gaboardi
 - Jiya Gupta
 - Jon Morris
 - Knaaptime
 - Levi John Wolf
 - Maria Alice
 - Martin Fleischmann
 - Pedro Amaral
 - Samay Mehar
 - Samay2504
 - Serge Rey
 - Shubham Singh
 - Wei Kang
