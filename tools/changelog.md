Overall, there were 704 commits that closed 363 issues since our last release on 2026-01-31.


<a name="changes-by-package"></a>
## Changes by Package


<a name="libpysal-v4.15.0"></a>
### libpysal v4.15.0
* [#882:](https://github.com/pysal/libpysal/pull/882) DOC: switch to sphinx-immaterial theme 
* [#925:](https://github.com/pysal/libpysal/pull/925) Bump actions/checkout from 6 to 7 
* [#924:](https://github.com/pysal/libpysal/pull/924) feat: support array-like bandwidth in kernel functions 
* [#922:](https://github.com/pysal/libpysal/pull/922) GHA: fix reverse dependency testing 
* [#923:](https://github.com/pysal/libpysal/pull/923) DEP: pin Pulp to <4 
* [#921:](https://github.com/pysal/libpysal/pull/921) update path for raster test data 
* [#917:](https://github.com/pysal/libpysal/pull/917) ENH: add kernel input validation 
* [#920:](https://github.com/pysal/libpysal/pull/920) Bump codecov/codecov-action from 6 to 7 
* [#889:](https://github.com/pysal/libpysal/issues/889) Support build of "adaptive" kernel graphs 
* [#905:](https://github.com/pysal/libpysal/pull/905) ENH: Add adaptive bandwidth support to Graph.build_kernel 
* [#915:](https://github.com/pysal/libpysal/pull/915) BUG: fix situations where linear geometries disappear in voronoi_frames 
* [#914:](https://github.com/pysal/libpysal/pull/914) GHA: use dedicated environment for docs action 
* [#913:](https://github.com/pysal/libpysal/pull/913) TST: eliminate warning about unary_union 
* [#912:](https://github.com/pysal/libpysal/pull/912) ENH: add Graph.from_lattice 
* [#911:](https://github.com/pysal/libpysal/pull/911) Sparse kernel weights 
* [#909:](https://github.com/pysal/libpysal/issues/909) ENH: use sparse distance computation for compact-support kernels in `Graph.build_kernel()` 
* [#904:](https://github.com/pysal/libpysal/pull/904) FIX: Fix index space mismatch in coplanar clique 
* [#897:](https://github.com/pysal/libpysal/issues/897) BUG: cannot reindex on an axis with duplicate labels when using clique on coplanar points 
* [#910:](https://github.com/pysal/libpysal/pull/910) add `pulp` and a "plus" dep to `pyproject.toml` 
* [#908:](https://github.com/pysal/libpysal/pull/908) Bump actions/github-script from 8 to 9 
* [#906:](https://github.com/pysal/libpysal/pull/906) [pre-commit.ci] pre-commit autoupdate 
* [#903:](https://github.com/pysal/libpysal/pull/903) Bump mamba-org/setup-micromamba from 2 to 3 
* [#902:](https://github.com/pysal/libpysal/pull/902) DOC: update docs infra based on momepy lessons 
* [#900:](https://github.com/pysal/libpysal/pull/900) source version info in docs from `packaging` 
* [#901:](https://github.com/pysal/libpysal/pull/901) cancel doc builds in progress 
* [#899:](https://github.com/pysal/libpysal/pull/899) Bump codecov/codecov-action from 5 to 6 
* [#890:](https://github.com/pysal/libpysal/pull/890) Delete .ipynb_checkpoints directory 
* [#891:](https://github.com/pysal/libpysal/pull/891) move to new `docs/` dir structure 
* [#898:](https://github.com/pysal/libpysal/pull/898) DOC: fix table rendering 
* [#441:](https://github.com/pysal/libpysal/issues/441) Getting an error while trying to calculate knn weights from dataframe 
* [#896:](https://github.com/pysal/libpysal/pull/896) DOC: resolve sphinx warnings 
* [#895:](https://github.com/pysal/libpysal/pull/895) DOC: fold ReadMe to the docs 
* [#893:](https://github.com/pysal/libpysal/pull/893) Optimise default weights generation using dict comprehension 
* [#894:](https://github.com/pysal/libpysal/pull/894) Fix reverse dependency check 
* [#892:](https://github.com/pysal/libpysal/pull/892) drop Python 311 support - spec0 
* [#881:](https://github.com/pysal/libpysal/pull/881) Add deprecation for `KDtree` and `Arc_KDTree` 
* [#886:](https://github.com/pysal/libpysal/issues/886) pandana link in build_travel_cost should point to pandarm instead 
* [#879:](https://github.com/pysal/libpysal/issues/879) deprecate `Arc_KDTree` 
* [#885:](https://github.com/pysal/libpysal/pull/885) Ref/docs 
* [#838:](https://github.com/pysal/libpysal/pull/838) ENH: construction of distance-based weights directly from KDTree/BallTree 
* [#805:](https://github.com/pysal/libpysal/issues/805) Allow construction of distance-based weights directly from KDTree/BallTree 
* [#887:](https://github.com/pysal/libpysal/pull/887) add make_symmetric function to Graph 
* [#664:](https://github.com/pysal/libpysal/issues/664) Interest in a mutual knn? 
* [#888:](https://github.com/pysal/libpysal/pull/888) DOC: link to source from pages 
* [#884:](https://github.com/pysal/libpysal/pull/884) DOC: convert RST to MD 
* [#883:](https://github.com/pysal/libpysal/pull/883) DOC: CI for documentation and version switching 
* [#835:](https://github.com/pysal/libpysal/pull/835) fix: handle distance arrays when calculating bandwidth 
* [#834:](https://github.com/pysal/libpysal/pull/834) Improve `graph._kernely.py` performance  
* [#837:](https://github.com/pysal/libpysal/pull/837) ENH:  Implement Graph.from_dense 
* [#828:](https://github.com/pysal/libpysal/issues/828) ENH: add Graph.from_dense 
* [#870:](https://github.com/pysal/libpysal/pull/870) Add next steps guidance after installation 
* [#875:](https://github.com/pysal/libpysal/pull/875) Add Python 3.14 support to testing matrix 
* [#876:](https://github.com/pysal/libpysal/pull/876) trusted publishing to pypi via action – `release_and_publish.yml` 
* [#871:](https://github.com/pysal/libpysal/pull/871) Enable MyST Markdown support for libpysal documentation 
* [#864:](https://github.com/pysal/libpysal/pull/864) Docs : added guidance on what to do after the installation    
* [#867:](https://github.com/pysal/libpysal/pull/867) Change return type from numpy.array to pandas.Series 
* [#866:](https://github.com/pysal/libpysal/issues/866) Docstring return type mismatch in component_labels 
* [#865:](https://github.com/pysal/libpysal/pull/865) Docs: add next steps after installation 
* [#860:](https://github.com/pysal/libpysal/pull/860) DOC: clarify Rook and Queen contiguity for point geometries 
* [#706:](https://github.com/pysal/libpysal/issues/706) Contiguity weight docs 


<a name="access-v1.1.10.post3"></a>
### access v1.1.10.post3


<a name="esda-v2.10.0"></a>
### esda v2.10.0
* [#509:](https://github.com/pysal/esda/pull/509) Bump actions/checkout from 6 to 7 
* [#404:](https://github.com/pysal/esda/issues/404) `moa_ratio` and `nmi` in the shapes module are the same thing, and should yield the same result 
* [#507:](https://github.com/pysal/esda/pull/507) DOC: polish user guide 
* [#496:](https://github.com/pysal/esda/pull/496) Docs user guide 
* [#505:](https://github.com/pysal/esda/pull/505) Bump codecov/codecov-action from 6 to 7 
* [#275:](https://github.com/pysal/esda/issues/275) Local G Autocorrelation causing Numba TypingError 
* [#442:](https://github.com/pysal/esda/pull/442) BUG: cast integers to floats for numba compatibility in Getis-Ord 
* [#503:](https://github.com/pysal/esda/pull/503) DOC: fix citation 
* [#501:](https://github.com/pysal/esda/pull/501) DOC: properly expose LocalCrossPlot 
* [#502:](https://github.com/pysal/esda/pull/502) DOC: fix expected value in docstring 
* [#445:](https://github.com/pysal/esda/issues/445) Implement the Westerholt Plot and/or combined LISALOSH plot 
* [#446:](https://github.com/pysal/esda/pull/446) ENH: add LocalCrossPlot (Westerholt) plot and ability to scale Moran scatterplot by LOSH 
* [#487:](https://github.com/pysal/esda/pull/487) BUG:  Moran_Local p_sim calculation should use crand 
* [#486:](https://github.com/pysal/esda/issues/486) alternative argument propagating? 
* [#498:](https://github.com/pysal/esda/pull/498) GHA: use dedicated env for docs build 
* [#497:](https://github.com/pysal/esda/pull/497) Doc 2.9 fix 
* [#493:](https://github.com/pysal/esda/pull/493) TST: load test data ahead of time 
* [#492:](https://github.com/pysal/esda/pull/492) fix rough edges in `upload_package.yml` 
* [#490:](https://github.com/pysal/esda/pull/490) Doc: missing ref for LOSH 
* [#489:](https://github.com/pysal/esda/pull/489) DOC: fixing sphinx warnings on doc build for shape.py 
* [#491:](https://github.com/pysal/esda/pull/491) DOC: update sphinx pin in env for doc build action 
* [#488:](https://github.com/pysal/esda/pull/488) Debugging broken search on esda docs 
* [#389:](https://github.com/pysal/esda/issues/389) test our examples nightly/regularly 
* [#481:](https://github.com/pysal/esda/pull/481) get doctests of Examples back running 
* [#485:](https://github.com/pysal/esda/issues/485) `map_comparision` examples using data from testing suite 
* [#484:](https://github.com/pysal/esda/issues/484) not passing `seed` in to `crand()` within the `.fit()` method 
* [#482:](https://github.com/pysal/esda/issues/482) `seed` not being passed to `crand()` in `Geary_Local.fit()` 
* [#480:](https://github.com/pysal/esda/pull/480) Update version string in v2.9.0 and stable 
* [#199:](https://github.com/pysal/esda/issues/199) Add `alternative` argument for determining one-tailed or two-tailed permutation tests 
* [#479:](https://github.com/pysal/esda/pull/479) actually add (and pass through) the `alternative` keyword arg for classes using `crand()` 
* [#478:](https://github.com/pysal/esda/issues/478) `alternative` keyword needed for classes that call `crand.crand()` 
* [#477:](https://github.com/pysal/esda/pull/477) reup SPEC000 - [2026-05-02] 
* [#476:](https://github.com/pysal/esda/pull/476) only run `push` CI on main branch 
* [#292:](https://github.com/pysal/esda/issues/292) Error when importing esda package in Python 
* [#475:](https://github.com/pysal/esda/pull/475) Handle more CI warnings CI – [2026-05-02] 
* [#429:](https://github.com/pysal/esda/issues/429) BUG: Replace deprecated np.row_stack with np.vstack for NumPy 1.25+ compatibility 
* [#212:](https://github.com/pysal/esda/issues/212) GSoC 2022 Interfaces for Consistent API Design 
* [#473:](https://github.com/pysal/esda/pull/473) Add root files for gh-pages versioning 
* [#291:](https://github.com/pysal/esda/issues/291) smoothers are undocumented 
* [#474:](https://github.com/pysal/esda/pull/474) update_versions_json.py belongs only in gh-pages 
* [#472:](https://github.com/pysal/esda/pull/472) clean up some warnings in CI – [2026-05-01] 
* [#471:](https://github.com/pysal/esda/pull/471) Update build docs for versioning 
* [#469:](https://github.com/pysal/esda/pull/469) dependencies; 
* [#470:](https://github.com/pysal/esda/pull/470) add deps for nb build 
* [#468:](https://github.com/pysal/esda/pull/468) Doc map comparison 
* [#465:](https://github.com/pysal/esda/pull/465) Move tests in __main__ blocks to actual tests or remove 
* [#467:](https://github.com/pysal/esda/pull/467) Docs federation build 
* [#460:](https://github.com/pysal/esda/issues/460) `G_Local.__crand()` -- time to deprecate? 
* [#466:](https://github.com/pysal/esda/pull/466) Deprecate legacy __crand in getisord 
* [#430:](https://github.com/pysal/esda/pull/430) fix(#428): convert strict raise DeprecationWarning to warnings.warn in smoothing.py 
* [#428:](https://github.com/pysal/esda/issues/428) BUG: raising DeprecationWarning halts execution instead of warning in smoothing.py 
* [#463:](https://github.com/pysal/esda/pull/463) remove deprecated 'by_col()' - `smoothing.py` 
* [#462:](https://github.com/pysal/esda/pull/462) remove deprecated 'by_col()' - join_counts.py 
* [#461:](https://github.com/pysal/esda/pull/461) remove `CHANGELOG.md` - no longer used 
* [#338:](https://github.com/pysal/esda/issues/338) `TestGeary::test_by_col[W]` failure -- `312-numba-dev` 
* [#458:](https://github.com/pysal/esda/pull/458) remove deprecated 'by_col()' - `{gamma,geary}.py` 
* [#447:](https://github.com/pysal/esda/pull/447) DOC: refactor to federation structure 
* [#399:](https://github.com/pysal/esda/issues/399) revisit outstanding items in docs build 
* [#456:](https://github.com/pysal/esda/pull/456) mark `by_col()` methods as deprecated 
* [#457:](https://github.com/pysal/esda/pull/457) remove deprecated 'by_col()' - `getisord.py` 
* [#455:](https://github.com/pysal/esda/pull/455) remove deprecated HeadBanging after 9 years 
* [#11:](https://github.com/pysal/esda/pull/11) WIP: Headbang re-work 
* [#229:](https://github.com/pysal/esda/issues/229) is `HeadBanging` deprecated? 
* [#452:](https://github.com/pysal/esda/pull/452) Stricter linting following #451 
* [#416:](https://github.com/pysal/esda/issues/416) Failing Moran plotting tests in dev 
* [#451:](https://github.com/pysal/esda/pull/451) Lint+turn on pre-commit for PRs -- [2026-04-25] 
* [#450:](https://github.com/pysal/esda/issues/450) unnecessary variable declaration in `shape.moment_of_inertia_regions()` 
* [#449:](https://github.com/pysal/esda/pull/449) re-up formatting [2026-04-19] 
* [#444:](https://github.com/pysal/esda/pull/444) DOC: move to new `docs/` dir structure 
* [#443:](https://github.com/pysal/esda/pull/443) Bump actions/github-script from 8 to 9 
* [#440:](https://github.com/pysal/esda/pull/440) Bump codecov/codecov-action from 5 to 6 
* [#441:](https://github.com/pysal/esda/pull/441) Bump mamba-org/setup-micromamba from 2 to 3 
* [#438:](https://github.com/pysal/esda/pull/438) Fix to calculation of normalized mass moment of inertia (in `moment_of_inertia_regions` function) 
* [#439:](https://github.com/pysal/esda/pull/439) Shape notebook 
* [#437:](https://github.com/pysal/esda/issues/437) Normalized mass moment of inertia is incorrect. 
* [#436:](https://github.com/pysal/esda/pull/436) BUG: fix the shapely version check 
* [#435:](https://github.com/pysal/esda/issues/435) shapely version isolator troubles 
* [#402:](https://github.com/pysal/esda/issues/402) update PyPI & `upload_package.yml` to use trusted publisher 
* [#434:](https://github.com/pysal/esda/pull/434) MAINT: Trusted publisher on pypi 
* [#433:](https://github.com/pysal/esda/pull/433) TST: geopandas plotting compatibility 
* [#432:](https://github.com/pysal/esda/pull/432) Fix for #431: changed incorrect sign for result of length_width_diff 
* [#431:](https://github.com/pysal/esda/issues/431) Sign on Length-Width Difference shape measure is backwards. 
* [#420:](https://github.com/pysal/esda/pull/420) Inertial shape measures - mass moment of inertia, speed improvements, other fixes 
* [#411:](https://github.com/pysal/esda/pull/411) Fix: Replace deprecated 3-argument np.minimum/np.maximum calls for NumPy 2.0+ compatibility 
* [#410:](https://github.com/pysal/esda/issues/410) Fix: NumPy 2.0+ compatibility replace deprecated 3-argument np.minimum/np.maximum calls 
* [#419:](https://github.com/pysal/esda/pull/419) fix: Add alternative hypothesis support to spatial statistics  
* [#412:](https://github.com/pysal/esda/issues/412) Update `.gitignore` to correctly exclude generated documentation (as opposed to source files) 
* [#418:](https://github.com/pysal/esda/pull/418) Fix broken citation link in documentation 


<a name="giddy-v2.3.9"></a>
### giddy v2.3.9
* [#257:](https://github.com/pysal/giddy/pull/257) [pre-commit.ci] pre-commit autoupdate 
* [#256:](https://github.com/pysal/giddy/pull/256) reup SPEC000 - [2026 05 10]  
* [#255:](https://github.com/pysal/giddy/pull/255) DEP: remove upper pin on esda 
* [#254:](https://github.com/pysal/giddy/pull/254) [pre-commit.ci] pre-commit autoupdate 
* [#253:](https://github.com/pysal/giddy/pull/253) github as trusted publisher on pypi 
* [#251:](https://github.com/pysal/giddy/pull/251) remove `CHANGELOG.md` - no longer used 


<a name="inequality-v1.1.2"></a>
### inequality v1.1.2


<a name="pointpats-v2.6.0"></a>
### pointpats v2.6.0
* [#205:](https://github.com/pysal/pointpats/issues/205) some `numba`-enhanced functionality failing tests 
* [#206:](https://github.com/pysal/pointpats/pull/206) fix skyums algorithm 
* [#204:](https://github.com/pysal/pointpats/pull/204) reup spec000 & some linting [2026-04-15] 
* [#201:](https://github.com/pysal/pointpats/pull/201) add ruff configuration - pre-commit 
* [#203:](https://github.com/pysal/pointpats/pull/203) rename `shapely` in `environment.yml` 
* [#202:](https://github.com/pysal/pointpats/issues/202) shapely dep wrong in environment file 
* [#200:](https://github.com/pysal/pointpats/pull/200) format+lint - `docs/user-guide/` 
* [#184:](https://github.com/pysal/pointpats/issues/184) ruff – format & lint `pointpats` 
* [#132:](https://github.com/pysal/pointpats/issues/132) distance_statistics.ipynb is corrupted 
* [#71:](https://github.com/pysal/pointpats/issues/71) BUG: statistics are now lower case. Notebooks and tutorials need to be updated 
* [#100:](https://github.com/pysal/pointpats/issues/100) operands could not be broadcast together with shapes (16,) (18,) 
* [#68:](https://github.com/pysal/pointpats/issues/68) PoissonPointProcess - ValueError: Length mismatch 
* [#83:](https://github.com/pysal/pointpats/issues/83) Tree types is missing now from Ripley.py  
* [#101:](https://github.com/pysal/pointpats/issues/101) cannot import name 'G' from 'pointpats' 
* [#113:](https://github.com/pysal/pointpats/issues/113) usage of `intensity` within `random.py` distributions? 
* [#70:](https://github.com/pysal/pointpats/issues/70) HAS_NUMBA left undefined 
* [#199:](https://github.com/pysal/pointpats/pull/199) retire the `notebooks/` directory 
* [#198:](https://github.com/pysal/pointpats/issues/198) time to scrap the `notebook/s` directory? 
* [#196:](https://github.com/pysal/pointpats/pull/196) format+lint - `{centography,distance_statistics}.py` + previous missed 
* [#144:](https://github.com/pysal/pointpats/issues/144) Figure out how to get legend for plot_density 
* [#197:](https://github.com/pysal/pointpats/pull/197) Added ContourSet object to return statement in plot_density.py 
* [#178:](https://github.com/pysal/pointpats/issues/178) need to adopt trusted publishing for release action 
* [#194:](https://github.com/pysal/pointpats/pull/194) swap to `FutureWarning` for `knox()` function 
* [#189:](https://github.com/pysal/pointpats/issues/189) deprecate old `knox` function 
* [#193:](https://github.com/pysal/pointpats/pull/193) `_spacetime_points_to_arrays()` - convert assertion to ValueError 
* [#190:](https://github.com/pysal/pointpats/issues/190) raise `ValueError` in `_spacetime_points_to_arrays()` rather than assertion 
* [#192:](https://github.com/pysal/pointpats/pull/192) remove deprecated `numpy.row_stack` for `numpy.vstack` 
* [#191:](https://github.com/pysal/pointpats/pull/191) format+lint - `{geometry,kde,pointpattern,process,quadrat_statistics}.py` 
* [#188:](https://github.com/pysal/pointpats/pull/188) format+lint - `{random,spacetime,window,util}.py` 
* [#187:](https://github.com/pysal/pointpats/issues/187) docstring for `cluster_normal()` seems to be copied from `cluster_poisson()` 
* [#186:](https://github.com/pysal/pointpats/pull/186) format /tests/*; lint + refactor `tests/*.py` 
* [#185:](https://github.com/pysal/pointpats/pull/185) format `/tests/*`; lint + refactor `tests/test_spacetime.py` 
* [#183:](https://github.com/pysal/pointpats/pull/183) REGR: fix GeoDataFrame input to QStatistics 
* [#182:](https://github.com/pysal/pointpats/issues/182) REGR: Qstatistics no longer accepts GeoDataFrame 
* [#181:](https://github.com/pysal/pointpats/pull/181) consistent `rng` for `cluster_poisson()` 
* [#180:](https://github.com/pysal/pointpats/issues/180) `rng` not passed into `_uniform_circle()` within `cluster_poisson()` 
* [#179:](https://github.com/pysal/pointpats/pull/179) prepare `release_and_publish.yml` for trusted publishing 
* [#177:](https://github.com/pysal/pointpats/pull/177) adopt dynamic versioning  
* [#176:](https://github.com/pysal/pointpats/issues/176) `__version__` never got updated when we switched to dynamic versioning 


<a name="segregation-v2.5.5"></a>
### segregation v2.5.5
* [#266:](https://github.com/pysal/segregation/pull/266) GHA: bump github-script 
* [#265:](https://github.com/pysal/segregation/pull/265) Segregation JOSS paper: V1 
* [#263:](https://github.com/pysal/segregation/pull/263) respect spec000, add oldest & dev testing 
* [#262:](https://github.com/pysal/segregation/issues/262) pinned oldest dependencies & tests against them 
* [#258:](https://github.com/pysal/segregation/issues/258) Pandarm obscured 
* [#260:](https://github.com/pysal/segregation/pull/260) fix pandarm link on Readme 
* [#259:](https://github.com/pysal/segregation/pull/259) MAINT: Trusted publisher on pypi 
* [#218:](https://github.com/pysal/segregation/issues/218) update build_docs.yml 
* [#248:](https://github.com/pysal/segregation/issues/248) current failing tests (<= Python 3.12 envs) 
* [#249:](https://github.com/pysal/segregation/issues/249) CI env dependency problem == Python 3.13 
* [#255:](https://github.com/pysal/segregation/pull/255) cleanup `pyproject.toml` - min Python version etc 
* [#256:](https://github.com/pysal/segregation/pull/256) dont use loky backend in tests 
* [#257:](https://github.com/pysal/segregation/issues/257) cast to strict ints in inference workflows 
* [#241:](https://github.com/pysal/segregation/pull/241) modernize macOS testing 
* [#243:](https://github.com/pysal/segregation/pull/243) Add Python 3.14 to CI test matrix (#1382) 
* [#247:](https://github.com/pysal/segregation/issues/247) `urbanaccess` dependency & Python 3.14 
* [#251:](https://github.com/pysal/segregation/pull/251) pandana-->pandarm 


<a name="spaghetti-v1.7.6"></a>
### spaghetti v1.7.6


<a name="mgwr-v2.2.1"></a>
### mgwr v2.2.1


<a name="momepy-v1.0.0"></a>
### momepy v1.0.0
* [#765:](https://github.com/pysal/momepy/pull/765) GHA: fix release action 
* [#739:](https://github.com/pysal/momepy/issues/739) ValueError when running "corners" and "squareness" 
* [#763:](https://github.com/pysal/momepy/pull/763) BUG: raise informative error for MultiPolygon in corner-based shape metrics (#739) 
* [#764:](https://github.com/pysal/momepy/pull/764) Bump actions/checkout from 6 to 7 
* [#762:](https://github.com/pysal/momepy/pull/762) DOC: update funding 
* [#761:](https://github.com/pysal/momepy/issues/761) Images not rendering on landing page 
* [#760:](https://github.com/pysal/momepy/pull/760) DOC: fix allowed and expected type docs in graph metrics 
* [#759:](https://github.com/pysal/momepy/pull/759) Bump codecov/codecov-action from 6 to 7 
* [#758:](https://github.com/pysal/momepy/pull/758) REF: refactor Streetscape to increase its performance 
* [#757:](https://github.com/pysal/momepy/pull/757) ENH: support generation of proximity bands 
* [#756:](https://github.com/pysal/momepy/pull/756) REF: simplify implementation of enclosures 
* [#755:](https://github.com/pysal/momepy/pull/755) GHA: use dedicated docs env 
* [#754:](https://github.com/pysal/momepy/pull/754) CI: remove iprogress 
* [#753:](https://github.com/pysal/momepy/pull/753) Reup spec000 [2026-04-25] 
* [#752:](https://github.com/pysal/momepy/issues/752) simplify by default in `*_tessellation()` 
* [#686:](https://github.com/pysal/momepy/pull/686) ENH: Added Cellular Automata-based Enclosed Tessellation 
* [#751:](https://github.com/pysal/momepy/pull/751) Bump actions/github-script from 8 to 9 
* [#750:](https://github.com/pysal/momepy/pull/750) [pre-commit.ci] pre-commit autoupdate 
* [#749:](https://github.com/pysal/momepy/pull/749) Bump mamba-org/setup-micromamba from 2 to 3 
* [#748:](https://github.com/pysal/momepy/pull/748) Bump codecov/codecov-action from 5 to 6 
* [#747:](https://github.com/pysal/momepy/pull/747) correct markdown link syntax in README 
* [#746:](https://github.com/pysal/momepy/pull/746) DOC: fix the versioning switcher 
* [#745:](https://github.com/pysal/momepy/pull/745) DOC: switch theme to new PySAL standard 
* [#744:](https://github.com/pysal/momepy/pull/744) DOC: fix docstring of neighbors 
* [#741:](https://github.com/pysal/momepy/pull/741) REF: refactor elongation and simplify its docstring 
* [#742:](https://github.com/pysal/momepy/pull/742) DOC: fix convexity reference 
* [#740:](https://github.com/pysal/momepy/issues/740) Wrong citations in the documentation 
* [#738:](https://github.com/pysal/momepy/pull/738) adapt `release_to_pypi.yml` for trusted publishing 


<a name="spglm-v1.1.0"></a>
### spglm v1.1.0


<a name="spint-v1.1.0"></a>
### spint v1.1.0
* [#55:](https://github.com/pysal/spint/pull/55) add release action - non-trusted publisher 
* [#54:](https://github.com/pysal/spint/pull/54) [pre-commit.ci] pre-commit autoupdate 
* [#53:](https://github.com/pysal/spint/pull/53) README cleanup and report code coverage 
* [#52:](https://github.com/pysal/spint/pull/52) migrate from `unittest` to `pytest` 
* [#51:](https://github.com/pysal/spint/pull/51) Lint notebooks & add notebooks dir to pre-commit 
* [#50:](https://github.com/pysal/spint/pull/50) Revise README with new badges and formatting 
* [#47:](https://github.com/pysal/spint/pull/47) proper linting of `spint` codebase 
* [#46:](https://github.com/pysal/spint/pull/46) format remaining notebooks Python 2.x syntax 
* [#45:](https://github.com/pysal/spint/pull/45) format `spint/` and some of notebooks 
* [#44:](https://github.com/pysal/spint/pull/44) Add Python 3.1{3,4} support; get testing green 
* [#40:](https://github.com/pysal/spint/issues/40) working out testing failures 
* [#41:](https://github.com/pysal/spint/issues/41) modernize infra - `pyproject.toml` [2024-07-08] 
* [#37:](https://github.com/pysal/spint/issues/37) update infra – pyproject.toml, setuptools_scm, ruff, etc. 
* [#42:](https://github.com/pysal/spint/issues/42) support Python 3.12 - add to CI matrix 
* [#43:](https://github.com/pysal/spint/pull/43) Update infra, supported Python, GHA, etc. 
* [#36:](https://github.com/pysal/spint/issues/36) rename `master` to `main` 
* [#39:](https://github.com/pysal/spint/issues/39) permissions for the steering council 


<a name="spml-v0.2.2"></a>
### spml v0.2.2
* [#130:](https://github.com/pysal/spml/pull/130) spatialml -> spml 
* [#129:](https://github.com/pysal/spml/pull/129) TST: ignore deprecation warning solved by joblib in upstream 
* [#128:](https://github.com/pysal/spml/pull/128) Bump actions/checkout from 6 to 7 in the github-actions group 
* [#91:](https://github.com/pysal/spml/pull/91) Add documentation contribution guide to `CONTRIBUTING.md` 
* [#87:](https://github.com/pysal/spml/pull/87) Add examples for GWLinearRegression and GWLogisticRegression 
* [#127:](https://github.com/pysal/spml/pull/127) rename package to spatialml 
* [#126:](https://github.com/pysal/spml/pull/126) Bump codecov/codecov-action from 6 to 7 in the github-actions group 
* [#124:](https://github.com/pysal/spml/pull/124) DOC: update examples for sklearn 1.9 
* [#122:](https://github.com/pysal/spml/pull/122) TST: ignore warnings about OOB scores 
* [#121:](https://github.com/pysal/spml/pull/121) GHA: use dedicated docs environment 
* [#120:](https://github.com/pysal/spml/pull/120) TST: use linear models in tests where possible for speed up 
* [#114:](https://github.com/pysal/spml/pull/114) Add validation for bandwidth and kernel parameters 
* [#118:](https://github.com/pysal/spml/pull/118) Add score_ attribute based on pooled local model performance (BaseClassifier) 
* [#119:](https://github.com/pysal/spml/pull/119) reup SPEC000 - 2026-04-18 
* [#117:](https://github.com/pysal/spml/pull/117) Bump actions/github-script from 8 to 9 in the github-actions group 
* [#115:](https://github.com/pysal/spml/pull/115) [pre-commit.ci] pre-commit autoupdate 
* [#113:](https://github.com/pysal/spml/pull/113) ENH: Added validation for bandwidth and kernel parameters in GWR models 
* [#112:](https://github.com/pysal/spml/pull/112) ENH: expose local_metric on regressors as well 
* [#111:](https://github.com/pysal/spml/pull/111) open-ended license 
* [#110:](https://github.com/pysal/spml/pull/110) DOC: update config 
* [#109:](https://github.com/pysal/spml/pull/109) Bump the github-actions group with 2 updates 
* [#107:](https://github.com/pysal/spml/pull/107) source version info in docs from `packaging` 
* [#108:](https://github.com/pysal/spml/pull/108) cancel doc builds in progress 
* [#106:](https://github.com/pysal/spml/pull/106) ENH: expose handling of coplanar points 
* [#105:](https://github.com/pysal/spml/pull/105) DOC: link to source and new libpysal location 
* [#104:](https://github.com/pysal/spml/pull/104) DOC: use markdown instead of RST in references 
* [#103:](https://github.com/pysal/spml/issues/103) [GSoC 2026] Proposal: Geographically Weighted Matrix Decomposition (GWPCA) with scikit-learn compatibility 
* [#102:](https://github.com/pysal/spml/pull/102) REF: rely on sklearn wrappers around joblib Parallel 
* [#94:](https://github.com/pysal/spml/pull/94) Fix inconsistent bandwidth validation for numpy scalar and non-scalar inputs 
* [#101:](https://github.com/pysal/spml/pull/101) BUG: fix batch fitting with non-default index 
* [#100:](https://github.com/pysal/spml/pull/100) compute information criteria only for linear models 
* [#92:](https://github.com/pysal/spml/issues/92) BUG: inconsistent bandwidth validation for numpy scalar and non-scalar inputs 
* [#97:](https://github.com/pysal/spml/pull/97) Added pooled scoring and robust aggregation 
* [#99:](https://github.com/pysal/spml/pull/99) Refactor: Centralize local model initialization in _BaseModel 
* [#96:](https://github.com/pysal/spml/issues/96) Add Spatial Consistency and Convergence Tests for Kernels 
* [#98:](https://github.com/pysal/spml/pull/98) Bump actions/upload-artifact from 6 to 7 in the github-actions group 
* [#95:](https://github.com/pysal/spml/issues/95) Modularize matrix operations and leverage calculation 
* [#93:](https://github.com/pysal/spml/issues/93) [BUG] Incorrect use of oob_score parameter in GWRandomForest models 
* [#86:](https://github.com/pysal/spml/pull/86) BUG: fix failure of BandwidthSearch in case of an invariant y 
* [#84:](https://github.com/pysal/spml/issues/84) BUG: bandwidth search fails if y is all True 
* [#89:](https://github.com/pysal/spml/pull/89) Remove outdated TODO block for performance metrics in parallel validation test 
* [#88:](https://github.com/pysal/spml/issues/88) Clarification: Equality vs Tolerance for Parallel Metric Validation Tests 
* [#82:](https://github.com/pysal/spml/pull/82) Expose local_class_support_ as fitted attribute 
* [#83:](https://github.com/pysal/spml/pull/83) Add structured fit-time input validation 
* [#81:](https://github.com/pysal/spml/pull/81) FMT: Ruff format ipynb files 
* [#80:](https://github.com/pysal/spml/issues/80) Missing ruff formats on ipynb files 
* [#77:](https://github.com/pysal/spml/pull/77) Add scikit-learn metadata routing support to BaseRegressor 
* [#78:](https://github.com/pysal/spml/pull/78) DOC: fix outdated BaseRegressor prediction docstring 


<a name="spreg-v1.9.0"></a>
### spreg v1.9.0
* [#192:](https://github.com/pysal/spreg/pull/192) Adding new panels module 


<a name="tobler-v0.14.0"></a>
### tobler v0.14.0
* [#280:](https://github.com/pysal/tobler/pull/280) [pre-commit.ci] pre-commit autoupdate 
* [#274:](https://github.com/pysal/tobler/pull/274) strict linting for code health 
* [#268:](https://github.com/pysal/tobler/pull/268) suggestion `miniforge`, not Anaconda 
* [#276:](https://github.com/pysal/tobler/pull/276) fix broken docs build action 
* [#275:](https://github.com/pysal/tobler/issues/275) broken docs recipe 
* [#273:](https://github.com/pysal/tobler/pull/273) catch several more expected warnings in CI 
* [#272:](https://github.com/pysal/tobler/pull/272) Delete `.gitattributes` -- no longer using `versioneer` 
* [#271:](https://github.com/pysal/tobler/pull/271) Format and lint `docs/*` -- check during `pre-commit` 
* [#270:](https://github.com/pysal/tobler/pull/270) Format and lint `tobler/tests/*` 
* [#269:](https://github.com/pysal/tobler/pull/269) clean up actions workflow yamls - run testing CI only on `main` push 
* [#261:](https://github.com/pysal/tobler/pull/261) migrate docs 
* [#266:](https://github.com/pysal/tobler/pull/266) Setup precommit & re-lint + format codebase 
* [#265:](https://github.com/pysal/tobler/pull/265) [maint] fix package name in license 
* [#267:](https://github.com/pysal/tobler/pull/267) fewer maps in dotdensity example 
* [#263:](https://github.com/pysal/tobler/pull/263) add pixel_values in notebook 
* [#262:](https://github.com/pysal/tobler/issues/262) Total area population not matching original population after interpolation 
* [#264:](https://github.com/pysal/tobler/pull/264) new structure of gh-pages 
* [#260:](https://github.com/pysal/tobler/pull/260) robust nan_handle 
* [#167:](https://github.com/pysal/tobler/issues/167) Issue with the version in docs 
* [#247:](https://github.com/pysal/tobler/pull/247) [ENH] add pointpattern functions 
* [#259:](https://github.com/pysal/tobler/pull/259) MAINT: Trusted publisher for pypi 


<a name="mapclassify-v2.10.0"></a>
### mapclassify v2.10.0


<a name="splot-v1.1.7"></a>
### splot v1.1.7


<a name="spopt-v0.7.0"></a>
### spopt v0.7.0


<a name="pysal-v26.07rc1"></a>
### pysal v26.07rc1
* [#1441:](https://github.com/pysal/pysal/pull/1441) Bump actions/setup-python from 6 to 7 
* [#1440:](https://github.com/pysal/pysal/pull/1440) Bump actions/checkout from 6 to 7 
* [#1438:](https://github.com/pysal/pysal/pull/1438) Bump codecov/codecov-action from 6 to 7 
* [#1436:](https://github.com/pysal/pysal/issues/1436) failure to install development (`giddy`/`esda`) in `ubuntu-latest, ci/314-dev.yaml` 
* [#1434:](https://github.com/pysal/pysal/pull/1434) Bump codecov/codecov-action from 5 to 6 
* [#1433:](https://github.com/pysal/pysal/pull/1433) Bump mamba-org/setup-micromamba from 2 to 3 
* [#1375:](https://github.com/pysal/pysal/issues/1375) Adopt spec 0 across pysal 
* [#1435:](https://github.com/pysal/pysal/pull/1435) minor docs cleanup 
* [#1324:](https://github.com/pysal/pysal/issues/1324) update docs & README 
* [#1339:](https://github.com/pysal/pysal/issues/1339) testing against oldest dependencies across federation 
* [#1277:](https://github.com/pysal/pysal/issues/1277) distutils will be deprecated in Python 3.12 
* [#1338:](https://github.com/pysal/pysal/issues/1338) standardize CI environment naming conventions 
* [#1340:](https://github.com/pysal/pysal/issues/1340) testing against Python 3.12 across federation 
* [#1216:](https://github.com/pysal/pysal/issues/1216) Consider using SCVersioning for docs 
* [#1282:](https://github.com/pysal/pysal/issues/1282) consider new theme and GHA for versioned docs 
* [#1298:](https://github.com/pysal/pysal/issues/1298) add Apple silicon CI envs across ecosystem 
* [#1372:](https://github.com/pysal/pysal/issues/1372) Testing against Python 3.13 across federation 
* [#1373:](https://github.com/pysal/pysal/issues/1373) remove testing against Python 3.10 across federation 
* [#1380:](https://github.com/pysal/pysal/issues/1380) CI: retire macOS-13 for macOS-15-intel 
* [#1382:](https://github.com/pysal/pysal/issues/1382) CI: Testing against Python 3.14 across federation 
* [#1383:](https://github.com/pysal/pysal/pull/1383) Modernize supported Python, etc 
* [#1431:](https://github.com/pysal/pysal/pull/1431) GHA: fix the release action 
* [#1430:](https://github.com/pysal/pysal/issues/1430) remove `publish.yml`? 
* [#1427:](https://github.com/pysal/pysal/pull/1427) REF: support older import style for lib 
* [#1403:](https://github.com/pysal/pysal/pull/1403) Add structured warnings to requires decorator 
* [#1356:](https://github.com/pysal/pysal/issues/1356) CI: pin fiona 
* [#1414:](https://github.com/pysal/pysal/pull/1414) Fix typo in README 
* [#1421:](https://github.com/pysal/pysal/issues/1421) Automate updating PySAL subpackage dependencies in pyproject.toml using release_info.py 
* [#1422:](https://github.com/pysal/pysal/pull/1422) ENH: Automate updating `pyproject.toml` dependencies using `release_info.py` 
* [#1420:](https://github.com/pysal/pysal/issues/1420) [DOC] Add user guide and example for GWLinearRegression and GWLogisticRegression 
* [#1419:](https://github.com/pysal/pysal/pull/1419) Add test for _installed_version with non-existent package 
* [#1416:](https://github.com/pysal/pysal/issues/1416) Add type hints to pysal/base.py and modernize version checking with importlib.metadata 
* [#1417:](https://github.com/pysal/pysal/issues/1417) GSoC project sizes. 
* [#1418:](https://github.com/pysal/pysal/issues/1418) Add return value to Versions.check() method and comprehensive tests 
* [#1412:](https://github.com/pysal/pysal/issues/1412) Build Docs Action is Failing 
* [#1413:](https://github.com/pysal/pysal/pull/1413) Add lazy_loader when running building docs action 
* [#1404:](https://github.com/pysal/pysal/pull/1404) Tooling updated for 26.01rc1 
* [#1411:](https://github.com/pysal/pysal/pull/1411) Drop 3.10 testing 

<a name="contributors"></a>
## Contributors

Many thanks to all of the following individuals who contributed to this release:


 - Ashish Raj
 - Dylan
 - Eli Knaap
 - Elliott Sales De Andrade
 - James Gaboardi
 - Jigyasa
 - Knaaptime
 - Lee Hachadoorian
 - Levi John Wolf
 - Martin Fleischmann
 - Pedro Amaral
 - Philip Stephens
 - Ps098
 - R Virinchi
 - Ramji Purwar
 - Renan Xavier Cortes
 - Serge Rey
 - Shubham Singh
 - Vincent Gao
 - Wei Kang
 - Yuta Sato