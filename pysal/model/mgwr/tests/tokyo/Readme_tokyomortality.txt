Dataset name: 
	Tokyo Mortality Data

File name
	tokyomortality.txt

File type
	text file (space delimited data matrix)

Fields
	IDnum0: sequential areal id
	X_CENTROID: x coordinate of areal centroid
	Y_CENTROID: y coordinate of areal centroid
	db2564: observed number of working age (25-64 yrs) deaths
	eb2564: expected number of working age (25-64 yrs) deaths
	OCC_TEC: proportion of professional workers
	OWNH: proportion of owned houses
	POP65: proportion of elderly people (equal to or older than 65)
	UNEMP: unemployment rate

Areal unit
	262 municipality

Areal extent
	The Tokyo metropolitan area is enclosed by an approximate 70 km radius 
	from the centroid of the Chiyoda ward of Tokyo where the Imperial Palace is located.

Year
	1990

Source
	Vital Statistics and Population Census of Japan

Sample session control file
	SampleTokyoMortalityGWPR.ctl
 
	Note 1: This sample fit a semiparametric geographically weighted Poisson regression model
		used in Nakaya et al. (2005).
	Note 2: Since all of the explanatory variables are standardised in the paper, the standardisation 
		option is turned on in this sample.

Additional file: 
   tokyomet262.shp, shx, dbf, prj: ESRI shape file of the Tokyo metropolitan area

   Note 1: IDnum0 in tokyomortality.txt can be matched with AreaID in this shapefile dbf.
   Note 2: Coordinates are projected using UTM54 (Tokyo datum).
   Note 3: distance unit is metre.

Reference
	Nakaya, T., Fotheringham, S., Brunsdon, C. and Charlton, M. (2005): 
	Geographically weighted Poisson regression for disease associative mapping,
	Statistics in Medicine 24, 2695-2717.

History
  15 May 2012	The dataset is prepared for GWR4 sample dataset by TN.
