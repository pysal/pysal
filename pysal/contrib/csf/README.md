The Local Moran's I statistic can be used to develop location-specific
indicators of spatial autocorrelation. These measures can identify different
forms of departures from spatial randomness, including hot-spots were the
focal unit and its geographical neighbors display elevated measures on an
attribute. Cold spots, which consist of low values surrounded by low values
can also be identified. A critical implementation issue in the application of
the Local statistics is the definition of the neighborhood sets, reflected in
the spatial weights. We rely on an embedding of PySAL components into the CSF framework to
develop spatial analytical workflows to compare the sensitivity of
hot(cold)-spot identification to the choice of spatial weights. We illustrate
this framework using data on homicide rates for US counties and compare the
rook and queen criteria in implementing the spatial weights.
