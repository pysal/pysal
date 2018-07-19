def georgia():
    import pysal as ps
    import geopandas as gpd
    import pandas as pd
    import shapely.geometry as geom

    data = pd.read_csv(ps.examples.get_path('GData_utm.csv'))
    data['geometry'] = [geom.Point(x,y) for x,y in data[['X','Y']].values]
    data = gpd.GeoDataFrame(data)

    return data
