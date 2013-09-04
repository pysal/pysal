import fiona
import json



def shp2geojsons(shapeFile):
    """
    Create a geojson stream from a shapefile

    Parameters
    ==========

    shapeFile: string
               Path to shape file (with .shp extension)


    Returns
    =======

    implicit: string
              GeoJSON string

    Examples
    ========
    >>> import pysal as ps
    >>> shpFile = ps.examples.get_path("columbus.shp")
    >>> jsons = shp2jsons(shpFile)
    >>> type(jsons)
    <type 'str'>

    """
    features = []
    with fiona.collection(shapeFile, 'r') as source:
        for feature in source:
            features.append(feature)
        bbox = source.bounds


    out_layer = {
            "type": "FeatureCollection",
            "bbox": bbox,
            "features": features
            }

    return json.dumps(out_layer)

def shp2geojsonf(shapeFile, geojsonFile):
    """
    Write a shapefile as a geojson file with bounding box for the
    FeatureCollection


    Parameters
    ==========

    shapeFile: string
               Path to shape file (with .shp extension)

    geojsonFile: string
               Path to geojson file (to be created)


    Examples
    ========
    >>> import pysal as ps
    >>> shpFile = ps.examples.get_path("columbus.shp")
    >>> outFile = 'columbus.json'
    >>> shp2geojsonf(shpFile, outFile)
    >>>
    """

    with open(geojsonFile, 'w') as f:
        f.write(shp2geojsons(shapeFile))



