from .. import fileio

errmsg = ''
try:
    try:
        from geomet import wkb
    except ImportError:
        from shapely import wkb
except ImportError:
    wkb = None
    errmsg += ('No WKB parser found. Please install one of the following packages '
             'to enable this functionality: [geomet, shapely]\n')

try:
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    nosql_mode = False
except ImportError:
    nosql_mode = True
    errmsg += ('No module named sqlalchemy. Please install'
               ' sqlalchemy to enable this functionality.')

class SQLConnection(fileio.FileIO):
    """
    Reads SQL mappable
    """

    FORMATS = ['sqlite', 'db']
    MODES = ['r']

    def __init__(self, *args, **kwargs):
        if errmsg != '':
            raise ImportError(errmsg)
        self._typ = str
        fileio.FileIO.__init__(self, *args, **kwargs)
        #self.file = open(self.dataPath, self.mode)

        self.dbname = args[0]
        self.Base = automap_base()
        self._engine = create_engine(self.dbname)
        self.Base.prepare(self._engine, reflect=True)
        self.metadata = self.Base.metadata

    def read(self, *args, **kwargs):
        return self._get_gjson(*args, **kwargs)

    def seek(self):
        pass

    def __next__(self):
        pass

    def close(self):
        self.file.close()
        fileio.FileIO.close(self)

    def _get_gjson(self, tablename, geom_column="GEOMETRY"):

        gjson = {
            "type": "FeatureCollection",
            "features": []}

        for row in self.session.query(self.metadata.tables[tablename]):
            feat = {"type": "Feature",
                    "geometry": {},
                    "properties":{}}
            feat["GEOMETRY"] = wkb.loads(getattr(row,geom_column))
            attributes = row._asdict()
            attributes.pop(geom_column, None)
            feat["properties"] = attributes
            gjson["features"].append(feat)

        return gjson


    @property
    def tables(self):
        if not hasattr(self, '_tables'):
            self._tables = list(self.metadata.tables.keys())
        return self._tables

    @property
    def session(self):
        #What happens if the session is externally closed?  Check for None?
        if not hasattr(self, '_session'):
            self._session = Session(self._engine)
        return self._session
