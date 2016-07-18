import pysal.core.FileIO as FileIO
from geomet import wkb
from sqlalchemy.ext.automap import automap_base

from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

class SQLConnection(FileIO.FileIO):
    """
    Reads SQL mappable
    """

    FORMATS = ['sqlite', 'db']
    MODES = ['r']

    def __init__(self, *args, **kwargs):
        self._typ = str
        FileIO.FileIO.__init__(self, *args, **kwargs)
        #self.file = open(self.dataPath, self.mode)

        self.dbname = args[0]
        self.Base = automap_base()
        self._engine = create_engine(self.dbname)
        self.Base.prepare(self._engine, reflect=True)

    def read(self, *args, **kwargs):
        return self._get_gjson(*args, **kwargs)

    def seek(self):
        pass

    def next(self):
        pass

    def close(self):
        self.file.close()
        FileIO.FileIO.close(self)

    def _get_gjson(self, tablename, attributes=False, geom_column="GEOMETRY"):

        gjson = {
            "type": "FeatureCollection",
            "features": []}

        for row in self.session.query(self.tables[tablename]):
            feat = {"type": "Feature",
                    "geometry": {},
                    "properties":{}}

            feat["geometry"] = wkb.loads(row.GEOMETRY)
            attributes = row.__dict__
            attributes.pop('_sa_instance_state', None)
            attributes.pop(geom_column, None)
            feat["properties"] = attributes
            gjson["features"].append(feat)

        return gjson


    @property
    def tables(self):
        if not hasattr(self, '_tables'):
            self._tables = {}
            for k in self.Base.classes.keys():
                self._tables[k] = self.Base.classes[k]
        return self._tables

    @property
    def session(self):
        #What happens if the session is externally closed?  Check for None?
        if not hasattr(self, '_session'):
            self._session = Session(self._engine)
        return self._session
