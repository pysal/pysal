import os
from ... import geotable as pdio
from ...fileio import FileIO as psopen
import unittest as ut
from .... import examples as pysal_examples

try:
    import sqlalchemy
    missing_sql = False
except ImportError:
    missing_sql = True

try:
    import geomet
    missing_geomet = False
except ImportError:
    missing_geomet = True


def to_wkb_point(c):
    """
    Super quick hack that does not actually belong in here
    """
    point = {'type': 'Point', 'coordinates':[c[0], c[1]]}
    return geomet.wkb.dumps(point)


@ut.skipIf(missing_sql or missing_geomet,
           'missing dependencies: Geomet ({}) & SQLAlchemy ({})'.format(missing_geomet, missing_sql))
class Test_sqlite_reader(ut.TestCase):

    def setUp(self):
        df = pdio.read_files(pysal_examples.get_path('new_haven_merged.dbf'))
        df['GEOMETRY'] = df['geometry'].apply(to_wkb_point)
        del df['geometry'] # This is a hack to not have to worry about a custom point type in the DB
        engine = sqlalchemy.create_engine('sqlite:///test.db')
        conn = engine.connect()
        df.to_sql('newhaven', conn, index=True,
                  dtype={'date': sqlalchemy.types.UnicodeText,  # Should convert the df date into a true date object, just a hack again
                      'dataset': sqlalchemy.types.UnicodeText,
                      'street': sqlalchemy.types.UnicodeText,
                      'intersection': sqlalchemy.types.UnicodeText,
                      'time': sqlalchemy.types.UnicodeText,  # As above re: date
                      'GEOMETRY': sqlalchemy.types.BLOB})  # This is converted to TEXT as lowest type common sqlite

    def test_deserialize(self):
        db = psopen('sqlite:///test.db')
        self.assertEqual(db.tables, ['newhaven'])

        gj = db._get_gjson('newhaven')
        self.assertEqual(gj["type"], 'FeatureCollection')

    def tearDown(self):
        os.remove('test.db')

if __name__ == '__main__':
    ut.main()
