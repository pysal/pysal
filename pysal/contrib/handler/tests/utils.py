def setup(config=''):
    if config == ''

    db = pysal.open(pysal.examples.get_path("columbus.dbf"), "r")
    y = np.array(db.by_col("CRIME"))
    y = np.reshape(y, (49,1))
    X = []
    X.append(db.by_col_array(["HOVAL", "INC"]))
    X = np.array(X).T

    df = dbf2df(pysal.examples.get_path('columbus.dbf'))
    formula = 'CRIME ~ INC + HOVAL'

    reg = h.Model(formula, data=df)
    return db, y, X, df, formula, reg
