import sqlite3 as sql
import numpy as np
import os
import sys
from warnings import warn
try:
    import dill
    import pickle as pkl
except ImportError as E:
    msg = 'The `dill` module is required to use the sqlite backend fully.'
    warn(msg, stacklevel=2)
    import pickle as pkl
LEGACY_PYTHON = sys.version_info[0] < 3

CREATE_TEMPLATE = "CREATE TABLE {} (iteration INTEGER PRIMARY KEY, {})"
INSERT_TEMPLATE = "INSERT INTO {} VALUES (?, {})"

byte_type = str if LEGACY_PYTHON else bytes

def customize_create_template(colnames, tablename):
    """
    Change the CREATE TABLE statement to bind a set of column names and table name
    """
    return CREATE_TEMPLATE.format(tablename, ' , '.join(['"' + param + '"' + ' BLOB'
                                            for param in colnames]))
def customize_insert_template(colnames, tablename):
    """
    Change the INSERT INTO statement to bind to a set of column names and table name
    """
    data_part = ' , '.join(['?' for _ in colnames])
    return INSERT_TEMPLATE.format(tablename, data_part)

def start_sql(model, tracename='model_trace.db'):
    """
    Start a SQLite connection to a local database, specified by tracename.
    """
    if os.path.isfile(tracename):
        raise Exception('Will not overwrite existing trace {}'.format(tracename))
    cxn = sql.connect(tracename)
    cursor = cxn.cursor()
    cursor.execute(customize_create_template(model.traced_params, 'trace'))
    return cxn, cursor
    
def head_to_sql(model, cursor, connection):
    """
    Send the most recent trace point to the sql database.
    """
    point_to_sql(model, cursor, connection, index=-1)

def point_to_sql(model, cursor, connection, index=0):
    """
    Send an arbitrary index point to the database
    """
    if index < 0:
        iteration = model.cycles - (index + 1)
    else:
        iteration = index
    ordered_point = (serialize(model.trace[param, index]) for param in model.traced_params)
    to_insert = [iteration]
    to_insert.extend(list(ordered_point))
    cursor.execute(customize_insert_template(model.traced_params, 'trace'), tuple(to_insert))
    connection.commit()

def trace_to_sql(model, cursor, connection):
    """
    Send a model's entire trace to the database
    """
    for i in range(model.cycles):
        ordered_point = (serialize(model.trace[param, i]) for param in model.traced_params)
        to_insert = [i]
        to_insert.extend(list(ordered_point))
        cursor.execute(customize_insert_template(model.traced_params, 'trace'), tuple(to_insert))
    connection.commit()
    
def trace_from_sql(filename, table='trace'):
    """
    Reconstruct a model trace from the database
    """
    #connect, parse header, setup trace object, then deserialize the sql
    cxn = sql.connect(filename)
    pragma = cxn.execute('PRAGMA table_info({})'.format(table)).fetchall()
    colnames = [t[1] for t in pragma]
    data = cxn.execute('SELECT * FROM {}'.format(table)).fetchall()
    cxn.close()
    records = zip(colnames, map(list, zip(*data)))

    # Import must occur here otherwise there's a circularity issue

    from .abstracts import Trace

    if table == 'trace':
        out = Trace(**{colname:[maybe_deserialize(entry) for entry in column]
                  for colname, column in records})
    else:
        out = Trace(**{colname:maybe_deserialize(column[0])
                  for colname, column in records})
    return out

def model_to_sql(model, cursor, connection):
    """
    Serialize an entire model into a sqlite database. This serializes the trace
    into the `trace` table, the state into the `state` table, and the model class 
    into the `model` table. All items are pickled using their own dumps method, if possible. 
    Otherwise, objects are reduced using dill.dumps, which is then passed to sqlite as a BLOB
    """
    trace_to_sql(model, cursor, connection)
    frozen_state_keys = list(model.state.varnames)
    frozen_state = (serialize(model.state[k]) for k in frozen_state_keys)
    cursor.execute(customize_create_template(frozen_state_keys, 'state'))
    insert_template = customize_insert_template(frozen_state_keys, 'state')
    to_insert = [model.cycles]
    to_insert.extend(list(frozen_state))
    cursor.execute(insert_template, tuple(to_insert)) 
    class_pkl = pkl.dumps(model.__class__) #want instance, not whole model
    cursor.execute(customize_create_template(['model_class'], 'model'))
    cursor.execute(customize_insert_template(['class'], 'model'), (None, class_pkl))
    connection.commit()

def model_from_sql(filename):
    """
    Reconstruct a model from a sqlite table with a given trace, state, and model tables. 

    If the serialization fails for the trace or state, the resulting
    trace/state may contain raw binary strings. If the serialization fails for the model/there
    is no model table, the function will fail. To just extract the trace or the state,
    use trace_from_sql.
    """
    trace = trace_from_sql(filename)
    state = trace_from_sql(filename, table='state')
    cxn = sql.connect(filename)
    model_class = cxn.execute('SELECT model_class FROM model')
    model_class = pkl.loads(model_class.fetchall()[0][0])
    try:
        model_class(**state)
    except:
        warn('initializing model {} from state failed! '
             'Returning trace, state, model.'.format(model_class), stacklevel=2)
        return model_class, trace, state

def maybe_deserialize(maybe_bytestring):
    """
    This attempts to deserialize an object, but may return the original object 
    if no deserialization is successful. 
    """
    if isinstance(maybe_bytestring, (list, tuple)):
        return type(maybe_bytestring)([maybe_deserialize(byte_element) 
                                        for byte_element in maybe_bytestring])
    try:
        return pkl.loads(maybe_bytestring)
    except:
        try:
            return dill.loads(maybe_bytestring)
        except:
            try:
                return float(maybe_bytestring)
            except:
                return maybe_bytestring

def serialize(v):
    """
    This serializes an object, but may return the original object if serialization 
    is not successful. 
    """
    if hasattr(v, 'dumps'):
        return v.dumps()
    elif isinstance(v, (float, int)):
        return v
    else:
        return pkl.dumps(v)
