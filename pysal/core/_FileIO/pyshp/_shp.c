/* Includes */
#include <Python.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include "shapefil.h"

/* Prototypes */
static PyObject *create_dbf_file(PyObject *self, PyObject *args);

static PyObject *read_dbf_header(PyObject *self, PyObject *args);

static PyObject *read_dbf_record_count(PyObject *self, PyObject *args);

static PyObject *read_dbf_records(PyObject *self, PyObject *args);

static PyObject *write_dbf_records(PyObject *self, PyObject *args);

static PyObject *create_shpshx_file(PyObject *self, PyObject *args);

static PyObject *read_shpshx_type(PyObject *self, PyObject *args);

static PyObject *read_shpshx_record_count(PyObject *self, PyObject *args);

static PyObject *read_shpshx_records(PyObject *self, PyObject *args);

static PyObject *write_shpshx_records(PyObject *self, PyObject *args);

PyMODINIT_FUNC init_shp(void);

/* Python stuff */
PyMethodDef methods[] = 
{
    {"create_dbf_file", (PyCFunction) create_dbf_file, METH_VARARGS, "Creates a dbf file"},
    {"read_dbf_header", (PyCFunction) read_dbf_header, METH_VARARGS, "Returns the field names in the dbf file"},
    {"read_dbf_record_count", (PyCFunction) read_dbf_record_count, METH_VARARGS, "Returns the number of records in the dbf file"},
    {"read_dbf_records", (PyCFunction) read_dbf_records, METH_VARARGS, "Reads a set of dbf records"},
    {"write_dbf_records", (PyCFunction) write_dbf_records, METH_VARARGS, "Writes a set of dbf records"},
    {"create_shpshx_file", (PyCFunction) create_shpshx_file, METH_VARARGS, "Creates a shp/shpx file"},
    {"read_shpshx_type", (PyCFunction) read_shpshx_type, METH_VARARGS, "Returns the shape type of the shp/shx file"},
    {"read_shpshx_record_count", (PyCFunction) read_shpshx_record_count, METH_VARARGS, "Returns the number of records in the shp/shx file"},
    {"read_shpshx_records", (PyCFunction) read_shpshx_records, METH_VARARGS, "Reads a set of shp/shx records"},
    {"write_shpshx_records", (PyCFunction) write_shpshx_records, METH_VARARGS, "Writes a set of shp/shx records"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_shp(void)
{
    (void) Py_InitModule("_shp", methods);
}

double min(double a, double b)
{
    if (a < b)
        return a;
    else
        return b;
}

static PyObject *read_shpshx_record_count(PyObject *self, PyObject *args)
{
    SHPHandle shp_handle;
    char *shp_name_ptr;
    int shp_record_count, shape_type;

    shp_name_ptr = PyString_AsString(PyTuple_GetItem(args, 0));	
    shp_handle = SHPOpen(shp_name_ptr, "rb");

    if(shp_handle == NULL) 
	return 0;

    SHPGetInfo(shp_handle, &shp_record_count, &shape_type, NULL, NULL);

    SHPClose(shp_handle);

    return Py_BuildValue("i", shp_record_count);
}

static PyObject *read_shpshx_type(PyObject *self, PyObject *args)
{
    SHPHandle shp_handle;
    char *shp_name, *type;
    int shape_type;   

    shp_name = PyString_AsString(PyTuple_GetItem(args, 0));
 
    shp_handle = SHPOpen(shp_name, "rb");
    if(shp_handle == NULL)
	return Py_None;

    SHPGetInfo(shp_handle, NULL, &shape_type, NULL, NULL);

    switch(shape_type)
    {
	case SHPT_POINT:
		type = "POINT";
		break;
	case SHPT_POINTZ:
		type = "POINTZ";
		break;
	case SHPT_POINTM:
		type = "POINTM";
		break;
	case SHPT_ARC:
		type = "ARC";
		break;
	case SHPT_ARCZ:
		type = "ARCZ";
		break;
	case SHPT_ARCM:
		type = "ARCM";
		break;
	case SHPT_POLYGON:
		type = "POLYGON";
		break;
	case SHPT_POLYGONZ:
		type = "POLYGONZ";
		break;
	case SHPT_POLYGONM:
		type = "POLYGONM";
		break;
	case SHPT_MULTIPOINT:
		type = "MULTIPOINT";
		break;
	case SHPT_MULTIPOINTZ:
		type = "MULTIPOINTZ";
		break;
	case SHPT_MULTIPOINTM:
		type = "MULTIPOINTM";
		break;
	case SHPT_MULTIPATCH:
		type = "MULTIPATCH";
		break;
	default:
	    type = "NULL";
	    break;
    }

    SHPClose(shp_handle);
    return Py_BuildValue("s", type);
}

int shp_type_convert(char *type_string)
{
    int shape_type;

    if(strcmp(type_string, "POINT") == 0)
    	shape_type = SHPT_POINT;
    else if (strcmp(type_string, "POINTZ") == 0)
    	shape_type = SHPT_POINTZ;
    else if (strcmp(type_string, "POINTM") == 0)
    	shape_type = SHPT_POINTM;
    else if (strcmp(type_string, "ARC") == 0)
    	shape_type = SHPT_ARC;
    else if (strcmp(type_string, "ARCZ") == 0)
    	shape_type = SHPT_ARCZ;
    else if (strcmp(type_string, "ARCM") == 0)
    	shape_type = SHPT_ARCM;
    else if (strcmp(type_string, "POLYGON") == 0)
    	shape_type = SHPT_POLYGON;
    else if (strcmp(type_string, "POLYGONZ") == 0)
    	shape_type = SHPT_POLYGONZ;
    else if (strcmp(type_string, "POLYGONM") == 0)
    	shape_type = SHPT_POLYGONM;
    else if (strcmp(type_string, "MULTIPOINT") == 0)
    	shape_type = SHPT_MULTIPOINT;
    else if (strcmp(type_string, "MULTIPOINTZ") == 0)
    	shape_type = SHPT_MULTIPOINTZ;
    else if (strcmp(type_string, "MULTIPOINTM") == 0)
    	shape_type = SHPT_MULTIPOINTM;
    else
	shape_type = SHPT_NULL;
    
    return shape_type;
}

static PyObject *create_shpshx_file(PyObject *self, PyObject *args)
{
    SHPHandle shp_handle;
    char *shp_name;
    int shp_type;  
 
    shp_name = PyString_AsString(PyTuple_GetItem(args, 0));
    shp_type = (int) shp_type_convert(PyString_AsString(PyTuple_GetItem(args, 1)));
 
    shp_handle = SHPCreate(shp_name, shp_type);
    SHPClose(shp_handle);

    return Py_None;
}


static PyObject *write_shpshx_records(PyObject *self, PyObject *args)
{
    SHPHandle shp_handle;
    SHPObject *cur_shp_obj;
    PyObject *cur_py_rec, *py_record_list, *py_pan_part;
    char *shp_name;
    int i, p, nSHPType, nShapeId, nParts, nVertices, record_count;
    int *panPartStart, *panPartType; 
    double *padfX, *padfY, *padfZ, *padfM;

    shp_name = PyString_AsString(PyTuple_GetItem(args, 0));	
    py_record_list = PyTuple_GetItem(args, 1);
    shp_handle = SHPOpen(shp_name, "rb+");

    if(shp_handle == NULL) 
	return Py_None;

    SHPGetInfo(shp_handle, &record_count, NULL, NULL, NULL);

    for(i = 0; i < PyList_Size(py_record_list); i++)
    {
    	cur_py_rec = PyList_GetItem(py_record_list, i);
	nSHPType = shp_type_convert(PyString_AsString(PyList_GetItem(cur_py_rec, 0)));
	nShapeId = -1; /* Attempting to append */
	nParts = PyList_Size(PyList_GetItem(cur_py_rec, 2));
	panPartStart = PyMem_Malloc(sizeof(int)*nParts);
	panPartType = PyMem_Malloc(sizeof(int)*nParts);
	for(p = 0; p < nParts; p++)	
	{
	    py_pan_part = PyList_GetItem(PyList_GetItem(cur_py_rec, 2), p);
	    panPartStart[p] = (int) PyInt_AsLong(PyTuple_GetItem(py_pan_part, 0));
	    if (strcmp(PyString_AsString(PyTuple_GetItem(py_pan_part, 1)), "MULTIPATCH") == 0)
		panPartType[p] = SHPT_MULTIPATCH;
	    else
		panPartType[p] = SHPP_RING;
	}
    
	nVertices = PyList_Size(PyList_GetItem(cur_py_rec, 3));
	padfX = PyMem_Malloc(sizeof(double)*nVertices);
	padfY = PyMem_Malloc(sizeof(double)*nVertices);
	padfZ = PyMem_Malloc(sizeof(double)*nVertices);
	padfM = PyMem_Malloc(sizeof(double)*nVertices);
	for(p = 0; p < nVertices; p++)
	{
	    padfX[p] = PyFloat_AsDouble(PyTuple_GetItem(PyList_GetItem(PyList_GetItem(cur_py_rec, 3), p), 0));     
	    padfY[p] = PyFloat_AsDouble(PyTuple_GetItem(PyList_GetItem(PyList_GetItem(cur_py_rec, 3), p), 1));     
	    padfZ[p] = 0;
	    padfM[p] = 0;
	}	

	cur_shp_obj = SHPCreateObject(nSHPType, nShapeId, nParts, panPartStart, panPartType, nVertices, 
	    padfX, padfY, padfZ, padfM); 
	SHPWriteObject(shp_handle, nShapeId, cur_shp_obj); 
	SHPDestroyObject(cur_shp_obj); 
	PyMem_Free(padfX);    	    
	PyMem_Free(padfY);    	    
	PyMem_Free(padfZ);    	    
	PyMem_Free(padfM); 
    } 

    SHPClose(shp_handle);
    return Py_BuildValue("i", i);
}


static PyObject *read_shpshx_records(PyObject *self, PyObject *args)
{
    SHPHandle shp_handle;
    SHPObject *cur_shp_obj;
    PyObject *cur_py_obj, *py_obj_list, *tmp;
    char *shp_name_ptr;
    int i, j, start_rec, end_rec, shp_record_count, shape_type;

    shp_name_ptr = PyString_AsString(PyTuple_GetItem(args, 0));	
    start_rec = PyInt_AsLong(PyTuple_GetItem(args, 1)); 
    end_rec = PyInt_AsLong(PyTuple_GetItem(args, 2));
    shp_handle = SHPOpen(shp_name_ptr, "rb");

    py_obj_list = PyList_New(0);

    if(shp_handle == NULL) 
	return py_obj_list;

    SHPGetInfo(shp_handle, &shp_record_count, &shape_type, NULL, NULL);

    if(start_rec < 1)
	start_rec = 1;
    if(shp_record_count < end_rec)
	end_rec = shp_record_count;

    /* Shift to zero-based index */
    start_rec--;

    for(i = start_rec; i < end_rec; i++)
    {
    	cur_py_obj = PyList_New(0);
	cur_shp_obj = SHPReadObject(shp_handle, i);

	/* Get .shp elements */
	switch(cur_shp_obj->nSHPType)
 	{
	    case SHPT_POINT:
	        PyList_Append(cur_py_obj, Py_BuildValue("s", "POINT"));
		break;
	    case SHPT_POINTZ:
	        PyList_Append(cur_py_obj, Py_BuildValue("s", "POINTZ"));
		break;
	    case SHPT_POINTM:
	        PyList_Append(cur_py_obj, Py_BuildValue("s", "POINTM"));
		break;
	    case SHPT_ARC:
	        PyList_Append(cur_py_obj, Py_BuildValue("s", "ARC"));
		break;
	    case SHPT_ARCZ:
	        PyList_Append(cur_py_obj, Py_BuildValue("s", "ARCZ"));
		break;
	    case SHPT_ARCM:
	        PyList_Append(cur_py_obj, Py_BuildValue("s", "ARCM"));
		break;
	    case SHPT_POLYGON:
		PyList_Append(cur_py_obj, Py_BuildValue("s", "POLYGON"));
		break;
	    case SHPT_POLYGONZ:
		PyList_Append(cur_py_obj, Py_BuildValue("s", "POLYGONZ"));
		break;
	    case SHPT_POLYGONM:
		PyList_Append(cur_py_obj, Py_BuildValue("s", "POLYGONM"));
		break;
	    case SHPT_MULTIPOINT:
		PyList_Append(cur_py_obj, Py_BuildValue("s", "MULTIPOINT"));
		break;
	    case SHPT_MULTIPOINTZ:
		PyList_Append(cur_py_obj, Py_BuildValue("s", "MULTIPOINTZ"));
		break;
	    case SHPT_MULTIPOINTM:
		PyList_Append(cur_py_obj, Py_BuildValue("s", "MULTIPOINTM"));
		break;
	    case SHPT_MULTIPATCH:
		PyList_Append(cur_py_obj, Py_BuildValue("s", "MULTIPATCH"));
		break;
	    default:
		PyList_Append(cur_py_obj, Py_BuildValue("i", cur_shp_obj->nSHPType));
		break;
	}

	PyList_Append(cur_py_obj, Py_BuildValue("i", cur_shp_obj->nShapeId));

	tmp = PyList_New(0);
	for(j = 0; j < cur_shp_obj->nParts; j++)
	{
	    if(cur_shp_obj->panPartType[j] == SHPT_MULTIPATCH)
	        PyList_Append(tmp, Py_BuildValue("(is)", cur_shp_obj->panPartStart[j], "MULTIPATCH"));
	    else
	        PyList_Append(tmp, Py_BuildValue("(is)", cur_shp_obj->panPartStart[j], "RING"));
	}
	PyList_Append(cur_py_obj, tmp);


	tmp = PyList_New(0);
	for(j = 0; j < cur_shp_obj->nVertices; j++)
	    PyList_Append(tmp, Py_BuildValue("(ff)", cur_shp_obj->padfX[j], 
	   	cur_shp_obj->padfY[j]/*, cur_shp_obj->padfZ[j], cur_shp_obj->padfM[j]*/));
	PyList_Append(cur_py_obj, tmp);

	PyList_Append(cur_py_obj, Py_BuildValue("(ff)", cur_shp_obj->dfXMin, 
	    cur_shp_obj->dfYMin/*, cur_shp_obj->dfZMin, cur_shp_obj->dfMMin*/));
		
	PyList_Append(cur_py_obj, Py_BuildValue("(ff)", cur_shp_obj->dfXMax, 
	    cur_shp_obj->dfYMax/*, cur_shp_obj->dfZMax, cur_shp_obj->dfMMax*/));

	PyList_Append(py_obj_list, cur_py_obj);
    } 

    SHPClose(shp_handle);

    return py_obj_list;
}


static PyObject *read_dbf_record_count(PyObject *self, PyObject *args)
{
    DBFHandle dbf_handle;
    char *dbf_name_ptr;

    /* To get rid of compile warnings */
    dbf_name_ptr = PyString_AsString(PyTuple_GetItem(args, 0));	
    dbf_handle = DBFOpen(dbf_name_ptr, "rb");

    if(dbf_handle == NULL)
	return 0;    

    DBFClose(dbf_handle);

    return Py_BuildValue("i", DBFGetRecordCount(dbf_handle));
}


static PyObject *create_dbf_file(PyObject *self, PyObject *args)
{
    DBFHandle dbf_handle;
    PyObject *py_fields, *py_type_obj;
    char *dbf_name, *field_name;
    int i;

    dbf_name = PyString_AsString(PyTuple_GetItem(args, 0));
    py_fields = PyTuple_GetItem(args, 1); 

    dbf_handle = DBFCreate(dbf_name);
    
    for(i = 0; i < PyList_Size(py_fields); i++)
    {	
	field_name = PyString_AsString(PyTuple_GetItem(PyList_GetItem(py_fields, i), 0));
	py_type_obj = PyTuple_GetItem(PyList_GetItem(py_fields, i), 1);
    	if(PyString_Check(py_type_obj))
	{
	    DBFAddField(dbf_handle, field_name, FTString, 20, 0); /* Be dumb and guess at good size */
	}  
    	else if(PyInt_Check(py_type_obj))
        {
	    DBFAddField(dbf_handle, field_name, FTInteger, 20, 0); /* Be dumb and guess at good size */
	}
    	if(PyFloat_Check(py_type_obj))
	{
	    DBFAddField(dbf_handle, field_name, FTDouble, 20, 8); /* Be dumb and guess at good size */
	}
	else
	    continue;	
    }

    DBFClose(dbf_handle); 

    return Py_None;
}


static PyObject *write_dbf_records(PyObject *self, PyObject *args)
{
    DBFHandle dbf_handle;
    PyObject *cur_py_rec_field, *cur_py_record, *py_record_list;
    char *dbf_name_ptr;
    long i, j, dbf_record_count;

    dbf_name_ptr = PyString_AsString(PyTuple_GetItem(args, 0));	
    py_record_list = PyTuple_GetItem(args, 1);

    dbf_handle = DBFOpen(dbf_name_ptr, "rb+");
    if(dbf_handle == NULL)
	return Py_None; 

    dbf_record_count = DBFGetRecordCount(dbf_handle);

    for(i = 0; i < PyList_Size(py_record_list); i++)
    {
	cur_py_record = PyList_GetItem(py_record_list, i);	
	
	/* Write .dbf elements */
	for(j = 0; j < DBFGetFieldCount(dbf_handle); j++)
	{
	    cur_py_rec_field = PyList_GetItem(cur_py_record, j);

	    if(PyString_Check(cur_py_rec_field))
		DBFWriteStringAttribute(dbf_handle, i + dbf_record_count, j, PyString_AsString(cur_py_rec_field));
	    else if(PyInt_Check(cur_py_rec_field))
		DBFWriteIntegerAttribute(dbf_handle, i + dbf_record_count, j, (int) PyInt_AsLong(cur_py_rec_field));
	    else if(PyFloat_Check(cur_py_rec_field))
		DBFWriteDoubleAttribute(dbf_handle, i + dbf_record_count, j, PyFloat_AsDouble(cur_py_rec_field));
	    else
		DBFWriteNULLAttribute(dbf_handle, i + dbf_record_count, j);
	}
    }

    DBFClose(dbf_handle);
    return Py_BuildValue("i", 0);
}

static PyObject *read_dbf_records(PyObject *self, PyObject *args)
{
    DBFHandle dbf_handle;
    PyObject *cur_py_obj, *py_obj_list;
    char *dbf_name_ptr;
    long i, j, dbf_record_count, start_rec, end_rec;

    dbf_name_ptr = PyString_AsString(PyTuple_GetItem(args, 0));	
    dbf_handle = DBFOpen(dbf_name_ptr, "rb");
    start_rec = PyInt_AsLong(PyTuple_GetItem(args, 1)); 
    end_rec = PyInt_AsLong(PyTuple_GetItem(args, 2));

    py_obj_list = PyList_New(0);

    if(dbf_handle == NULL)
	return py_obj_list;    

    dbf_record_count = DBFGetRecordCount(dbf_handle);

    if(start_rec < 1)
	start_rec = 1;
    if(dbf_record_count < end_rec)
	end_rec = dbf_record_count;

    /* Shift to zero-based index */
    start_rec--;
   
    for(i = start_rec; i < end_rec; i++)
    {
    	cur_py_obj = PyList_New(0);
	
	/* Get .dbf elements */
	for(j = 0; j < DBFGetFieldCount(dbf_handle); j++)
	{
	    switch(DBFGetFieldInfo(dbf_handle, j, NULL, NULL, NULL))
	    {
	        case FTString:
		    PyList_Append(cur_py_obj, Py_BuildValue("s", DBFReadStringAttribute(dbf_handle, i, j)));
		    break;

		case FTInteger:
		    PyList_Append(cur_py_obj, Py_BuildValue("i", DBFReadIntegerAttribute(dbf_handle, i, j)));
		    break;

	 	case FTDouble:
		    PyList_Append(cur_py_obj, Py_BuildValue("f", DBFReadDoubleAttribute(dbf_handle, i, j)));
		    break;

		default:
		    PyList_Append(cur_py_obj, Py_None);
		    break;
	    }	

	}
	   
        PyList_Append(py_obj_list, cur_py_obj);

    }

    DBFClose(dbf_handle);
	
    return py_obj_list;
}


static PyObject *read_dbf_header(PyObject *self, PyObject *args)
{
    DBFHandle dbf_handle;
    PyObject *py_dbf_header;
    char *dbf_name_ptr, field_buf[100];
    long i;

    /* To get rid of compile warnings */
    dbf_name_ptr = PyString_AsString(PyTuple_GetItem(args, 0));	
    dbf_handle = DBFOpen(dbf_name_ptr, "rb");

    py_dbf_header = PyList_New(0);

    if(dbf_handle == NULL)
	return py_dbf_header;    

    for(i = 0; i < DBFGetFieldCount(dbf_handle); i++)
    {
    	DBFGetFieldInfo(dbf_handle, i, field_buf, NULL, NULL);
	PyList_Append(py_dbf_header, Py_BuildValue("s", field_buf));
    } 

    DBFClose(dbf_handle);

    return py_dbf_header;
}










/******************** SHAPELIB SOURCE BELOW ONLY *********************/










/******************************************************************************
 * $Id: shpopen.c,v 1.39 2002/08/26 06:46:56 warmerda Exp $
 *
 * Project:  Shapelib
 * Purpose:  Implementation of core Shapefile read/write functions.
 * Author:   Frank Warmerdam, warmerdam@pobox.com
 *
 ******************************************************************************
 * Copyright (c) 1999, 2001, Frank Warmerdam
 *
 * This software is available under the following "MIT Style" license,
 * or at the option of the licensee under the LGPL (see LICENSE.LGPL).  This
 * option is discussed in more detail in shapelib.html.
 *
 * --
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ******************************************************************************
 *
 * $Log: shpopen.c,v $
 * Revision 1.39  2002/08/26 06:46:56  warmerda
 * avoid c++ comments
 *
 * Revision 1.38  2002/05/07 16:43:39  warmerda
 * Removed debugging printf.
 *
 * Revision 1.37  2002/04/10 17:35:22  warmerda
 * fixed bug in ring reversal code
 *
 * Revision 1.36  2002/04/10 16:59:54  warmerda
 * added SHPRewindObject
 *
 * Revision 1.35  2001/12/07 15:10:44  warmerda
 * fix if .shx fails to open
 *
 * Revision 1.34  2001/11/01 16:29:55  warmerda
 * move pabyRec into SHPInfo for thread safety
 *
 * Revision 1.33  2001/07/03 12:18:15  warmerda
 * Improved cleanup if SHX not found, provied by Riccardo Cohen.
 *
 * Revision 1.32  2001/06/22 01:58:07  warmerda
 * be more careful about establishing initial bounds in face of NULL shapes
 *
 * Revision 1.31  2001/05/31 19:35:29  warmerda
 * added support for writing null shapes
 *
 * Revision 1.30  2001/05/28 12:46:29  warmerda
 * Add some checking on reasonableness of record count when opening.
 *
 * Revision 1.29  2001/05/23 13:36:52  warmerda
 * added use of SHPAPI_CALL
 *
 * Revision 1.28  2001/02/06 22:25:06  warmerda
 * fixed memory leaks when SHPOpen() fails
 *
 * Revision 1.27  2000/07/18 15:21:33  warmerda
 * added better enforcement of -1 for append in SHPWriteObject
 *
 * Revision 1.26  2000/02/16 16:03:51  warmerda
 * added null shape support
 *
 * Revision 1.25  1999/12/15 13:47:07  warmerda
 * Fixed record size settings in .shp file (was 4 words too long)
 * Added stdlib.h.
 *
 * Revision 1.24  1999/11/05 14:12:04  warmerda
 * updated license terms
 *
 * Revision 1.23  1999/07/27 00:53:46  warmerda
 * added support for rewriting shapes
 *
 * Revision 1.22  1999/06/11 19:19:11  warmerda
 * Cleanup pabyRec static buffer on SHPClose().
 *
 * Revision 1.21  1999/06/02 14:57:56  kshih
 * Remove unused variables
 *
 * Revision 1.20  1999/04/19 21:04:17  warmerda
 * Fixed syntax error.
 *
 * Revision 1.19  1999/04/19 21:01:57  warmerda
 * Force access string to binary in SHPOpen().
 *
 * Revision 1.18  1999/04/01 18:48:07  warmerda
 * Try upper case extensions if lower case doesn't work.
 *
 * Revision 1.17  1998/12/31 15:29:39  warmerda
 * Disable writing measure values to multipatch objects if
 * DISABLE_MULTIPATCH_MEASURE is defined.
 *
 * Revision 1.16  1998/12/16 05:14:33  warmerda
 * Added support to write MULTIPATCH.  Fixed reading Z coordinate of
 * MULTIPATCH. Fixed record size written for all feature types.
 *
 * Revision 1.15  1998/12/03 16:35:29  warmerda
 * r+b is proper binary access string, not rb+.
 *
 * Revision 1.14  1998/12/03 15:47:56  warmerda
 * Fixed setting of nVertices in SHPCreateObject().
 *
 * Revision 1.13  1998/12/03 15:33:54  warmerda
 * Made SHPCalculateExtents() separately callable.
 *
 * Revision 1.12  1998/11/11 20:01:50  warmerda
 * Fixed bug writing ArcM/Z, and PolygonM/Z for big endian machines.
 *
 * Revision 1.11  1998/11/09 20:56:44  warmerda
 * Fixed up handling of file wide bounds.
 *
 * Revision 1.10  1998/11/09 20:18:51  warmerda
 * Converted to support 3D shapefiles, and use of SHPObject.
 *
 * Revision 1.9  1998/02/24 15:09:05  warmerda
 * Fixed memory leak.
 *
 * Revision 1.8  1997/12/04 15:40:29  warmerda
 * Fixed byte swapping of record number, and record length fields in the
 * .shp file.
 *
 * Revision 1.7  1995/10/21 03:15:58  warmerda
 * Added support for binary file access, the magic cookie 9997
 * and tried to improve the int32 selection logic for 16bit systems.
 *
 * Revision 1.6  1995/09/04  04:19:41  warmerda
 * Added fix for file bounds.
 *
 * Revision 1.5  1995/08/25  15:16:44  warmerda
 * Fixed a couple of problems with big endian systems ... one with bounds
 * and the other with multipart polygons.
 *
 * Revision 1.4  1995/08/24  18:10:17  warmerda
 * Switch to use SfRealloc() to avoid problems with pre-ANSI realloc()
 * functions (such as on the Sun).
 *
 * Revision 1.3  1995/08/23  02:23:15  warmerda
 * Added support for reading bounds, and fixed up problems in setting the
 * file wide bounds.
 *
 * Revision 1.2  1995/08/04  03:16:57  warmerda
 * Added header.
 *
 */



typedef unsigned char uchar;

#if UINT_MAX == 65535
typedef long	      int32;
#else
typedef int	      int32;
#endif

#ifndef FALSE
#  define FALSE		0
#  define TRUE		1
#endif

#define ByteCopy( a, b, c )	memcpy( b, a, c )
#ifndef MAX
#  define MIN(a,b)      ((a<b) ? a : b)
#  define MAX(a,b)      ((a>b) ? a : b)
#endif

static int 	bBigEndian;


/************************************************************************/
/*                              SwapWord()                              */
/*                                                                      */
/*      Swap a 2, 4 or 8 byte word.                                     */
/************************************************************************/

static void	SwapWord( int length, void * wordP )

{
    int		i;
    uchar	temp;

    for( i=0; i < length/2; i++ )
    {
	temp = ((uchar *) wordP)[i];
	((uchar *)wordP)[i] = ((uchar *) wordP)[length-i-1];
	((uchar *) wordP)[length-i-1] = temp;
    }
}

/************************************************************************/
/*                             SfRealloc()                              */
/*                                                                      */
/*      A realloc cover function that will access a NULL pointer as     */
/*      a valid input.                                                  */
/************************************************************************/

static void * SfRealloc( void * pMem, int nNewSize )

{
    if( pMem == NULL )
        return( (void *) malloc(nNewSize) );
    else
        return( (void *) realloc(pMem,nNewSize) );
}

/************************************************************************/
/*                          SHPWriteHeader()                            */
/*                                                                      */
/*      Write out a header for the .shp and .shx files as well as the	*/
/*	contents of the index (.shx) file.				*/
/************************************************************************/

static void SHPWriteHeader( SHPHandle psSHP )

{
    uchar     	abyHeader[100];
    int		i;
    int32	i32;
    double	dValue;
    int32	*panSHX;

/* -------------------------------------------------------------------- */
/*      Prepare header block for .shp file.                             */
/* -------------------------------------------------------------------- */
    for( i = 0; i < 100; i++ )
      abyHeader[i] = 0;

    abyHeader[2] = 0x27;				/* magic cookie */
    abyHeader[3] = 0x0a;

    i32 = psSHP->nFileSize/2;				/* file size */
    ByteCopy( &i32, abyHeader+24, 4 );
    if( !bBigEndian ) SwapWord( 4, abyHeader+24 );
    
    i32 = 1000;						/* version */
    ByteCopy( &i32, abyHeader+28, 4 );
    if( bBigEndian ) SwapWord( 4, abyHeader+28 );
    
    i32 = psSHP->nShapeType;				/* shape type */
    ByteCopy( &i32, abyHeader+32, 4 );
    if( bBigEndian ) SwapWord( 4, abyHeader+32 );

    dValue = psSHP->adBoundsMin[0];			/* set bounds */
    ByteCopy( &dValue, abyHeader+36, 8 );
    if( bBigEndian ) SwapWord( 8, abyHeader+36 );

    dValue = psSHP->adBoundsMin[1];
    ByteCopy( &dValue, abyHeader+44, 8 );
    if( bBigEndian ) SwapWord( 8, abyHeader+44 );

    dValue = psSHP->adBoundsMax[0];
    ByteCopy( &dValue, abyHeader+52, 8 );
    if( bBigEndian ) SwapWord( 8, abyHeader+52 );

    dValue = psSHP->adBoundsMax[1];
    ByteCopy( &dValue, abyHeader+60, 8 );
    if( bBigEndian ) SwapWord( 8, abyHeader+60 );

    dValue = psSHP->adBoundsMin[2];			/* z */
    ByteCopy( &dValue, abyHeader+68, 8 );
    if( bBigEndian ) SwapWord( 8, abyHeader+68 );

    dValue = psSHP->adBoundsMax[2];
    ByteCopy( &dValue, abyHeader+76, 8 );
    if( bBigEndian ) SwapWord( 8, abyHeader+76 );

    dValue = psSHP->adBoundsMin[3];			/* m */
    ByteCopy( &dValue, abyHeader+84, 8 );
    if( bBigEndian ) SwapWord( 8, abyHeader+84 );

    dValue = psSHP->adBoundsMax[3];
    ByteCopy( &dValue, abyHeader+92, 8 );
    if( bBigEndian ) SwapWord( 8, abyHeader+92 );

/* -------------------------------------------------------------------- */
/*      Write .shp file header.                                         */
/* -------------------------------------------------------------------- */
    fseek( psSHP->fpSHP, 0, 0 );
    fwrite( abyHeader, 100, 1, psSHP->fpSHP );

/* -------------------------------------------------------------------- */
/*      Prepare, and write .shx file header.                            */
/* -------------------------------------------------------------------- */
    i32 = (psSHP->nRecords * 2 * sizeof(int32) + 100)/2;   /* file size */
    ByteCopy( &i32, abyHeader+24, 4 );
    if( !bBigEndian ) SwapWord( 4, abyHeader+24 );
    
    fseek( psSHP->fpSHX, 0, 0 );
    fwrite( abyHeader, 100, 1, psSHP->fpSHX );

/* -------------------------------------------------------------------- */
/*      Write out the .shx contents.                                    */
/* -------------------------------------------------------------------- */
    panSHX = (int32 *) malloc(sizeof(int32) * 2 * psSHP->nRecords);

    for( i = 0; i < psSHP->nRecords; i++ )
    {
	panSHX[i*2  ] = psSHP->panRecOffset[i]/2;
	panSHX[i*2+1] = psSHP->panRecSize[i]/2;
	if( !bBigEndian ) SwapWord( 4, panSHX+i*2 );
	if( !bBigEndian ) SwapWord( 4, panSHX+i*2+1 );
    }

    fwrite( panSHX, sizeof(int32) * 2, psSHP->nRecords, psSHP->fpSHX );

    free( panSHX );
}

/************************************************************************/
/*                              SHPOpen()                               */
/*                                                                      */
/*      Open the .shp and .shx files based on the basename of the       */
/*      files or either file name.                                      */
/************************************************************************/
   
SHPHandle SHPAPI_CALL
SHPOpen( const char * pszLayer, const char * pszAccess )

{
    char		*pszFullname, *pszBasename;
    SHPHandle		psSHP;
    
    uchar		*pabyBuf;
    int			i;
    double		dValue;
    
/* -------------------------------------------------------------------- */
/*      Ensure the access string is one of the legal ones.  We          */
/*      ensure the result string indicates binary to avoid common       */
/*      problems on Windows.                                            */
/* -------------------------------------------------------------------- */
    if( strcmp(pszAccess,"rb+") == 0 || strcmp(pszAccess,"r+b") == 0
        || strcmp(pszAccess,"r+") == 0 )
        pszAccess = "r+b";
    else
        pszAccess = "rb";
    
/* -------------------------------------------------------------------- */
/*	Establish the byte order on this machine.			*/
/* -------------------------------------------------------------------- */
    i = 1;
    if( *((uchar *) &i) == 1 )
        bBigEndian = FALSE;
    else
        bBigEndian = TRUE;

/* -------------------------------------------------------------------- */
/*	Initialize the info structure.					*/
/* -------------------------------------------------------------------- */
    psSHP = (SHPHandle) calloc(sizeof(SHPInfo),1);

    psSHP->bUpdated = FALSE;

/* -------------------------------------------------------------------- */
/*	Compute the base (layer) name.  If there is any extension	*/
/*	on the passed in filename we will strip it off.			*/
/* -------------------------------------------------------------------- */
    pszBasename = (char *) malloc(strlen(pszLayer)+5);
    strcpy( pszBasename, pszLayer );
    for( i = strlen(pszBasename)-1; 
	 i > 0 && pszBasename[i] != '.' && pszBasename[i] != '/'
	       && pszBasename[i] != '\\';
	 i-- ) {}

    if( pszBasename[i] == '.' )
        pszBasename[i] = '\0';

/* -------------------------------------------------------------------- */
/*	Open the .shp and .shx files.  Note that files pulled from	*/
/*	a PC to Unix with upper case filenames won't work!		*/
/* -------------------------------------------------------------------- */
    pszFullname = (char *) malloc(strlen(pszBasename) + 5);
    sprintf( pszFullname, "%s.shp", pszBasename );
    psSHP->fpSHP = fopen(pszFullname, pszAccess );
    if( psSHP->fpSHP == NULL )
    {
        sprintf( pszFullname, "%s.SHP", pszBasename );
        psSHP->fpSHP = fopen(pszFullname, pszAccess );
    }
    
    if( psSHP->fpSHP == NULL )
    {
        free( psSHP );
        free( pszBasename );
        free( pszFullname );
        return( NULL );
    }

    sprintf( pszFullname, "%s.shx", pszBasename );
    psSHP->fpSHX = fopen(pszFullname, pszAccess );
    if( psSHP->fpSHX == NULL )
    {
        sprintf( pszFullname, "%s.SHX", pszBasename );
        psSHP->fpSHX = fopen(pszFullname, pszAccess );
    }
    
    if( psSHP->fpSHX == NULL )
    {
        fclose( psSHP->fpSHP );
        free( psSHP );
        free( pszBasename );
        free( pszFullname );
        return( NULL );
    }

    free( pszFullname );
    free( pszBasename );

/* -------------------------------------------------------------------- */
/*  Read the file size from the SHP file.				*/
/* -------------------------------------------------------------------- */
    pabyBuf = (uchar *) malloc(100);
    fread( pabyBuf, 100, 1, psSHP->fpSHP );

    psSHP->nFileSize = (pabyBuf[24] * 256 * 256 * 256
			+ pabyBuf[25] * 256 * 256
			+ pabyBuf[26] * 256
			+ pabyBuf[27]) * 2;

/* -------------------------------------------------------------------- */
/*  Read SHX file Header info                                           */
/* -------------------------------------------------------------------- */
    fread( pabyBuf, 100, 1, psSHP->fpSHX );

    if( pabyBuf[0] != 0 
        || pabyBuf[1] != 0 
        || pabyBuf[2] != 0x27 
        || (pabyBuf[3] != 0x0a && pabyBuf[3] != 0x0d) )
    {
	fclose( psSHP->fpSHP );
	fclose( psSHP->fpSHX );
	free( psSHP );

	return( NULL );
    }

    psSHP->nRecords = pabyBuf[27] + pabyBuf[26] * 256
      + pabyBuf[25] * 256 * 256 + pabyBuf[24] * 256 * 256 * 256;
    psSHP->nRecords = (psSHP->nRecords*2 - 100) / 8;

    psSHP->nShapeType = pabyBuf[32];

    if( psSHP->nRecords < 0 || psSHP->nRecords > 256000000 )
    {
        /* this header appears to be corrupt.  Give up. */
	fclose( psSHP->fpSHP );
	fclose( psSHP->fpSHX );
	free( psSHP );

	return( NULL );
    }

/* -------------------------------------------------------------------- */
/*      Read the bounds.                                                */
/* -------------------------------------------------------------------- */
    if( bBigEndian ) SwapWord( 8, pabyBuf+36 );
    memcpy( &dValue, pabyBuf+36, 8 );
    psSHP->adBoundsMin[0] = dValue;

    if( bBigEndian ) SwapWord( 8, pabyBuf+44 );
    memcpy( &dValue, pabyBuf+44, 8 );
    psSHP->adBoundsMin[1] = dValue;

    if( bBigEndian ) SwapWord( 8, pabyBuf+52 );
    memcpy( &dValue, pabyBuf+52, 8 );
    psSHP->adBoundsMax[0] = dValue;

    if( bBigEndian ) SwapWord( 8, pabyBuf+60 );
    memcpy( &dValue, pabyBuf+60, 8 );
    psSHP->adBoundsMax[1] = dValue;

    if( bBigEndian ) SwapWord( 8, pabyBuf+68 );		/* z */
    memcpy( &dValue, pabyBuf+68, 8 );
    psSHP->adBoundsMin[2] = dValue;
    
    if( bBigEndian ) SwapWord( 8, pabyBuf+76 );
    memcpy( &dValue, pabyBuf+76, 8 );
    psSHP->adBoundsMax[2] = dValue;
    
    if( bBigEndian ) SwapWord( 8, pabyBuf+84 );		/* z */
    memcpy( &dValue, pabyBuf+84, 8 );
    psSHP->adBoundsMin[3] = dValue;

    if( bBigEndian ) SwapWord( 8, pabyBuf+92 );
    memcpy( &dValue, pabyBuf+92, 8 );
    psSHP->adBoundsMax[3] = dValue;

    free( pabyBuf );

/* -------------------------------------------------------------------- */
/*	Read the .shx file to get the offsets to each record in 	*/
/*	the .shp file.							*/
/* -------------------------------------------------------------------- */
    psSHP->nMaxRecords = psSHP->nRecords;

    psSHP->panRecOffset =
        (int *) malloc(sizeof(int) * MAX(1,psSHP->nMaxRecords) );
    psSHP->panRecSize =
        (int *) malloc(sizeof(int) * MAX(1,psSHP->nMaxRecords) );

    pabyBuf = (uchar *) malloc(8 * MAX(1,psSHP->nRecords) );
    fread( pabyBuf, 8, psSHP->nRecords, psSHP->fpSHX );

    for( i = 0; i < psSHP->nRecords; i++ )
    {
	int32		nOffset, nLength;

	memcpy( &nOffset, pabyBuf + i * 8, 4 );
	if( !bBigEndian ) SwapWord( 4, &nOffset );

	memcpy( &nLength, pabyBuf + i * 8 + 4, 4 );
	if( !bBigEndian ) SwapWord( 4, &nLength );

	psSHP->panRecOffset[i] = nOffset*2;
	psSHP->panRecSize[i] = nLength*2;
    }
    free( pabyBuf );

    return( psSHP );
}

/************************************************************************/
/*                              SHPClose()                              */
/*								       	*/
/*	Close the .shp and .shx files.					*/
/************************************************************************/

void SHPAPI_CALL
SHPClose(SHPHandle psSHP )

{
/* -------------------------------------------------------------------- */
/*	Update the header if we have modified anything.			*/
/* -------------------------------------------------------------------- */
    if( psSHP->bUpdated )
    {
	SHPWriteHeader( psSHP );
    }

/* -------------------------------------------------------------------- */
/*      Free all resources, and close files.                            */
/* -------------------------------------------------------------------- */
    free( psSHP->panRecOffset );
    free( psSHP->panRecSize );

    fclose( psSHP->fpSHX );
    fclose( psSHP->fpSHP );

    if( psSHP->pabyRec != NULL )
    {
        free( psSHP->pabyRec );
    }
    
    free( psSHP );
}

/************************************************************************/
/*                             SHPGetInfo()                             */
/*                                                                      */
/*      Fetch general information about the shape file.                 */
/************************************************************************/

void SHPAPI_CALL
SHPGetInfo(SHPHandle psSHP, int * pnEntities, int * pnShapeType,
           double * padfMinBound, double * padfMaxBound )

{
    int		i;
    
    if( pnEntities != NULL )
        *pnEntities = psSHP->nRecords;

    if( pnShapeType != NULL )
        *pnShapeType = psSHP->nShapeType;

    for( i = 0; i < 4; i++ )
    {
        if( padfMinBound != NULL )
            padfMinBound[i] = psSHP->adBoundsMin[i];
        if( padfMaxBound != NULL )
            padfMaxBound[i] = psSHP->adBoundsMax[i];
    }
}

/************************************************************************/
/*                             SHPCreate()                              */
/*                                                                      */
/*      Create a new shape file and return a handle to the open         */
/*      shape file with read/write access.                              */
/************************************************************************/

SHPHandle SHPAPI_CALL
SHPCreate( const char * pszLayer, int nShapeType )

{
    char	*pszBasename, *pszFullname;
    int		i;
    FILE	*fpSHP, *fpSHX;
    uchar     	abyHeader[100];
    int32	i32;
    double	dValue;
    
/* -------------------------------------------------------------------- */
/*      Establish the byte order on this system.                        */
/* -------------------------------------------------------------------- */
    i = 1;
    if( *((uchar *) &i) == 1 )
        bBigEndian = FALSE;
    else
        bBigEndian = TRUE;

/* -------------------------------------------------------------------- */
/*	Compute the base (layer) name.  If there is any extension	*/
/*	on the passed in filename we will strip it off.			*/
/* -------------------------------------------------------------------- */
    pszBasename = (char *) malloc(strlen(pszLayer)+5);
    strcpy( pszBasename, pszLayer );
    for( i = strlen(pszBasename)-1; 
	 i > 0 && pszBasename[i] != '.' && pszBasename[i] != '/'
	       && pszBasename[i] != '\\';
	 i-- ) {}

    if( pszBasename[i] == '.' )
        pszBasename[i] = '\0';

/* -------------------------------------------------------------------- */
/*      Open the two files so we can write their headers.               */
/* -------------------------------------------------------------------- */
    pszFullname = (char *) malloc(strlen(pszBasename) + 5);
    sprintf( pszFullname, "%s.shp", pszBasename );
    fpSHP = fopen(pszFullname, "wb" );
    if( fpSHP == NULL )
        return( NULL );

    sprintf( pszFullname, "%s.shx", pszBasename );
    fpSHX = fopen(pszFullname, "wb" );
    if( fpSHX == NULL )
        return( NULL );

    free( pszFullname );
    free( pszBasename );

/* -------------------------------------------------------------------- */
/*      Prepare header block for .shp file.                             */
/* -------------------------------------------------------------------- */
    for( i = 0; i < 100; i++ )
      abyHeader[i] = 0;

    abyHeader[2] = 0x27;				/* magic cookie */
    abyHeader[3] = 0x0a;

    i32 = 50;						/* file size */
    ByteCopy( &i32, abyHeader+24, 4 );
    if( !bBigEndian ) SwapWord( 4, abyHeader+24 );
    
    i32 = 1000;						/* version */
    ByteCopy( &i32, abyHeader+28, 4 );
    if( bBigEndian ) SwapWord( 4, abyHeader+28 );
    
    i32 = nShapeType;					/* shape type */
    ByteCopy( &i32, abyHeader+32, 4 );
    if( bBigEndian ) SwapWord( 4, abyHeader+32 );

    dValue = 0.0;					/* set bounds */
    ByteCopy( &dValue, abyHeader+36, 8 );
    ByteCopy( &dValue, abyHeader+44, 8 );
    ByteCopy( &dValue, abyHeader+52, 8 );
    ByteCopy( &dValue, abyHeader+60, 8 );

/* -------------------------------------------------------------------- */
/*      Write .shp file header.                                         */
/* -------------------------------------------------------------------- */
    fwrite( abyHeader, 100, 1, fpSHP );

/* -------------------------------------------------------------------- */
/*      Prepare, and write .shx file header.                            */
/* -------------------------------------------------------------------- */
    i32 = 50;						/* file size */
    ByteCopy( &i32, abyHeader+24, 4 );
    if( !bBigEndian ) SwapWord( 4, abyHeader+24 );
    
    fwrite( abyHeader, 100, 1, fpSHX );

/* -------------------------------------------------------------------- */
/*      Close the files, and then open them as regular existing files.  */
/* -------------------------------------------------------------------- */
    fclose( fpSHP );
    fclose( fpSHX );

    return( SHPOpen( pszLayer, "r+b" ) );
}

/************************************************************************/
/*                           _SHPSetBounds()                            */
/*                                                                      */
/*      Compute a bounds rectangle for a shape, and set it into the     */
/*      indicated location in the record.                               */
/************************************************************************/

static void	_SHPSetBounds( uchar * pabyRec, SHPObject * psShape )

{
    ByteCopy( &(psShape->dfXMin), pabyRec +  0, 8 );
    ByteCopy( &(psShape->dfYMin), pabyRec +  8, 8 );
    ByteCopy( &(psShape->dfXMax), pabyRec + 16, 8 );
    ByteCopy( &(psShape->dfYMax), pabyRec + 24, 8 );

    if( bBigEndian )
    {
        SwapWord( 8, pabyRec + 0 );
        SwapWord( 8, pabyRec + 8 );
        SwapWord( 8, pabyRec + 16 );
        SwapWord( 8, pabyRec + 24 );
    }
}

/************************************************************************/
/*                         SHPComputeExtents()                          */
/*                                                                      */
/*      Recompute the extents of a shape.  Automatically done by        */
/*      SHPCreateObject().                                              */
/************************************************************************/

void SHPAPI_CALL
SHPComputeExtents( SHPObject * psObject )

{
    int		i;
    
/* -------------------------------------------------------------------- */
/*      Build extents for this object.                                  */
/* -------------------------------------------------------------------- */
    if( psObject->nVertices > 0 )
    {
        psObject->dfXMin = psObject->dfXMax = psObject->padfX[0];
        psObject->dfYMin = psObject->dfYMax = psObject->padfY[0];
        psObject->dfZMin = psObject->dfZMax = psObject->padfZ[0];
        psObject->dfMMin = psObject->dfMMax = psObject->padfM[0];
    }
    
    for( i = 0; i < psObject->nVertices; i++ )
    {
        psObject->dfXMin = MIN(psObject->dfXMin, psObject->padfX[i]);
        psObject->dfYMin = MIN(psObject->dfYMin, psObject->padfY[i]);
        psObject->dfZMin = MIN(psObject->dfZMin, psObject->padfZ[i]);
        psObject->dfMMin = MIN(psObject->dfMMin, psObject->padfM[i]);

        psObject->dfXMax = MAX(psObject->dfXMax, psObject->padfX[i]);
        psObject->dfYMax = MAX(psObject->dfYMax, psObject->padfY[i]);
        psObject->dfZMax = MAX(psObject->dfZMax, psObject->padfZ[i]);
        psObject->dfMMax = MAX(psObject->dfMMax, psObject->padfM[i]);
    }
}

/************************************************************************/
/*                          SHPCreateObject()                           */
/*                                                                      */
/*      Create a shape object.  It should be freed with                 */
/*      SHPDestroyObject().                                             */
/************************************************************************/

SHPObject SHPAPI_CALL1(*)
SHPCreateObject( int nSHPType, int nShapeId, int nParts,
                 int * panPartStart, int * panPartType,
                 int nVertices, double * padfX, double * padfY,
                 double * padfZ, double * padfM )

{
    SHPObject	*psObject;
    int		i, bHasM, bHasZ;

    psObject = (SHPObject *) calloc(1,sizeof(SHPObject));
    psObject->nSHPType = nSHPType;
    psObject->nShapeId = nShapeId;

/* -------------------------------------------------------------------- */
/*	Establish whether this shape type has M, and Z values.		*/
/* -------------------------------------------------------------------- */
    if( nSHPType == SHPT_ARCM
        || nSHPType == SHPT_POINTM
        || nSHPType == SHPT_POLYGONM
        || nSHPType == SHPT_MULTIPOINTM )
    {
        bHasM = TRUE;
        bHasZ = FALSE;
    }
    else if( nSHPType == SHPT_ARCZ
             || nSHPType == SHPT_POINTZ
             || nSHPType == SHPT_POLYGONZ
             || nSHPType == SHPT_MULTIPOINTZ
             || nSHPType == SHPT_MULTIPATCH )
    {
        bHasM = TRUE;
        bHasZ = TRUE;
    }
    else
    {
        bHasM = FALSE;
        bHasZ = FALSE;
    }

/* -------------------------------------------------------------------- */
/*      Capture parts.  Note that part type is optional, and            */
/*      defaults to ring.                                               */
/* -------------------------------------------------------------------- */
    if( nSHPType == SHPT_ARC || nSHPType == SHPT_POLYGON
        || nSHPType == SHPT_ARCM || nSHPType == SHPT_POLYGONM
        || nSHPType == SHPT_ARCZ || nSHPType == SHPT_POLYGONZ
        || nSHPType == SHPT_MULTIPATCH )
    {
        psObject->nParts = MAX(1,nParts);

        psObject->panPartStart = (int *)
            malloc(sizeof(int) * psObject->nParts);
        psObject->panPartType = (int *)
            malloc(sizeof(int) * psObject->nParts);

        psObject->panPartStart[0] = 0;
        psObject->panPartType[0] = SHPP_RING;
        
        for( i = 0; i < nParts; i++ )
        {
            psObject->panPartStart[i] = panPartStart[i];
            if( panPartType != NULL )
                psObject->panPartType[i] = panPartType[i];
            else
                psObject->panPartType[i] = SHPP_RING;
        }
    }

/* -------------------------------------------------------------------- */
/*      Capture vertices.  Note that Z and M are optional, but X and    */
/*      Y are not.                                                      */
/* -------------------------------------------------------------------- */
    if( nVertices > 0 )
    {
        psObject->padfX = (double *) calloc(sizeof(double),nVertices);
        psObject->padfY = (double *) calloc(sizeof(double),nVertices);
        psObject->padfZ = (double *) calloc(sizeof(double),nVertices);
        psObject->padfM = (double *) calloc(sizeof(double),nVertices);

        assert( padfX != NULL );
        assert( padfY != NULL );
    
        for( i = 0; i < nVertices; i++ )
        {
            psObject->padfX[i] = padfX[i];
            psObject->padfY[i] = padfY[i];
            if( padfZ != NULL && bHasZ )
                psObject->padfZ[i] = padfZ[i];
            if( padfM != NULL && bHasM )
                psObject->padfM[i] = padfM[i];
        }
    }

/* -------------------------------------------------------------------- */
/*      Compute the extents.                                            */
/* -------------------------------------------------------------------- */
    psObject->nVertices = nVertices;
    SHPComputeExtents( psObject );

    return( psObject );
}

/************************************************************************/
/*                       SHPCreateSimpleObject()                        */
/*                                                                      */
/*      Create a simple (common) shape object.  Destroy with            */
/*      SHPDestroyObject().                                             */
/************************************************************************/

SHPObject SHPAPI_CALL1(*)
SHPCreateSimpleObject( int nSHPType, int nVertices,
                       double * padfX, double * padfY,
                       double * padfZ )

{
    return( SHPCreateObject( nSHPType, -1, 0, NULL, NULL,
                             nVertices, padfX, padfY, padfZ, NULL ) );
}
                                  
/************************************************************************/
/*                           SHPWriteObject()                           */
/*                                                                      */
/*      Write out the vertices of a new structure.  Note that it is     */
/*      only possible to write vertices at the end of the file.         */
/************************************************************************/

int SHPAPI_CALL
SHPWriteObject(SHPHandle psSHP, int nShapeId, SHPObject * psObject )
		      
{
    int	       	nRecordOffset, i, nRecordSize;
    uchar	*pabyRec;
    int32	i32;

/* Just to get rid of compile warning */
nRecordOffset = nRecordSize = 0;

    psSHP->bUpdated = TRUE;

/* -------------------------------------------------------------------- */
/*      Ensure that shape object matches the type of the file it is     */
/*      being written to.                                               */
/* -------------------------------------------------------------------- */
    assert( psObject->nSHPType == psSHP->nShapeType 
            || psObject->nSHPType == SHPT_NULL );

/* -------------------------------------------------------------------- */
/*      Ensure that -1 is used for appends.  Either blow an             */
/*      assertion, or if they are disabled, set the shapeid to -1       */
/*      for appends.                                                    */
/* -------------------------------------------------------------------- */
    assert( nShapeId == -1 
            || (nShapeId >= 0 && nShapeId < psSHP->nRecords) );

    if( nShapeId != -1 && nShapeId >= psSHP->nRecords )
        nShapeId = -1;

/* -------------------------------------------------------------------- */
/*      Add the new entity to the in memory index.                      */
/* -------------------------------------------------------------------- */
    if( nShapeId == -1 && psSHP->nRecords+1 > psSHP->nMaxRecords )
    {
	psSHP->nMaxRecords =(int) ( psSHP->nMaxRecords * 1.3 + 100);

	psSHP->panRecOffset = (int *) 
            SfRealloc(psSHP->panRecOffset,sizeof(int) * psSHP->nMaxRecords );
	psSHP->panRecSize = (int *) 
            SfRealloc(psSHP->panRecSize,sizeof(int) * psSHP->nMaxRecords );
    }

/* -------------------------------------------------------------------- */
/*      Initialize record.                                              */
/* -------------------------------------------------------------------- */
    pabyRec = (uchar *) malloc(psObject->nVertices * 4 * sizeof(double) 
			       + psObject->nParts * 8 + 128);
    
/* -------------------------------------------------------------------- */
/*  Extract vertices for a Polygon or Arc.				*/
/* -------------------------------------------------------------------- */
    if( psObject->nSHPType == SHPT_POLYGON
        || psObject->nSHPType == SHPT_POLYGONZ
        || psObject->nSHPType == SHPT_POLYGONM
        || psObject->nSHPType == SHPT_ARC 
        || psObject->nSHPType == SHPT_ARCZ
        || psObject->nSHPType == SHPT_ARCM
        || psObject->nSHPType == SHPT_MULTIPATCH )
    {
	int32		nPoints, nParts;
	int    		i;

	nPoints = psObject->nVertices;
	nParts = psObject->nParts;

	_SHPSetBounds( pabyRec + 12, psObject );

	if( bBigEndian ) SwapWord( 4, &nPoints );
	if( bBigEndian ) SwapWord( 4, &nParts );

	ByteCopy( &nPoints, pabyRec + 40 + 8, 4 );
	ByteCopy( &nParts, pabyRec + 36 + 8, 4 );

        nRecordSize = 52;

        /*
         * Write part start positions.
         */
	ByteCopy( psObject->panPartStart, pabyRec + 44 + 8,
                  4 * psObject->nParts );
	for( i = 0; i < psObject->nParts; i++ )
	{
	    if( bBigEndian ) SwapWord( 4, pabyRec + 44 + 8 + 4*i );
            nRecordSize += 4;
	}

        /*
         * Write multipatch part types if needed.
         */
        if( psObject->nSHPType == SHPT_MULTIPATCH )
        {
            memcpy( pabyRec + nRecordSize, psObject->panPartType,
                    4*psObject->nParts );
            for( i = 0; i < psObject->nParts; i++ )
            {
                if( bBigEndian ) SwapWord( 4, pabyRec + nRecordSize );
                nRecordSize += 4;
            }
        }

        /*
         * Write the (x,y) vertex values.
         */
	for( i = 0; i < psObject->nVertices; i++ )
	{
	    ByteCopy( psObject->padfX + i, pabyRec + nRecordSize, 8 );
	    ByteCopy( psObject->padfY + i, pabyRec + nRecordSize + 8, 8 );

	    if( bBigEndian )
                SwapWord( 8, pabyRec + nRecordSize );
            
	    if( bBigEndian )
                SwapWord( 8, pabyRec + nRecordSize + 8 );

            nRecordSize += 2 * 8;
	}

        /*
         * Write the Z coordinates (if any).
         */
        if( psObject->nSHPType == SHPT_POLYGONZ
            || psObject->nSHPType == SHPT_ARCZ
            || psObject->nSHPType == SHPT_MULTIPATCH )
        {
            ByteCopy( &(psObject->dfZMin), pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;
            
            ByteCopy( &(psObject->dfZMax), pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;

            for( i = 0; i < psObject->nVertices; i++ )
            {
                ByteCopy( psObject->padfZ + i, pabyRec + nRecordSize, 8 );
                if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
                nRecordSize += 8;
            }
        }

        /*
         * Write the M values, if any.
         */
        if( psObject->nSHPType == SHPT_POLYGONM
            || psObject->nSHPType == SHPT_ARCM
#ifndef DISABLE_MULTIPATCH_MEASURE            
            || psObject->nSHPType == SHPT_MULTIPATCH
#endif            
            || psObject->nSHPType == SHPT_POLYGONZ
            || psObject->nSHPType == SHPT_ARCZ )
        {
            ByteCopy( &(psObject->dfMMin), pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;
            
            ByteCopy( &(psObject->dfMMax), pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;

            for( i = 0; i < psObject->nVertices; i++ )
            {
                ByteCopy( psObject->padfM + i, pabyRec + nRecordSize, 8 );
                if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
                nRecordSize += 8;
            }
        }
    }

/* -------------------------------------------------------------------- */
/*  Extract vertices for a MultiPoint.					*/
/* -------------------------------------------------------------------- */
    else if( psObject->nSHPType == SHPT_MULTIPOINT
             || psObject->nSHPType == SHPT_MULTIPOINTZ
             || psObject->nSHPType == SHPT_MULTIPOINTM )
    {
	int32		nPoints;
	int    		i;

	nPoints = psObject->nVertices;

        _SHPSetBounds( pabyRec + 12, psObject );

	if( bBigEndian ) SwapWord( 4, &nPoints );
	ByteCopy( &nPoints, pabyRec + 44, 4 );
	
	for( i = 0; i < psObject->nVertices; i++ )
	{
	    ByteCopy( psObject->padfX + i, pabyRec + 48 + i*16, 8 );
	    ByteCopy( psObject->padfY + i, pabyRec + 48 + i*16 + 8, 8 );

	    if( bBigEndian ) SwapWord( 8, pabyRec + 48 + i*16 );
	    if( bBigEndian ) SwapWord( 8, pabyRec + 48 + i*16 + 8 );
	}

	nRecordSize = 48 + 16 * psObject->nVertices;

        if( psObject->nSHPType == SHPT_MULTIPOINTZ )
        {
            ByteCopy( &(psObject->dfZMin), pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;

            ByteCopy( &(psObject->dfZMax), pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;
            
            for( i = 0; i < psObject->nVertices; i++ )
            {
                ByteCopy( psObject->padfZ + i, pabyRec + nRecordSize, 8 );
                if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
                nRecordSize += 8;
            }
        }

        if( psObject->nSHPType == SHPT_MULTIPOINTZ
            || psObject->nSHPType == SHPT_MULTIPOINTM )
        {
            ByteCopy( &(psObject->dfMMin), pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;

            ByteCopy( &(psObject->dfMMax), pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;
            
            for( i = 0; i < psObject->nVertices; i++ )
            {
                ByteCopy( psObject->padfM + i, pabyRec + nRecordSize, 8 );
                if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
                nRecordSize += 8;
            }
        }
    }

/* -------------------------------------------------------------------- */
/*      Write point.							*/
/* -------------------------------------------------------------------- */
    else if( psObject->nSHPType == SHPT_POINT
             || psObject->nSHPType == SHPT_POINTZ
             || psObject->nSHPType == SHPT_POINTM )
    {

	ByteCopy( psObject->padfX, pabyRec + 12, 8 );
	ByteCopy( psObject->padfY, pabyRec + 20, 8 );

	if( bBigEndian ) SwapWord( 8, pabyRec + 12 );
	if( bBigEndian ) SwapWord( 8, pabyRec + 20 );


        nRecordSize = 28;
        
        if( psObject->nSHPType == SHPT_POINTZ )
        {
            ByteCopy( psObject->padfZ, pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;
        }

        
        if( psObject->nSHPType == SHPT_POINTZ
            || psObject->nSHPType == SHPT_POINTM )
        {
            ByteCopy( psObject->padfM, pabyRec + nRecordSize, 8 );
            if( bBigEndian ) SwapWord( 8, pabyRec + nRecordSize );
            nRecordSize += 8;
        }
    }

/* -------------------------------------------------------------------- */
/*      Not much to do for null geometries.                             */
/* -------------------------------------------------------------------- */
    else if( psObject->nSHPType == SHPT_NULL )
    {
        nRecordSize = 12;
    }

    else
    {
        /* unknown type */
        assert( FALSE );
    }

/* -------------------------------------------------------------------- */
/*      Establish where we are going to put this record. If we are      */
/*      rewriting and existing record, and it will fit, then put it     */
/*      back where the original came from.  Otherwise write at the end. */
/* -------------------------------------------------------------------- */
    if( nShapeId == -1 || psSHP->panRecSize[nShapeId] < nRecordSize-8 )
    {
        if( nShapeId == -1 )
            nShapeId = psSHP->nRecords++;

        psSHP->panRecOffset[nShapeId] = nRecordOffset = psSHP->nFileSize;
        psSHP->panRecSize[nShapeId] = nRecordSize-8;
        psSHP->nFileSize += nRecordSize;
    }
    else
    {
        nRecordOffset = psSHP->panRecOffset[nShapeId];
    }

/* -------------------------------------------------------------------- */
/*      Set the shape type, record number, and record size.             */
/* -------------------------------------------------------------------- */
    i32 = nShapeId+1;					/* record # */
    if( !bBigEndian ) SwapWord( 4, &i32 );
    ByteCopy( &i32, pabyRec, 4 );

    i32 = (nRecordSize-8)/2;				/* record size */
    if( !bBigEndian ) SwapWord( 4, &i32 );
    ByteCopy( &i32, pabyRec + 4, 4 );

    i32 = psObject->nSHPType;				/* shape type */
    if( bBigEndian ) SwapWord( 4, &i32 );
    ByteCopy( &i32, pabyRec + 8, 4 );

/* -------------------------------------------------------------------- */
/*      Write out record.                                               */
/* -------------------------------------------------------------------- */
    if( fseek( psSHP->fpSHP, nRecordOffset, 0 ) != 0
        || fwrite( pabyRec, nRecordSize, 1, psSHP->fpSHP ) < 1 )
    {
        printf( "Error in fseek() or fwrite().\n" );
        free( pabyRec );
        return -1;
    }
    
    free( pabyRec );

/* -------------------------------------------------------------------- */
/*	Expand file wide bounds based on this shape.			*/
/* -------------------------------------------------------------------- */
    if( psSHP->adBoundsMin[0] == 0.0
        && psSHP->adBoundsMax[0] == 0.0
        && psSHP->adBoundsMin[1] == 0.0
        && psSHP->adBoundsMax[1] == 0.0 
        && psObject->nSHPType != SHPT_NULL )
    {
        psSHP->adBoundsMin[0] = psSHP->adBoundsMax[0] = psObject->padfX[0];
        psSHP->adBoundsMin[1] = psSHP->adBoundsMax[1] = psObject->padfY[0];
        psSHP->adBoundsMin[2] = psSHP->adBoundsMax[2] = psObject->padfZ[0];
        psSHP->adBoundsMin[3] = psSHP->adBoundsMax[3] = psObject->padfM[0];
    }

    for( i = 0; i < psObject->nVertices; i++ )
    {
	psSHP->adBoundsMin[0] = MIN(psSHP->adBoundsMin[0],psObject->padfX[i]);
	psSHP->adBoundsMin[1] = MIN(psSHP->adBoundsMin[1],psObject->padfY[i]);
	psSHP->adBoundsMin[2] = MIN(psSHP->adBoundsMin[2],psObject->padfZ[i]);
	psSHP->adBoundsMin[3] = MIN(psSHP->adBoundsMin[3],psObject->padfM[i]);
	psSHP->adBoundsMax[0] = MAX(psSHP->adBoundsMax[0],psObject->padfX[i]);
	psSHP->adBoundsMax[1] = MAX(psSHP->adBoundsMax[1],psObject->padfY[i]);
	psSHP->adBoundsMax[2] = MAX(psSHP->adBoundsMax[2],psObject->padfZ[i]);
	psSHP->adBoundsMax[3] = MAX(psSHP->adBoundsMax[3],psObject->padfM[i]);
    }

    return( nShapeId  );
}

/************************************************************************/
/*                          SHPReadObject()                             */
/*                                                                      */
/*      Read the vertices, parts, and other non-attribute information	*/
/*	for one shape.							*/
/************************************************************************/

SHPObject SHPAPI_CALL1(*)
SHPReadObject( SHPHandle psSHP, int hEntity )

{
    SHPObject		*psShape;

/* -------------------------------------------------------------------- */
/*      Validate the record/entity number.                              */
/* -------------------------------------------------------------------- */
    if( hEntity < 0 || hEntity >= psSHP->nRecords )
        return( NULL );

/* -------------------------------------------------------------------- */
/*      Ensure our record buffer is large enough.                       */
/* -------------------------------------------------------------------- */
    if( psSHP->panRecSize[hEntity]+8 > psSHP->nBufSize )
    {
	psSHP->nBufSize = psSHP->panRecSize[hEntity]+8;
	psSHP->pabyRec = (uchar *) SfRealloc(psSHP->pabyRec,psSHP->nBufSize);
    }

/* -------------------------------------------------------------------- */
/*      Read the record.                                                */
/* -------------------------------------------------------------------- */
    fseek( psSHP->fpSHP, psSHP->panRecOffset[hEntity], 0 );
    fread( psSHP->pabyRec, psSHP->panRecSize[hEntity]+8, 1, psSHP->fpSHP );

/* -------------------------------------------------------------------- */
/*	Allocate and minimally initialize the object.			*/
/* -------------------------------------------------------------------- */
    psShape = (SHPObject *) calloc(1,sizeof(SHPObject));
    psShape->nShapeId = hEntity;

    memcpy( &psShape->nSHPType, psSHP->pabyRec + 8, 4 );
    if( bBigEndian ) SwapWord( 4, &(psShape->nSHPType) );

/* ==================================================================== */
/*  Extract vertices for a Polygon or Arc.				*/
/* ==================================================================== */
    if( psShape->nSHPType == SHPT_POLYGON || psShape->nSHPType == SHPT_ARC
        || psShape->nSHPType == SHPT_POLYGONZ
        || psShape->nSHPType == SHPT_POLYGONM
        || psShape->nSHPType == SHPT_ARCZ
        || psShape->nSHPType == SHPT_ARCM
        || psShape->nSHPType == SHPT_MULTIPATCH )
    {
	int32		nPoints, nParts;
	int    		i, nOffset;

/* -------------------------------------------------------------------- */
/*	Get the X/Y bounds.						*/
/* -------------------------------------------------------------------- */
        memcpy( &(psShape->dfXMin), psSHP->pabyRec + 8 +  4, 8 );
        memcpy( &(psShape->dfYMin), psSHP->pabyRec + 8 + 12, 8 );
        memcpy( &(psShape->dfXMax), psSHP->pabyRec + 8 + 20, 8 );
        memcpy( &(psShape->dfYMax), psSHP->pabyRec + 8 + 28, 8 );

	if( bBigEndian ) SwapWord( 8, &(psShape->dfXMin) );
	if( bBigEndian ) SwapWord( 8, &(psShape->dfYMin) );
	if( bBigEndian ) SwapWord( 8, &(psShape->dfXMax) );
	if( bBigEndian ) SwapWord( 8, &(psShape->dfYMax) );

/* -------------------------------------------------------------------- */
/*      Extract part/point count, and build vertex and part arrays      */
/*      to proper size.                                                 */
/* -------------------------------------------------------------------- */
	memcpy( &nPoints, psSHP->pabyRec + 40 + 8, 4 );
	memcpy( &nParts, psSHP->pabyRec + 36 + 8, 4 );

	if( bBigEndian ) SwapWord( 4, &nPoints );
	if( bBigEndian ) SwapWord( 4, &nParts );

	psShape->nVertices = nPoints;
        psShape->padfX = (double *) calloc(nPoints,sizeof(double));
        psShape->padfY = (double *) calloc(nPoints,sizeof(double));
        psShape->padfZ = (double *) calloc(nPoints,sizeof(double));
        psShape->padfM = (double *) calloc(nPoints,sizeof(double));

	psShape->nParts = nParts;
        psShape->panPartStart = (int *) calloc(nParts,sizeof(int));
        psShape->panPartType = (int *) calloc(nParts,sizeof(int));

        for( i = 0; i < nParts; i++ )
            psShape->panPartType[i] = SHPP_RING;

/* -------------------------------------------------------------------- */
/*      Copy out the part array from the record.                        */
/* -------------------------------------------------------------------- */
	memcpy( psShape->panPartStart, psSHP->pabyRec + 44 + 8, 4 * nParts );
	for( i = 0; i < nParts; i++ )
	{
	    if( bBigEndian ) SwapWord( 4, psShape->panPartStart+i );
	}

	nOffset = 44 + 8 + 4*nParts;

/* -------------------------------------------------------------------- */
/*      If this is a multipatch, we will also have parts types.         */
/* -------------------------------------------------------------------- */
        if( psShape->nSHPType == SHPT_MULTIPATCH )
        {
            memcpy( psShape->panPartType, psSHP->pabyRec + nOffset, 4*nParts );
            for( i = 0; i < nParts; i++ )
            {
                if( bBigEndian ) SwapWord( 4, psShape->panPartType+i );
            }

            nOffset += 4*nParts;
        }
        
/* -------------------------------------------------------------------- */
/*      Copy out the vertices from the record.                          */
/* -------------------------------------------------------------------- */
	for( i = 0; i < nPoints; i++ )
	{
	    memcpy(psShape->padfX + i,
		   psSHP->pabyRec + nOffset + i * 16,
		   8 );

	    memcpy(psShape->padfY + i,
		   psSHP->pabyRec + nOffset + i * 16 + 8,
		   8 );

	    if( bBigEndian ) SwapWord( 8, psShape->padfX + i );
	    if( bBigEndian ) SwapWord( 8, psShape->padfY + i );
	}

        nOffset += 16*nPoints;
        
/* -------------------------------------------------------------------- */
/*      If we have a Z coordinate, collect that now.                    */
/* -------------------------------------------------------------------- */
        if( psShape->nSHPType == SHPT_POLYGONZ
            || psShape->nSHPType == SHPT_ARCZ
            || psShape->nSHPType == SHPT_MULTIPATCH )
        {
            memcpy( &(psShape->dfZMin), psSHP->pabyRec + nOffset, 8 );
            memcpy( &(psShape->dfZMax), psSHP->pabyRec + nOffset + 8, 8 );
            
            if( bBigEndian ) SwapWord( 8, &(psShape->dfZMin) );
            if( bBigEndian ) SwapWord( 8, &(psShape->dfZMax) );
            
            for( i = 0; i < nPoints; i++ )
            {
                memcpy( psShape->padfZ + i,
                        psSHP->pabyRec + nOffset + 16 + i*8, 8 );
                if( bBigEndian ) SwapWord( 8, psShape->padfZ + i );
            }

            nOffset += 16 + 8*nPoints;
        }

/* -------------------------------------------------------------------- */
/*      If we have a M measure value, then read it now.  We assume      */
/*      that the measure can be present for any shape if the size is    */
/*      big enough, but really it will only occur for the Z shapes      */
/*      (options), and the M shapes.                                    */
/* -------------------------------------------------------------------- */
        if( psSHP->panRecSize[hEntity]+8 >= nOffset + 16 + 8*nPoints )
        {
            memcpy( &(psShape->dfMMin), psSHP->pabyRec + nOffset, 8 );
            memcpy( &(psShape->dfMMax), psSHP->pabyRec + nOffset + 8, 8 );
            
            if( bBigEndian ) SwapWord( 8, &(psShape->dfMMin) );
            if( bBigEndian ) SwapWord( 8, &(psShape->dfMMax) );
            
            for( i = 0; i < nPoints; i++ )
            {
                memcpy( psShape->padfM + i,
                        psSHP->pabyRec + nOffset + 16 + i*8, 8 );
                if( bBigEndian ) SwapWord( 8, psShape->padfM + i );
            }
        }
        
    }

/* ==================================================================== */
/*  Extract vertices for a MultiPoint.					*/
/* ==================================================================== */
    else if( psShape->nSHPType == SHPT_MULTIPOINT
             || psShape->nSHPType == SHPT_MULTIPOINTM
             || psShape->nSHPType == SHPT_MULTIPOINTZ )
    {
	int32		nPoints;
	int    		i, nOffset;

	memcpy( &nPoints, psSHP->pabyRec + 44, 4 );
	if( bBigEndian ) SwapWord( 4, &nPoints );

	psShape->nVertices = nPoints;
        psShape->padfX = (double *) calloc(nPoints,sizeof(double));
        psShape->padfY = (double *) calloc(nPoints,sizeof(double));
        psShape->padfZ = (double *) calloc(nPoints,sizeof(double));
        psShape->padfM = (double *) calloc(nPoints,sizeof(double));

	for( i = 0; i < nPoints; i++ )
	{
	    memcpy(psShape->padfX+i, psSHP->pabyRec + 48 + 16 * i, 8 );
	    memcpy(psShape->padfY+i, psSHP->pabyRec + 48 + 16 * i + 8, 8 );

	    if( bBigEndian ) SwapWord( 8, psShape->padfX + i );
	    if( bBigEndian ) SwapWord( 8, psShape->padfY + i );
	}

        nOffset = 48 + 16*nPoints;
        
/* -------------------------------------------------------------------- */
/*	Get the X/Y bounds.						*/
/* -------------------------------------------------------------------- */
        memcpy( &(psShape->dfXMin), psSHP->pabyRec + 8 +  4, 8 );
        memcpy( &(psShape->dfYMin), psSHP->pabyRec + 8 + 12, 8 );
        memcpy( &(psShape->dfXMax), psSHP->pabyRec + 8 + 20, 8 );
        memcpy( &(psShape->dfYMax), psSHP->pabyRec + 8 + 28, 8 );

	if( bBigEndian ) SwapWord( 8, &(psShape->dfXMin) );
	if( bBigEndian ) SwapWord( 8, &(psShape->dfYMin) );
	if( bBigEndian ) SwapWord( 8, &(psShape->dfXMax) );
	if( bBigEndian ) SwapWord( 8, &(psShape->dfYMax) );

/* -------------------------------------------------------------------- */
/*      If we have a Z coordinate, collect that now.                    */
/* -------------------------------------------------------------------- */
        if( psShape->nSHPType == SHPT_MULTIPOINTZ )
        {
            memcpy( &(psShape->dfZMin), psSHP->pabyRec + nOffset, 8 );
            memcpy( &(psShape->dfZMax), psSHP->pabyRec + nOffset + 8, 8 );
            
            if( bBigEndian ) SwapWord( 8, &(psShape->dfZMin) );
            if( bBigEndian ) SwapWord( 8, &(psShape->dfZMax) );
            
            for( i = 0; i < nPoints; i++ )
            {
                memcpy( psShape->padfZ + i,
                        psSHP->pabyRec + nOffset + 16 + i*8, 8 );
                if( bBigEndian ) SwapWord( 8, psShape->padfZ + i );
            }

            nOffset += 16 + 8*nPoints;
        }

/* -------------------------------------------------------------------- */
/*      If we have a M measure value, then read it now.  We assume      */
/*      that the measure can be present for any shape if the size is    */
/*      big enough, but really it will only occur for the Z shapes      */
/*      (options), and the M shapes.                                    */
/* -------------------------------------------------------------------- */
        if( psSHP->panRecSize[hEntity]+8 >= nOffset + 16 + 8*nPoints )
        {
            memcpy( &(psShape->dfMMin), psSHP->pabyRec + nOffset, 8 );
            memcpy( &(psShape->dfMMax), psSHP->pabyRec + nOffset + 8, 8 );
            
            if( bBigEndian ) SwapWord( 8, &(psShape->dfMMin) );
            if( bBigEndian ) SwapWord( 8, &(psShape->dfMMax) );
            
            for( i = 0; i < nPoints; i++ )
            {
                memcpy( psShape->padfM + i,
                        psSHP->pabyRec + nOffset + 16 + i*8, 8 );
                if( bBigEndian ) SwapWord( 8, psShape->padfM + i );
            }
        }
    }

/* ==================================================================== */
/*      Extract vertices for a point.                                   */
/* ==================================================================== */
    else if( psShape->nSHPType == SHPT_POINT
             || psShape->nSHPType == SHPT_POINTM
             || psShape->nSHPType == SHPT_POINTZ )
    {
        int	nOffset;
        
	psShape->nVertices = 1;
        psShape->padfX = (double *) calloc(1,sizeof(double));
        psShape->padfY = (double *) calloc(1,sizeof(double));
        psShape->padfZ = (double *) calloc(1,sizeof(double));
        psShape->padfM = (double *) calloc(1,sizeof(double));

	memcpy( psShape->padfX, psSHP->pabyRec + 12, 8 );
	memcpy( psShape->padfY, psSHP->pabyRec + 20, 8 );

	if( bBigEndian ) SwapWord( 8, psShape->padfX );
	if( bBigEndian ) SwapWord( 8, psShape->padfY );

        nOffset = 20 + 8;
        
/* -------------------------------------------------------------------- */
/*      If we have a Z coordinate, collect that now.                    */
/* -------------------------------------------------------------------- */
        if( psShape->nSHPType == SHPT_POINTZ )
        {
            memcpy( psShape->padfZ, psSHP->pabyRec + nOffset, 8 );
        
            if( bBigEndian ) SwapWord( 8, psShape->padfZ );
            
            nOffset += 8;
        }

/* -------------------------------------------------------------------- */
/*      If we have a M measure value, then read it now.  We assume      */
/*      that the measure can be present for any shape if the size is    */
/*      big enough, but really it will only occur for the Z shapes      */
/*      (options), and the M shapes.                                    */
/* -------------------------------------------------------------------- */
        if( psSHP->panRecSize[hEntity]+8 >= nOffset + 8 )
        {
            memcpy( psShape->padfM, psSHP->pabyRec + nOffset, 8 );
        
            if( bBigEndian ) SwapWord( 8, psShape->padfM );
        }

/* -------------------------------------------------------------------- */
/*      Since no extents are supplied in the record, we will apply      */
/*      them from the single vertex.                                    */
/* -------------------------------------------------------------------- */
        psShape->dfXMin = psShape->dfXMax = psShape->padfX[0];
        psShape->dfYMin = psShape->dfYMax = psShape->padfY[0];
        psShape->dfZMin = psShape->dfZMax = psShape->padfZ[0];
        psShape->dfMMin = psShape->dfMMax = psShape->padfM[0];
    }

    return( psShape );
}

/************************************************************************/
/*                            SHPTypeName()                             */
/************************************************************************/

const char SHPAPI_CALL1(*)
SHPTypeName( int nSHPType )

{
    switch( nSHPType )
    {
      case SHPT_NULL:
        return "NullShape";

      case SHPT_POINT:
        return "Point";

      case SHPT_ARC:
        return "Arc";

      case SHPT_POLYGON:
        return "Polygon";

      case SHPT_MULTIPOINT:
        return "MultiPoint";
        
      case SHPT_POINTZ:
        return "PointZ";

      case SHPT_ARCZ:
        return "ArcZ";

      case SHPT_POLYGONZ:
        return "PolygonZ";

      case SHPT_MULTIPOINTZ:
        return "MultiPointZ";
        
      case SHPT_POINTM:
        return "PointM";

      case SHPT_ARCM:
        return "ArcM";

      case SHPT_POLYGONM:
        return "PolygonM";

      case SHPT_MULTIPOINTM:
        return "MultiPointM";

      case SHPT_MULTIPATCH:
        return "MultiPatch";

      default:
        return "UnknownShapeType";
    }
}

/************************************************************************/
/*                          SHPPartTypeName()                           */
/************************************************************************/

const char SHPAPI_CALL1(*)
SHPPartTypeName( int nPartType )

{
    switch( nPartType )
    {
      case SHPP_TRISTRIP:
        return "TriangleStrip";
        
      case SHPP_TRIFAN:
        return "TriangleFan";

      case SHPP_OUTERRING:
        return "OuterRing";

      case SHPP_INNERRING:
        return "InnerRing";

      case SHPP_FIRSTRING:
        return "FirstRing";

      case SHPP_RING:
        return "Ring";

      default:
        return "UnknownPartType";
    }
}

/************************************************************************/
/*                          SHPDestroyObject()                          */
/************************************************************************/

void SHPAPI_CALL
SHPDestroyObject( SHPObject * psShape )

{
    if( psShape == NULL )
        return;
    
    if( psShape->padfX != NULL )
        free( psShape->padfX );
    if( psShape->padfY != NULL )
        free( psShape->padfY );
    if( psShape->padfZ != NULL )
        free( psShape->padfZ );
    if( psShape->padfM != NULL )
        free( psShape->padfM );

    if( psShape->panPartStart != NULL )
        free( psShape->panPartStart );
    if( psShape->panPartType != NULL )
        free( psShape->panPartType );

    free( psShape );
}

/************************************************************************/
/*                          SHPRewindObject()                           */
/*                                                                      */
/*      Reset the winding of polygon objects to adhere to the           */
/*      specification.                                                  */
/************************************************************************/

int SHPAPI_CALL
SHPRewindObject( SHPHandle hSHP, SHPObject * psObject )

{
    int  iOpRing, bAltered = 0;

/* -------------------------------------------------------------------- */
/*      Do nothing if this is not a polygon object.                     */
/* -------------------------------------------------------------------- */
    if( psObject->nSHPType != SHPT_POLYGON
        && psObject->nSHPType != SHPT_POLYGONZ
        && psObject->nSHPType != SHPT_POLYGONM )
        return 0;

/* -------------------------------------------------------------------- */
/*      Process each of the rings.                                      */
/* -------------------------------------------------------------------- */
    for( iOpRing = 0; iOpRing < psObject->nParts; iOpRing++ )
    {
        int      bInner, iVert, nVertCount, nVertStart, iCheckRing;
        double   dfSum, dfTestX, dfTestY;

/* -------------------------------------------------------------------- */
/*      Determine if this ring is an inner ring or an outer ring        */
/*      relative to all the other rings.  For now we assume the         */
/*      first ring is outer and all others are inner, but eventually    */
/*      we need to fix this to handle multiple island polygons and      */
/*      unordered sets of rings.                                        */
/* -------------------------------------------------------------------- */
        dfTestX = psObject->padfX[psObject->panPartStart[iOpRing]];
        dfTestY = psObject->padfY[psObject->panPartStart[iOpRing]];

        bInner = FALSE;
        for( iCheckRing = 0; iCheckRing < psObject->nParts; iCheckRing++ )
        {
            int iEdge;

            if( iCheckRing == iOpRing )
                continue;
            
            nVertStart = psObject->panPartStart[iCheckRing];

            if( iCheckRing == psObject->nParts-1 )
                nVertCount = psObject->nVertices 
                    - psObject->panPartStart[iCheckRing];
            else
                nVertCount = psObject->panPartStart[iCheckRing+1] 
                    - psObject->panPartStart[iCheckRing];

            for( iEdge = 0; iEdge < nVertCount; iEdge++ )
            {
                int iNext;

                if( iEdge < nVertCount-1 )
                    iNext = iEdge+1;
                else
                    iNext = 0;

                if( (psObject->padfY[iEdge+nVertStart] < dfTestY 
                     && psObject->padfY[iNext+nVertStart] >= dfTestY)
                    || (psObject->padfY[iNext+nVertStart] < dfTestY 
                        && psObject->padfY[iEdge+nVertStart] >= dfTestY) )
                {
                    if( psObject->padfX[iEdge+nVertStart] 
                        + (dfTestY - psObject->padfY[iEdge+nVertStart])
                           / (psObject->padfY[iNext+nVertStart]
                              - psObject->padfY[iEdge+nVertStart])
                           * (psObject->padfX[iNext+nVertStart]
                              - psObject->padfX[iEdge+nVertStart]) < dfTestX )
                        bInner = !bInner;
                }
            }
        }

/* -------------------------------------------------------------------- */
/*      Determine the current order of this ring so we will know if     */
/*      it has to be reversed.                                          */
/* -------------------------------------------------------------------- */
        nVertStart = psObject->panPartStart[iOpRing];

        if( iOpRing == psObject->nParts-1 )
            nVertCount = psObject->nVertices - psObject->panPartStart[iOpRing];
        else
            nVertCount = psObject->panPartStart[iOpRing+1] 
                - psObject->panPartStart[iOpRing];

        dfSum = 0.0;
        for( iVert = nVertStart; iVert < nVertStart+nVertCount-1; iVert++ )
        {
            dfSum += psObject->padfX[iVert] * psObject->padfY[iVert+1]
                - psObject->padfY[iVert] * psObject->padfX[iVert+1];
        }

        dfSum += psObject->padfX[iVert] * psObject->padfY[nVertStart]
               - psObject->padfY[iVert] * psObject->padfX[nVertStart];

/* -------------------------------------------------------------------- */
/*      Reverse if necessary.                                           */
/* -------------------------------------------------------------------- */
        if( (dfSum < 0.0 && bInner) || (dfSum > 0.0 && !bInner) )
        {
            int   i;

            bAltered++;
            for( i = 0; i < nVertCount/2; i++ )
            {
                double dfSaved;

                /* Swap X */
                dfSaved = psObject->padfX[nVertStart+i];
                psObject->padfX[nVertStart+i] = 
                    psObject->padfX[nVertStart+nVertCount-i-1];
                psObject->padfX[nVertStart+nVertCount-i-1] = dfSaved;

                /* Swap Y */
                dfSaved = psObject->padfY[nVertStart+i];
                psObject->padfY[nVertStart+i] = 
                    psObject->padfY[nVertStart+nVertCount-i-1];
                psObject->padfY[nVertStart+nVertCount-i-1] = dfSaved;

                /* Swap Z */
                if( psObject->padfZ )
                {
                    dfSaved = psObject->padfZ[nVertStart+i];
                    psObject->padfZ[nVertStart+i] = 
                        psObject->padfZ[nVertStart+nVertCount-i-1];
                    psObject->padfZ[nVertStart+nVertCount-i-1] = dfSaved;
                }

                /* Swap M */
                if( psObject->padfM )
                {
                    dfSaved = psObject->padfM[nVertStart+i];
                    psObject->padfM[nVertStart+i] = 
                        psObject->padfM[nVertStart+nVertCount-i-1];
                    psObject->padfM[nVertStart+nVertCount-i-1] = dfSaved;
                }
            }
        }
    }

    return bAltered;
}
































/******************************************************************************
 * $Id: dbfopen.c,v 1.48 2003/03/10 14:51:27 warmerda Exp $
 *
 * Project:  Shapelib
 * Purpose:  Implementation of .dbf access API documented in dbf_api.html.
 * Author:   Frank Warmerdam, warmerdam@pobox.com
 *
 ******************************************************************************
 * Copyright (c) 1999, Frank Warmerdam
 *
 * This software is available under the following "MIT Style" license,
 * or at the option of the licensee under the LGPL (see LICENSE.LGPL).  This
 * option is discussed in more detail in shapelib.html.
 *
 * --
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 ******************************************************************************
 *
 * $Log: dbfopen.c,v $
 * Revision 1.48  2003/03/10 14:51:27  warmerda
 * DBFWrite* calls now return FALSE if they have to truncate
 *
 * Revision 1.47  2002/11/20 03:32:22  warmerda
 * Ensure field name in DBFGetFieldIndex() is properly terminated.
 *
 * Revision 1.46  2002/10/09 13:10:21  warmerda
 * Added check that width is positive.
 *
 * Revision 1.45  2002/09/29 00:00:08  warmerda
 * added FTLogical and logical attribute read/write calls
 *
 * Revision 1.44  2002/05/07 13:46:11  warmerda
 * Added DBFWriteAttributeDirectly().
 *
 * Revision 1.43  2002/02/13 19:39:21  warmerda
 * Fix casting issues in DBFCloneEmpty().
 *
 * Revision 1.42  2002/01/15 14:36:07  warmerda
 * updated email address
 *
 * Revision 1.41  2002/01/15 14:31:49  warmerda
 * compute rather than copying nHeaderLength in DBFCloneEmpty()
 *
 * Revision 1.40  2002/01/09 04:32:35  warmerda
 * fixed to read correct amount of header
 *
 * Revision 1.39  2001/12/11 22:41:03  warmerda
 * improve io related error checking when reading header
 *
 * Revision 1.38  2001/11/28 16:07:31  warmerda
 * Cleanup to avoid compiler warnings as suggested by Richard Hash.
 *
 * Revision 1.37  2001/07/04 05:18:09  warmerda
 * do last fix properly
 *
 * Revision 1.36  2001/07/04 05:16:09  warmerda
 * fixed fieldname comparison in DBFGetFieldIndex
 *
 * Revision 1.35  2001/06/22 02:10:06  warmerda
 * fixed NULL shape support with help from Jim Matthews
 *
 * Revision 1.33  2001/05/31 19:20:13  warmerda
 * added DBFGetFieldIndex()
 *
 * Revision 1.32  2001/05/31 18:15:40  warmerda
 * Added support for NULL fields in DBF files
 *
 * Revision 1.31  2001/05/23 13:36:52  warmerda
 * added use of SHPAPI_CALL
 *
 * Revision 1.30  2000/12/05 14:43:38  warmerda
 * DBReadAttribute() white space trimming bug fix
 *
 * Revision 1.29  2000/10/05 14:36:44  warmerda
 * fix bug with writing very wide numeric fields
 *
 * Revision 1.28  2000/09/25 14:18:07  warmerda
 * Added some casts of strlen() return result to fix warnings on some
 * systems, as submitted by Daniel.
 *
 * Revision 1.27  2000/09/25 14:15:51  warmerda
 * added DBFGetNativeFieldType()
 *
 * Revision 1.26  2000/07/07 13:39:45  warmerda
 * removed unused variables, and added system include files
 *
 * Revision 1.25  2000/05/29 18:19:13  warmerda
 * avoid use of uchar, and adding casting fix
 *
 * Revision 1.24  2000/05/23 13:38:27  warmerda
 * Added error checks on return results of fread() and fseek().
 *
 * Revision 1.23  2000/05/23 13:25:49  warmerda
 * Avoid crashing if field or record are out of range in dbfread*attribute().
 *
 * Revision 1.22  1999/12/15 13:47:24  warmerda
 * Added stdlib.h to ensure that atof() is prototyped.
 *
 * Revision 1.21  1999/12/13 17:25:46  warmerda
 * Added support for upper case .DBF extention.
 *
 * Revision 1.20  1999/11/30 16:32:11  warmerda
 * Use atof() instead of sscanf().
 *
 * Revision 1.19  1999/11/05 14:12:04  warmerda
 * updated license terms
 *
 * Revision 1.18  1999/07/27 00:53:28  warmerda
 * ensure that whole old field value clear on write of string
 *
 * Revision 1.1  1999/07/05 18:58:07  warmerda
 * New
 *
 * Revision 1.17  1999/06/11 19:14:12  warmerda
 * Fixed some memory leaks.
 *
 * Revision 1.16  1999/06/11 19:04:11  warmerda
 * Remoted some unused variables.
 *
 * Revision 1.15  1999/05/11 03:19:28  warmerda
 * added new Tuple api, and improved extension handling - add from candrsn
 *
 * Revision 1.14  1999/05/04 15:01:48  warmerda
 * Added 'F' support.
 *
 * Revision 1.13  1999/03/23 17:38:59  warmerda
 * DBFAddField() now actually does return the new field number, or -1 if
 * it fails.
 *
 * Revision 1.12  1999/03/06 02:54:46  warmerda
 * Added logic to convert shapefile name to dbf filename in DBFOpen()
 * for convenience.
 *
 * Revision 1.11  1998/12/31 15:30:34  warmerda
 * Improved the interchangability of numeric and string attributes.  Add
 * white space trimming option for attributes.
 *
 * Revision 1.10  1998/12/03 16:36:44  warmerda
 * Use r+b instead of rb+ for binary access.
 *
 * Revision 1.9  1998/12/03 15:34:23  warmerda
 * Updated copyright message.
 *
 * Revision 1.8  1997/12/04 15:40:15  warmerda
 * Added newline character after field definitions.
 *
 * Revision 1.7  1997/03/06 14:02:10  warmerda
 * Ensure bUpdated is initialized.
 *
 * Revision 1.6  1996/02/12 04:54:41  warmerda
 * Ensure that DBFWriteAttribute() returns TRUE if it succeeds.
 *
 * Revision 1.5  1995/10/21  03:15:12  warmerda
 * Changed to use binary file access, and ensure that the
 * field name field is zero filled, and limited to 10 chars.
 *
 * Revision 1.4  1995/08/24  18:10:42  warmerda
 * Added use of SfRealloc() to avoid pre-ANSI realloc() functions such
 * as on the Sun.
 *
 * Revision 1.3  1995/08/04  03:15:16  warmerda
 * Fixed up header.
 *
 * Revision 1.2  1995/08/04  03:14:43  warmerda
 * Added header.
 */


#ifndef FALSE
#  define FALSE		0
#  define TRUE		1
#endif

static int	nStringFieldLen = 0;
static char * pszStringField = NULL;


/************************************************************************/
/*                           DBFWriteHeader()                           */
/*                                                                      */
/*      This is called to write out the file header, and field          */
/*      descriptions before writing any actual data records.  This      */
/*      also computes all the DBFDataSet field offset/size/decimals     */
/*      and so forth values.                                            */
/************************************************************************/

static void DBFWriteHeader(DBFHandle psDBF)

{
    unsigned char	abyHeader[XBASE_FLDHDR_SZ];
    int		i;

    if( !psDBF->bNoHeader )
        return;

    psDBF->bNoHeader = FALSE;

/* -------------------------------------------------------------------- */
/*	Initialize the file header information.				*/
/* -------------------------------------------------------------------- */
    for( i = 0; i < XBASE_FLDHDR_SZ; i++ )
        abyHeader[i] = 0;

    abyHeader[0] = 0x03;		/* memo field? - just copying 	*/

    /* date updated on close, record count preset at zero */

    abyHeader[8] = psDBF->nHeaderLength % 256;
    abyHeader[9] = psDBF->nHeaderLength / 256;
    
    abyHeader[10] = psDBF->nRecordLength % 256;
    abyHeader[11] = psDBF->nRecordLength / 256;

/* -------------------------------------------------------------------- */
/*      Write the initial 32 byte file header, and all the field        */
/*      descriptions.                                     		*/
/* -------------------------------------------------------------------- */
    fseek( psDBF->fp, 0, 0 );
    fwrite( abyHeader, XBASE_FLDHDR_SZ, 1, psDBF->fp );
    fwrite( psDBF->pszHeader, XBASE_FLDHDR_SZ, psDBF->nFields, psDBF->fp );

/* -------------------------------------------------------------------- */
/*      Write out the newline character if there is room for it.        */
/* -------------------------------------------------------------------- */
    if( psDBF->nHeaderLength > 32*psDBF->nFields + 32 )
    {
        char	cNewline;

        cNewline = 0x0d;
        fwrite( &cNewline, 1, 1, psDBF->fp );
    }
}

/************************************************************************/
/*                           DBFFlushRecord()                           */
/*                                                                      */
/*      Write out the current record if there is one.                   */
/************************************************************************/

static void DBFFlushRecord( DBFHandle psDBF )

{
    int		nRecordOffset;

    if( psDBF->bCurrentRecordModified && psDBF->nCurrentRecord > -1 )
    {
	psDBF->bCurrentRecordModified = FALSE;

	nRecordOffset = psDBF->nRecordLength * psDBF->nCurrentRecord 
	                                             + psDBF->nHeaderLength;

	fseek( psDBF->fp, nRecordOffset, 0 );
	fwrite( psDBF->pszCurrentRecord, psDBF->nRecordLength, 1, psDBF->fp );
    }
}

/************************************************************************/
/*                              DBFOpen()                               */
/*                                                                      */
/*      Open a .dbf file.                                               */
/************************************************************************/
   
DBFHandle SHPAPI_CALL
DBFOpen( const char * pszFilename, const char * pszAccess )

{
    DBFHandle		psDBF;
    unsigned char		*pabyBuf;
    int			nFields, nHeadLen, nRecLen, iField, i;
    char		*pszBasename, *pszFullname;

/* -------------------------------------------------------------------- */
/*      We only allow the access strings "rb" and "r+".                  */
/* -------------------------------------------------------------------- */
    if( strcmp(pszAccess,"r") != 0 && strcmp(pszAccess,"r+") != 0 
        && strcmp(pszAccess,"rb") != 0 && strcmp(pszAccess,"rb+") != 0
        && strcmp(pszAccess,"r+b") != 0 )
        return( NULL );

    if( strcmp(pszAccess,"r") == 0 )
        pszAccess = "rb";
 
    if( strcmp(pszAccess,"r+") == 0 )
        pszAccess = "rb+";

/* -------------------------------------------------------------------- */
/*	Compute the base (layer) name.  If there is any extension	*/
/*	on the passed in filename we will strip it off.			*/
/* -------------------------------------------------------------------- */
    pszBasename = (char *) malloc(strlen(pszFilename)+5);
    strcpy( pszBasename, pszFilename );
    for( i = strlen(pszBasename)-1; 
	 i > 0 && pszBasename[i] != '.' && pszBasename[i] != '/'
	       && pszBasename[i] != '\\';
	 i-- ) {}

    if( pszBasename[i] == '.' )
        pszBasename[i] = '\0';

    pszFullname = (char *) malloc(strlen(pszBasename) + 5);
    sprintf( pszFullname, "%s.dbf", pszBasename );
        
    psDBF = (DBFHandle) calloc( 1, sizeof(DBFInfo) );
    psDBF->fp = fopen( pszFullname, pszAccess );

    if( psDBF->fp == NULL )
    {
        sprintf( pszFullname, "%s.DBF", pszBasename );
        psDBF->fp = fopen(pszFullname, pszAccess );
    }
    
    free( pszBasename );
    free( pszFullname );
    
    if( psDBF->fp == NULL )
    {
        free( psDBF );
        return( NULL );
    }

    psDBF->bNoHeader = FALSE;
    psDBF->nCurrentRecord = -1;
    psDBF->bCurrentRecordModified = FALSE;

/* -------------------------------------------------------------------- */
/*  Read Table Header info                                              */
/* -------------------------------------------------------------------- */
    pabyBuf = (unsigned char *) malloc(500);
    if( fread( pabyBuf, 32, 1, psDBF->fp ) != 1 )
    {
        fclose( psDBF->fp );
        free( pabyBuf );
        free( psDBF );
        return NULL;
    }

    psDBF->nRecords = 
     pabyBuf[4] + pabyBuf[5]*256 + pabyBuf[6]*256*256 + pabyBuf[7]*256*256*256;

    psDBF->nHeaderLength = nHeadLen = pabyBuf[8] + pabyBuf[9]*256;
    psDBF->nRecordLength = nRecLen = pabyBuf[10] + pabyBuf[11]*256;
    
    psDBF->nFields = nFields = (nHeadLen - 32) / 32;

    psDBF->pszCurrentRecord = (char *) malloc(nRecLen);

/* -------------------------------------------------------------------- */
/*  Read in Field Definitions                                           */
/* -------------------------------------------------------------------- */
    
    pabyBuf = (unsigned char *) SfRealloc(pabyBuf,nHeadLen);
    psDBF->pszHeader = (char *) pabyBuf;

    fseek( psDBF->fp, 32, 0 );
    if( fread( pabyBuf, nHeadLen-32, 1, psDBF->fp ) != 1 )
    {
        fclose( psDBF->fp );
        free( pabyBuf );
        free( psDBF );
        return NULL;
    }

    psDBF->panFieldOffset = (int *) malloc(sizeof(int) * nFields);
    psDBF->panFieldSize = (int *) malloc(sizeof(int) * nFields);
    psDBF->panFieldDecimals = (int *) malloc(sizeof(int) * nFields);
    psDBF->pachFieldType = (char *) malloc(sizeof(char) * nFields);

    for( iField = 0; iField < nFields; iField++ )
    {
	unsigned char		*pabyFInfo;

	pabyFInfo = pabyBuf+iField*32;

	if( pabyFInfo[11] == 'N' || pabyFInfo[11] == 'F' )
	{
	    psDBF->panFieldSize[iField] = pabyFInfo[16];
	    psDBF->panFieldDecimals[iField] = pabyFInfo[17];
	}
	else
	{
	    psDBF->panFieldSize[iField] = pabyFInfo[16] + pabyFInfo[17]*256;
	    psDBF->panFieldDecimals[iField] = 0;
	}

	psDBF->pachFieldType[iField] = (char) pabyFInfo[11];
	if( iField == 0 )
	    psDBF->panFieldOffset[iField] = 1;
	else
	    psDBF->panFieldOffset[iField] = 
	      psDBF->panFieldOffset[iField-1] + psDBF->panFieldSize[iField-1];
    }

    return( psDBF );
}

/************************************************************************/
/*                              DBFClose()                              */
/************************************************************************/

void SHPAPI_CALL
DBFClose(DBFHandle psDBF)
{
/* -------------------------------------------------------------------- */
/*      Write out header if not already written.                        */
/* -------------------------------------------------------------------- */
    if( psDBF->bNoHeader )
        DBFWriteHeader( psDBF );

    DBFFlushRecord( psDBF );

/* -------------------------------------------------------------------- */
/*      Update last access date, and number of records if we have	*/
/*	write access.                					*/
/* -------------------------------------------------------------------- */
    if( psDBF->bUpdated )
    {
	unsigned char		abyFileHeader[32];

	fseek( psDBF->fp, 0, 0 );
	fread( abyFileHeader, 32, 1, psDBF->fp );

	abyFileHeader[1] = 95;			/* YY */
	abyFileHeader[2] = 7;			/* MM */
	abyFileHeader[3] = 26;			/* DD */

	abyFileHeader[4] = psDBF->nRecords % 256;
	abyFileHeader[5] = (psDBF->nRecords/256) % 256;
	abyFileHeader[6] = (psDBF->nRecords/(256*256)) % 256;
	abyFileHeader[7] = (psDBF->nRecords/(256*256*256)) % 256;

	fseek( psDBF->fp, 0, 0 );
	fwrite( abyFileHeader, 32, 1, psDBF->fp );
    }

/* -------------------------------------------------------------------- */
/*      Close, and free resources.                                      */
/* -------------------------------------------------------------------- */
    fclose( psDBF->fp );

    if( psDBF->panFieldOffset != NULL )
    {
        free( psDBF->panFieldOffset );
        free( psDBF->panFieldSize );
        free( psDBF->panFieldDecimals );
        free( psDBF->pachFieldType );
    }

    free( psDBF->pszHeader );
    free( psDBF->pszCurrentRecord );

    free( psDBF );

    if( pszStringField != NULL )
    {
        free( pszStringField );
        pszStringField = NULL;
        nStringFieldLen = 0;
    }
}

/************************************************************************/
/*                             DBFCreate()                              */
/*                                                                      */
/*      Create a new .dbf file.                                         */
/************************************************************************/

DBFHandle SHPAPI_CALL
DBFCreate( const char * pszFilename )

{
    DBFHandle	psDBF;
    FILE	*fp;
    char	*pszFullname, *pszBasename;
    int		i;

/* -------------------------------------------------------------------- */
/*	Compute the base (layer) name.  If there is any extension	*/
/*	on the passed in filename we will strip it off.			*/
/* -------------------------------------------------------------------- */
    pszBasename = (char *) malloc(strlen(pszFilename)+5);
    strcpy( pszBasename, pszFilename );
    for( i = strlen(pszBasename)-1; 
	 i > 0 && pszBasename[i] != '.' && pszBasename[i] != '/'
	       && pszBasename[i] != '\\';
	 i-- ) {}

    if( pszBasename[i] == '.' )
        pszBasename[i] = '\0';

    pszFullname = (char *) malloc(strlen(pszBasename) + 5);
    sprintf( pszFullname, "%s.dbf", pszBasename );
    free( pszBasename );

/* -------------------------------------------------------------------- */
/*      Create the file.                                                */
/* -------------------------------------------------------------------- */
    fp = fopen( pszFullname, "wb" );
    if( fp == NULL )
        return( NULL );

    fputc( 0, fp );
    fclose( fp );

    fp = fopen( pszFullname, "rb+" );
    if( fp == NULL )
        return( NULL );

    free( pszFullname );

/* -------------------------------------------------------------------- */
/*	Create the info structure.					*/
/* -------------------------------------------------------------------- */
    psDBF = (DBFHandle) malloc(sizeof(DBFInfo));

    psDBF->fp = fp;
    psDBF->nRecords = 0;
    psDBF->nFields = 0;
    psDBF->nRecordLength = 1;
    psDBF->nHeaderLength = 33;
    
    psDBF->panFieldOffset = NULL;
    psDBF->panFieldSize = NULL;
    psDBF->panFieldDecimals = NULL;
    psDBF->pachFieldType = NULL;
    psDBF->pszHeader = NULL;

    psDBF->nCurrentRecord = -1;
    psDBF->bCurrentRecordModified = FALSE;
    psDBF->pszCurrentRecord = NULL;

    psDBF->bNoHeader = TRUE;

    return( psDBF );
}

/************************************************************************/
/*                            DBFAddField()                             */
/*                                                                      */
/*      Add a field to a newly created .dbf file before any records     */
/*      are written.                                                    */
/************************************************************************/

int SHPAPI_CALL
DBFAddField(DBFHandle psDBF, const char * pszFieldName, 
            DBFFieldType eType, int nWidth, int nDecimals )

{
    char	*pszFInfo;
    int		i;

/* -------------------------------------------------------------------- */
/*      Do some checking to ensure we can add records to this file.     */
/* -------------------------------------------------------------------- */
    if( psDBF->nRecords > 0 )
        return( -1 );

    if( !psDBF->bNoHeader )
        return( -1 );

    if( eType != FTDouble && nDecimals != 0 )
        return( -1 );

    if( nWidth < 1 )
        return -1;

/* -------------------------------------------------------------------- */
/*      SfRealloc all the arrays larger to hold the additional field      */
/*      information.                                                    */
/* -------------------------------------------------------------------- */
    psDBF->nFields++;

    psDBF->panFieldOffset = (int *) 
      SfRealloc( psDBF->panFieldOffset, sizeof(int) * psDBF->nFields );

    psDBF->panFieldSize = (int *) 
      SfRealloc( psDBF->panFieldSize, sizeof(int) * psDBF->nFields );

    psDBF->panFieldDecimals = (int *) 
      SfRealloc( psDBF->panFieldDecimals, sizeof(int) * psDBF->nFields );

    psDBF->pachFieldType = (char *) 
      SfRealloc( psDBF->pachFieldType, sizeof(char) * psDBF->nFields );

/* -------------------------------------------------------------------- */
/*      Assign the new field information fields.                        */
/* -------------------------------------------------------------------- */
    psDBF->panFieldOffset[psDBF->nFields-1] = psDBF->nRecordLength;
    psDBF->nRecordLength += nWidth;
    psDBF->panFieldSize[psDBF->nFields-1] = nWidth;
    psDBF->panFieldDecimals[psDBF->nFields-1] = nDecimals;

    if( eType == FTLogical )
        psDBF->pachFieldType[psDBF->nFields-1] = 'L';
    else if( eType == FTString )
        psDBF->pachFieldType[psDBF->nFields-1] = 'C';
    else
        psDBF->pachFieldType[psDBF->nFields-1] = 'N';

/* -------------------------------------------------------------------- */
/*      Extend the required header information.                         */
/* -------------------------------------------------------------------- */
    psDBF->nHeaderLength += 32;
    psDBF->bUpdated = FALSE;

    psDBF->pszHeader = (char *) SfRealloc(psDBF->pszHeader,psDBF->nFields*32);

    pszFInfo = psDBF->pszHeader + 32 * (psDBF->nFields-1);

    for( i = 0; i < 32; i++ )
        pszFInfo[i] = '\0';

    if( (int) strlen(pszFieldName) < 10 )
        strncpy( pszFInfo, pszFieldName, strlen(pszFieldName));
    else
        strncpy( pszFInfo, pszFieldName, 10);

    pszFInfo[11] = psDBF->pachFieldType[psDBF->nFields-1];

    if( eType == FTString )
    {
        pszFInfo[16] = nWidth % 256;
        pszFInfo[17] = nWidth / 256;
    }
    else
    {
        pszFInfo[16] = nWidth;
        pszFInfo[17] = nDecimals;
    }
    
/* -------------------------------------------------------------------- */
/*      Make the current record buffer appropriately larger.            */
/* -------------------------------------------------------------------- */
    psDBF->pszCurrentRecord = (char *) SfRealloc(psDBF->pszCurrentRecord,
					       psDBF->nRecordLength);

    return( psDBF->nFields-1 );
}

/************************************************************************/
/*                          DBFReadAttribute()                          */
/*                                                                      */
/*      Read one of the attribute fields of a record.                   */
/************************************************************************/

static void *DBFReadAttribute(DBFHandle psDBF, int hEntity, int iField,
                              char chReqType )

{
    int	       	nRecordOffset;
    unsigned char	*pabyRec;
    void	*pReturnField = NULL;

    static double dDoubleField;

/* -------------------------------------------------------------------- */
/*      Verify selection.                                               */
/* -------------------------------------------------------------------- */
    if( hEntity < 0 || hEntity >= psDBF->nRecords )
        return( NULL );

    if( iField < 0 || iField >= psDBF->nFields )
        return( NULL );

/* -------------------------------------------------------------------- */
/*	Have we read the record?					*/
/* -------------------------------------------------------------------- */
    if( psDBF->nCurrentRecord != hEntity )
    {
	DBFFlushRecord( psDBF );

	nRecordOffset = psDBF->nRecordLength * hEntity + psDBF->nHeaderLength;

	if( fseek( psDBF->fp, nRecordOffset, 0 ) != 0 )
        {
            fprintf( stderr, "fseek(%d) failed on DBF file.\n",
                     nRecordOffset );
            return NULL;
        }

	if( fread( psDBF->pszCurrentRecord, psDBF->nRecordLength, 
                   1, psDBF->fp ) != 1 )
        {
            fprintf( stderr, "fread(%d) failed on DBF file.\n",
                     psDBF->nRecordLength );
            return NULL;
        }

	psDBF->nCurrentRecord = hEntity;
    }

    pabyRec = (unsigned char *) psDBF->pszCurrentRecord;

/* -------------------------------------------------------------------- */
/*	Ensure our field buffer is large enough to hold this buffer.	*/
/* -------------------------------------------------------------------- */
    if( psDBF->panFieldSize[iField]+1 > nStringFieldLen )
    {
	nStringFieldLen = psDBF->panFieldSize[iField]*2 + 10;
	pszStringField = (char *) SfRealloc(pszStringField,nStringFieldLen);
    }

/* -------------------------------------------------------------------- */
/*	Extract the requested field.					*/
/* -------------------------------------------------------------------- */
    strncpy( pszStringField, 
	     ((const char *) pabyRec) + psDBF->panFieldOffset[iField],
	     psDBF->panFieldSize[iField] );
    pszStringField[psDBF->panFieldSize[iField]] = '\0';

    pReturnField = pszStringField;

/* -------------------------------------------------------------------- */
/*      Decode the field.                                               */
/* -------------------------------------------------------------------- */
    if( chReqType == 'N' )
    {
        dDoubleField = atof(pszStringField);

	pReturnField = &dDoubleField;
    }

/* -------------------------------------------------------------------- */
/*      Should we trim white space off the string attribute value?      */
/* -------------------------------------------------------------------- */
#ifdef TRIM_DBF_WHITESPACE
    else
    {
        char	*pchSrc, *pchDst;

        pchDst = pchSrc = pszStringField;
        while( *pchSrc == ' ' )
            pchSrc++;

        while( *pchSrc != '\0' )
            *(pchDst++) = *(pchSrc++);
        *pchDst = '\0';

        while( pchDst != pszStringField && *(--pchDst) == ' ' )
            *pchDst = '\0';
    }
#endif
    
    return( pReturnField );
}

/************************************************************************/
/*                        DBFReadIntAttribute()                         */
/*                                                                      */
/*      Read an integer attribute.                                      */
/************************************************************************/

int SHPAPI_CALL
DBFReadIntegerAttribute( DBFHandle psDBF, int iRecord, int iField )

{
    double	*pdValue;

    pdValue = (double *) DBFReadAttribute( psDBF, iRecord, iField, 'N' );

    if( pdValue == NULL )
        return 0;
    else
        return( (int) *pdValue );
}

/************************************************************************/
/*                        DBFReadDoubleAttribute()                      */
/*                                                                      */
/*      Read a double attribute.                                        */
/************************************************************************/

double SHPAPI_CALL
DBFReadDoubleAttribute( DBFHandle psDBF, int iRecord, int iField )

{
    double	*pdValue;

    pdValue = (double *) DBFReadAttribute( psDBF, iRecord, iField, 'N' );

    if( pdValue == NULL )
        return 0.0;
    else
        return( *pdValue );
}

/************************************************************************/
/*                        DBFReadStringAttribute()                      */
/*                                                                      */
/*      Read a string attribute.                                        */
/************************************************************************/

const char SHPAPI_CALL1(*)
DBFReadStringAttribute( DBFHandle psDBF, int iRecord, int iField )

{
    return( (const char *) DBFReadAttribute( psDBF, iRecord, iField, 'C' ) );
}

/************************************************************************/
/*                        DBFReadLogicalAttribute()                     */
/*                                                                      */
/*      Read a logical attribute.                                       */
/************************************************************************/

const char SHPAPI_CALL1(*)
DBFReadLogicalAttribute( DBFHandle psDBF, int iRecord, int iField )

{
    return( (const char *) DBFReadAttribute( psDBF, iRecord, iField, 'L' ) );
}

/************************************************************************/
/*                         DBFIsAttributeNULL()                         */
/*                                                                      */
/*      Return TRUE if value for field is NULL.                         */
/*                                                                      */
/*      Contributed by Jim Matthews.                                    */
/************************************************************************/

int SHPAPI_CALL
DBFIsAttributeNULL( DBFHandle psDBF, int iRecord, int iField )

{
    const char	*pszValue;

    pszValue = DBFReadStringAttribute( psDBF, iRecord, iField );

    switch(psDBF->pachFieldType[iField])
    {
      case 'N':
      case 'F':
        /* NULL numeric fields have value "****************" */
        return pszValue[0] == '*';

      case 'D':
        /* NULL date fields have value "00000000" */
        return strncmp(pszValue,"00000000",8) == 0;

      case 'L':
        /* NULL boolean fields have value "?" */ 
        return pszValue[0] == '?';

      default:
        /* empty string fields are considered NULL */
        return strlen(pszValue) == 0;
    }
}

/************************************************************************/
/*                          DBFGetFieldCount()                          */
/*                                                                      */
/*      Return the number of fields in this table.                      */
/************************************************************************/

int SHPAPI_CALL
DBFGetFieldCount( DBFHandle psDBF )

{
    return( psDBF->nFields );
}

/************************************************************************/
/*                         DBFGetRecordCount()                          */
/*                                                                      */
/*      Return the number of records in this table.                     */
/************************************************************************/

int SHPAPI_CALL
DBFGetRecordCount( DBFHandle psDBF )

{
    return( psDBF->nRecords );
}

/************************************************************************/
/*                          DBFGetFieldInfo()                           */
/*                                                                      */
/*      Return any requested information about the field.               */
/************************************************************************/

DBFFieldType SHPAPI_CALL
DBFGetFieldInfo( DBFHandle psDBF, int iField, char * pszFieldName,
                 int * pnWidth, int * pnDecimals )

{
    if( iField < 0 || iField >= psDBF->nFields )
        return( FTInvalid );

    if( pnWidth != NULL )
        *pnWidth = psDBF->panFieldSize[iField];

    if( pnDecimals != NULL )
        *pnDecimals = psDBF->panFieldDecimals[iField];

    if( pszFieldName != NULL )
    {
	int	i;

	strncpy( pszFieldName, (char *) psDBF->pszHeader+iField*32, 11 );
	pszFieldName[11] = '\0';
	for( i = 10; i > 0 && pszFieldName[i] == ' '; i-- )
	    pszFieldName[i] = '\0';
    }

    if ( psDBF->pachFieldType[iField] == 'L' )
	return( FTLogical);

    else if( psDBF->pachFieldType[iField] == 'N' 
             || psDBF->pachFieldType[iField] == 'F'
             || psDBF->pachFieldType[iField] == 'D' )
    {
	if( psDBF->panFieldDecimals[iField] > 0 )
	    return( FTDouble );
	else
	    return( FTInteger );
    }
    else
    {
	return( FTString );
    }
}

/************************************************************************/
/*                         DBFWriteAttribute()                          */
/*									*/
/*	Write an attribute record to the file.				*/
/************************************************************************/

static int DBFWriteAttribute(DBFHandle psDBF, int hEntity, int iField,
			     void * pValue )

{
    int	       	nRecordOffset, i, j, nRetResult = TRUE;
    unsigned char	*pabyRec;
    char	szSField[400], szFormat[20];

/* -------------------------------------------------------------------- */
/*	Is this a valid record?						*/
/* -------------------------------------------------------------------- */
    if( hEntity < 0 || hEntity > psDBF->nRecords )
        return( FALSE );

    if( psDBF->bNoHeader )
        DBFWriteHeader(psDBF);

/* -------------------------------------------------------------------- */
/*      Is this a brand new record?                                     */
/* -------------------------------------------------------------------- */
    if( hEntity == psDBF->nRecords )
    {
	DBFFlushRecord( psDBF );

	psDBF->nRecords++;
	for( i = 0; i < psDBF->nRecordLength; i++ )
	    psDBF->pszCurrentRecord[i] = ' ';

	psDBF->nCurrentRecord = hEntity;
    }

/* -------------------------------------------------------------------- */
/*      Is this an existing record, but different than the last one     */
/*      we accessed?                                                    */
/* -------------------------------------------------------------------- */
    if( psDBF->nCurrentRecord != hEntity )
    {
	DBFFlushRecord( psDBF );

	nRecordOffset = psDBF->nRecordLength * hEntity + psDBF->nHeaderLength;

	fseek( psDBF->fp, nRecordOffset, 0 );
	fread( psDBF->pszCurrentRecord, psDBF->nRecordLength, 1, psDBF->fp );

	psDBF->nCurrentRecord = hEntity;
    }

    pabyRec = (unsigned char *) psDBF->pszCurrentRecord;

    psDBF->bCurrentRecordModified = TRUE;
    psDBF->bUpdated = TRUE;

/* -------------------------------------------------------------------- */
/*      Translate NULL value to valid DBF file representation.          */
/*                                                                      */
/*      Contributed by Jim Matthews.                                    */
/* -------------------------------------------------------------------- */
    if( pValue == NULL )
    {
        switch(psDBF->pachFieldType[iField])
        {
          case 'N':
          case 'F':
	    /* NULL numeric fields have value "****************" */
            memset( (char *) (pabyRec+psDBF->panFieldOffset[iField]), '*', 
                    psDBF->panFieldSize[iField] );
            break;

          case 'D':
	    /* NULL date fields have value "00000000" */
            memset( (char *) (pabyRec+psDBF->panFieldOffset[iField]), '0', 
                    psDBF->panFieldSize[iField] );
            break;

          case 'L':
	    /* NULL boolean fields have value "?" */ 
            memset( (char *) (pabyRec+psDBF->panFieldOffset[iField]), '?', 
                    psDBF->panFieldSize[iField] );
            break;

          default:
            /* empty string fields are considered NULL */
            memset( (char *) (pabyRec+psDBF->panFieldOffset[iField]), '\0', 
                    psDBF->panFieldSize[iField] );
            break;
        }
        return TRUE;
    }

/* -------------------------------------------------------------------- */
/*      Assign all the record fields.                                   */
/* -------------------------------------------------------------------- */
    switch( psDBF->pachFieldType[iField] )
    {
      case 'D':
      case 'N':
      case 'F':
	if( psDBF->panFieldDecimals[iField] == 0 )
	{
            int		nWidth = psDBF->panFieldSize[iField];

            if( sizeof(szSField)-2 < nWidth )
                nWidth = sizeof(szSField)-2;

	    sprintf( szFormat, "%%%dd", nWidth );
	    sprintf(szSField, szFormat, (int) *((double *) pValue) );
	    if( (int)strlen(szSField) > psDBF->panFieldSize[iField] )
            {
	        szSField[psDBF->panFieldSize[iField]] = '\0';
                nRetResult = FALSE;
            }

	    strncpy((char *) (pabyRec+psDBF->panFieldOffset[iField]),
		    szSField, strlen(szSField) );
	}
	else
	{
            int		nWidth = psDBF->panFieldSize[iField];

            if( sizeof(szSField)-2 < nWidth )
                nWidth = sizeof(szSField)-2;

	    sprintf( szFormat, "%%%d.%df", 
                     nWidth, psDBF->panFieldDecimals[iField] );
	    sprintf(szSField, szFormat, *((double *) pValue) );
	    if( (int) strlen(szSField) > psDBF->panFieldSize[iField] )
            {
	        szSField[psDBF->panFieldSize[iField]] = '\0';
                nRetResult = FALSE;
            }
	    strncpy((char *) (pabyRec+psDBF->panFieldOffset[iField]),
		    szSField, strlen(szSField) );
	}
	break;

      case 'L':
        if (psDBF->panFieldSize[iField] >= 1  && 
            (*(char*)pValue == 'F' || *(char*)pValue == 'T'))
            *(pabyRec+psDBF->panFieldOffset[iField]) = *(char*)pValue;
        break;

      default:
	if( (int) strlen((char *) pValue) > psDBF->panFieldSize[iField] )
        {
	    j = psDBF->panFieldSize[iField];
            nRetResult = FALSE;
        }
	else
        {
            memset( pabyRec+psDBF->panFieldOffset[iField], ' ',
                    psDBF->panFieldSize[iField] );
	    j = strlen((char *) pValue);
        }

	strncpy((char *) (pabyRec+psDBF->panFieldOffset[iField]),
		(char *) pValue, j );
	break;
    }

    return( nRetResult );
}

/************************************************************************/
/*                     DBFWriteAttributeDirectly()                      */
/*                                                                      */
/*      Write an attribute record to the file, but without any          */
/*      reformatting based on type.  The provided buffer is written     */
/*      as is to the field position in the record.                      */
/************************************************************************/

int DBFWriteAttributeDirectly(DBFHandle psDBF, int hEntity, int iField,
                              void * pValue )

{
    int	       	nRecordOffset, i, j;
    unsigned char	*pabyRec;

/* -------------------------------------------------------------------- */
/*	Is this a valid record?						*/
/* -------------------------------------------------------------------- */
    if( hEntity < 0 || hEntity > psDBF->nRecords )
        return( FALSE );

    if( psDBF->bNoHeader )
        DBFWriteHeader(psDBF);

/* -------------------------------------------------------------------- */
/*      Is this a brand new record?                                     */
/* -------------------------------------------------------------------- */
    if( hEntity == psDBF->nRecords )
    {
	DBFFlushRecord( psDBF );

	psDBF->nRecords++;
	for( i = 0; i < psDBF->nRecordLength; i++ )
	    psDBF->pszCurrentRecord[i] = ' ';

	psDBF->nCurrentRecord = hEntity;
    }

/* -------------------------------------------------------------------- */
/*      Is this an existing record, but different than the last one     */
/*      we accessed?                                                    */
/* -------------------------------------------------------------------- */
    if( psDBF->nCurrentRecord != hEntity )
    {
	DBFFlushRecord( psDBF );

	nRecordOffset = psDBF->nRecordLength * hEntity + psDBF->nHeaderLength;

	fseek( psDBF->fp, nRecordOffset, 0 );
	fread( psDBF->pszCurrentRecord, psDBF->nRecordLength, 1, psDBF->fp );

	psDBF->nCurrentRecord = hEntity;
    }

    pabyRec = (unsigned char *) psDBF->pszCurrentRecord;

/* -------------------------------------------------------------------- */
/*      Assign all the record fields.                                   */
/* -------------------------------------------------------------------- */
    if( (int)strlen((char *) pValue) > psDBF->panFieldSize[iField] )
        j = psDBF->panFieldSize[iField];
    else
    {
        memset( pabyRec+psDBF->panFieldOffset[iField], ' ',
                psDBF->panFieldSize[iField] );
        j = strlen((char *) pValue);
    }

    strncpy((char *) (pabyRec+psDBF->panFieldOffset[iField]),
            (char *) pValue, j );

    psDBF->bCurrentRecordModified = TRUE;
    psDBF->bUpdated = TRUE;

    return( TRUE );
}

/************************************************************************/
/*                      DBFWriteDoubleAttribute()                       */
/*                                                                      */
/*      Write a double attribute.                                       */
/************************************************************************/

int SHPAPI_CALL
DBFWriteDoubleAttribute( DBFHandle psDBF, int iRecord, int iField,
                         double dValue )

{
    return( DBFWriteAttribute( psDBF, iRecord, iField, (void *) &dValue ) );
}

/************************************************************************/
/*                      DBFWriteIntegerAttribute()                      */
/*                                                                      */
/*      Write a integer attribute.                                      */
/************************************************************************/

int SHPAPI_CALL
DBFWriteIntegerAttribute( DBFHandle psDBF, int iRecord, int iField,
                          int nValue )

{
    double	dValue = nValue;

    return( DBFWriteAttribute( psDBF, iRecord, iField, (void *) &dValue ) );
}

/************************************************************************/
/*                      DBFWriteStringAttribute()                       */
/*                                                                      */
/*      Write a string attribute.                                       */
/************************************************************************/

int SHPAPI_CALL
DBFWriteStringAttribute( DBFHandle psDBF, int iRecord, int iField,
                         const char * pszValue )

{
    return( DBFWriteAttribute( psDBF, iRecord, iField, (void *) pszValue ) );
}

/************************************************************************/
/*                      DBFWriteNULLAttribute()                         */
/*                                                                      */
/*      Write a string attribute.                                       */
/************************************************************************/

int SHPAPI_CALL
DBFWriteNULLAttribute( DBFHandle psDBF, int iRecord, int iField )

{
    return( DBFWriteAttribute( psDBF, iRecord, iField, NULL ) );
}

/************************************************************************/
/*                      DBFWriteLogicalAttribute()                      */
/*                                                                      */
/*      Write a logical attribute.                                      */
/************************************************************************/

int SHPAPI_CALL
DBFWriteLogicalAttribute( DBFHandle psDBF, int iRecord, int iField,
		       const char lValue)

{
    return( DBFWriteAttribute( psDBF, iRecord, iField, (void *) (&lValue) ) );
}

/************************************************************************/
/*                         DBFWriteTuple()                              */
/*									*/
/*	Write an attribute record to the file.				*/
/************************************************************************/

int SHPAPI_CALL
DBFWriteTuple(DBFHandle psDBF, int hEntity, void * pRawTuple )

{
    int	       	nRecordOffset, i;
    unsigned char	*pabyRec;

/* -------------------------------------------------------------------- */
/*	Is this a valid record?						*/
/* -------------------------------------------------------------------- */
    if( hEntity < 0 || hEntity > psDBF->nRecords )
        return( FALSE );

    if( psDBF->bNoHeader )
        DBFWriteHeader(psDBF);

/* -------------------------------------------------------------------- */
/*      Is this a brand new record?                                     */
/* -------------------------------------------------------------------- */
    if( hEntity == psDBF->nRecords )
    {
	DBFFlushRecord( psDBF );

	psDBF->nRecords++;
	for( i = 0; i < psDBF->nRecordLength; i++ )
	    psDBF->pszCurrentRecord[i] = ' ';

	psDBF->nCurrentRecord = hEntity;
    }

/* -------------------------------------------------------------------- */
/*      Is this an existing record, but different than the last one     */
/*      we accessed?                                                    */
/* -------------------------------------------------------------------- */
    if( psDBF->nCurrentRecord != hEntity )
    {
	DBFFlushRecord( psDBF );

	nRecordOffset = psDBF->nRecordLength * hEntity + psDBF->nHeaderLength;

	fseek( psDBF->fp, nRecordOffset, 0 );
	fread( psDBF->pszCurrentRecord, psDBF->nRecordLength, 1, psDBF->fp );

	psDBF->nCurrentRecord = hEntity;
    }

    pabyRec = (unsigned char *) psDBF->pszCurrentRecord;

    memcpy ( pabyRec, pRawTuple,  psDBF->nRecordLength );

    psDBF->bCurrentRecordModified = TRUE;
    psDBF->bUpdated = TRUE;

    return( TRUE );
}

/************************************************************************/
/*                          DBFReadTuple()                              */
/*                                                                      */
/*      Read one of the attribute fields of a record.                   */
/************************************************************************/

const char SHPAPI_CALL1(*)
DBFReadTuple(DBFHandle psDBF, int hEntity )

{
    int	       	nRecordOffset;
    unsigned char	*pabyRec;
    static char	*pReturnTuple = NULL;

    static int	nTupleLen = 0;

/* -------------------------------------------------------------------- */
/*	Have we read the record?					*/
/* -------------------------------------------------------------------- */
    if( hEntity < 0 || hEntity >= psDBF->nRecords )
        return( NULL );

    if( psDBF->nCurrentRecord != hEntity )
    {
	DBFFlushRecord( psDBF );

	nRecordOffset = psDBF->nRecordLength * hEntity + psDBF->nHeaderLength;

	fseek( psDBF->fp, nRecordOffset, 0 );
	fread( psDBF->pszCurrentRecord, psDBF->nRecordLength, 1, psDBF->fp );

	psDBF->nCurrentRecord = hEntity;
    }

    pabyRec = (unsigned char *) psDBF->pszCurrentRecord;

    if ( nTupleLen < psDBF->nRecordLength) {
      nTupleLen = psDBF->nRecordLength;
      pReturnTuple = (char *) SfRealloc(pReturnTuple, psDBF->nRecordLength);
    }
    
    memcpy ( pReturnTuple, pabyRec, psDBF->nRecordLength );
        
    return( pReturnTuple );
}

/************************************************************************/
/*                          DBFCloneEmpty()                              */
/*                                                                      */
/*      Read one of the attribute fields of a record.                   */
/************************************************************************/

DBFHandle SHPAPI_CALL
DBFCloneEmpty(DBFHandle psDBF, const char * pszFilename ) 
{
    DBFHandle	newDBF;

   newDBF = DBFCreate ( pszFilename );
   if ( newDBF == NULL ) return ( NULL ); 
   
   newDBF->pszHeader = (char *) malloc ( 32 * psDBF->nFields );
   memcpy ( newDBF->pszHeader, psDBF->pszHeader, 32 * psDBF->nFields );
   
   newDBF->nFields = psDBF->nFields;
   newDBF->nRecordLength = psDBF->nRecordLength;
   newDBF->nHeaderLength = 32 * (psDBF->nFields+1);
    
   newDBF->panFieldOffset = (int *) malloc ( sizeof(int) * psDBF->nFields ); 
   memcpy ( newDBF->panFieldOffset, psDBF->panFieldOffset, sizeof(int) * psDBF->nFields );
   newDBF->panFieldSize = (int *) malloc ( sizeof(int) * psDBF->nFields );
   memcpy ( newDBF->panFieldSize, psDBF->panFieldSize, sizeof(int) * psDBF->nFields );
   newDBF->panFieldDecimals = (int *) malloc ( sizeof(int) * psDBF->nFields );
   memcpy ( newDBF->panFieldDecimals, psDBF->panFieldDecimals, sizeof(int) * psDBF->nFields );
   newDBF->pachFieldType = (char *) malloc ( sizeof(int) * psDBF->nFields );
   memcpy ( newDBF->pachFieldType, psDBF->pachFieldType, sizeof(int) * psDBF->nFields );

   newDBF->bNoHeader = TRUE;
   newDBF->bUpdated = TRUE;
   
   DBFWriteHeader ( newDBF );
   DBFClose ( newDBF );
   
   newDBF = DBFOpen ( pszFilename, "rb+" );

   return ( newDBF );
}

/************************************************************************/
/*                       DBFGetNativeFieldType()                        */
/*                                                                      */
/*      Return the DBase field type for the specified field.            */
/*                                                                      */
/*      Value can be one of: 'C' (String), 'D' (Date), 'F' (Float),     */
/*                           'N' (Numeric, with or without decimal),    */
/*                           'L' (Logical),                             */
/*                           'M' (Memo: 10 digits .DBT block ptr)       */
/************************************************************************/

char SHPAPI_CALL
DBFGetNativeFieldType( DBFHandle psDBF, int iField )

{
    if( iField >=0 && iField < psDBF->nFields )
        return psDBF->pachFieldType[iField];

    return  ' ';
}

/************************************************************************/
/*                            str_to_upper()                            */
/************************************************************************/

static void str_to_upper (char *string)
{
    int len;
    short i = -1;

    len = strlen (string);

    while (++i < len)
        if (isalpha(string[i]) && islower(string[i]))
            string[i] = toupper ((int)string[i]);
}

/************************************************************************/
/*                          DBFGetFieldIndex()                          */
/*                                                                      */
/*      Get the index number for a field in a .dbf file.                */
/*                                                                      */
/*      Contributed by Jim Matthews.                                    */
/************************************************************************/

int SHPAPI_CALL
DBFGetFieldIndex(DBFHandle psDBF, const char *pszFieldName)

{
    char          name[12], name1[12], name2[12];
    int           i;

    strncpy(name1, pszFieldName,11);
    name1[11] = '\0';
    str_to_upper(name1);

    for( i = 0; i < DBFGetFieldCount(psDBF); i++ )
    {
        DBFGetFieldInfo( psDBF, i, name, NULL, NULL );
        strncpy(name2,name,11);
        str_to_upper(name2);

        if(!strncmp(name1,name2,10))
            return(i);
    }
    return(-1);
}


















