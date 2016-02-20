#GUI for pySIM - Framework for spatial interaction modelling

from Tkinter import *
import ttk
import Tix
import tkFileDialog
import tkMessageBox
#import os
import sys
import FileIO
import numpy as np
import csv
import shapely
from math import sqrt, exp
from scipy import stats
from scipy.stats.mstats import mquantiles
from copy import copy
from datetime import datetime

FILE_TYPE = {"csv":0, "dbf":1, "txt":2, "shp":3, "ctl":4}
OPT_CRITERIA = {0: "AICc", 1: "AIC", 2: "BIC", 3: "CV"}
OPTION = {0:"OFF",1:"ON"}
DIST_TYPE = {0: "Euclidean distance", 1: "Spherical distance"}

class Info(Toplevel):
    """
    output window
    """
    def __init__(self, master=None,summary=''):
        Toplevel.__init__(self, master)
        self.transient(master)
        #Frame.__init__(self, master)
        #self.pack()

        # make window stretchable
        top=self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.title("Summary")
        self.txt_summary = ttk.Tkinter.Text(self)
        #self.txt_summary["width"] = 100
        #self.txt_summary["scrollregion"]=(0, 0, 1200, 800)
        self.txt_summary.grid(row=0,column=0,sticky=N+S+E+W)
        self.txt_summary.insert(END,summary)
        #self.txt_summary = ttk.Label(self)
        #self.txt_summary["width"] = 200
        #self.txt_summary.grid(row=0,column=0)
        #self.txt_summary["text"] = summary

        self.scrollY = ttk.Scrollbar(self, orient=VERTICAL, command=self.txt_summary.yview )
        self.scrollY.grid(row=0, column=1, sticky=N+S )
        self.scrollX = ttk.Scrollbar ( self, orient=HORIZONTAL, command=self.txt_summary.xview )
        self.scrollX.grid(row=1, column=0, sticky=E+W )

        self.txt_summary["xscrollcommand"] = self.scrollX.set
        self.txt_summary["yscrollcommand"] = self.scrollY.set

class mainGUI(Frame):
    """
    GUI to implement GWR functions

    """
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()

        self.createWidget()

        #self.openInfo = False # output window

    def reset(self):
	"""
	reset GUI to default settings
	"""
	self.txt_open.delete(0, END)
	self.txt_id.delete(0,END)
	self.txt_xcoord.delete(0,END)
	self.txt_ycoord.delete(0,END)

	self.var_disttype.set(0)

	self.cmb_wtype.current(0)
	self.cmb_bdsel.current(0)
	self.txt_bdval.delete(0,END)
	self.txt_bdmax.delete(0,END)
	self.txt_bdmin.delete(0,END)
	self.txt_bdstep.delete(0,END)

	self.var_modtype.set(0)
	self.var_varstd.set(0)
	self.var_vartest.set(0)
	self.var_l2g.set(0)
	self.var_g2l.set(0)
	self.var_optcri.set(0)

	self.txt_y.delete(0,END)
	self.txt_yoff.delete(0,END)
	self.txt_yoff["state"] = DISABLED
	self.lstb_xloc.delete(0,END)
	self.lstb_xloc.insert(0,"000 Intercept")
	self.lstb_xglob.delete(0,END)
	self.lstb_allvars.delete(0,END)

	self.txt_flectrl.delete(0,END)
	self.txt_flesum.delete(0,END)
	self.txt_flebeta.delete(0,END)

	self.var_pred.set(0)

	self.txt_flepred.delete(0,END)
	self.txt_flepred["state"] = DISABLED
	self.txt_flepredout.delete(0,END)
	self.txt_flepredout["state"] = DISABLED


    def openFile(self,target):
        """
        open file dialog for input data or prediction data
        target: 0--input data for regression; 1--input data for prediction
        """
        options = {}
        if target == 0: # regression data file
            options["filetypes"] = [("CSV", "*.csv"), ("DBF", "*.dbf"), ("TEXT", "*.txt"), ("SHP", "*.shp"), ("CONTROL", "*.ctl")]
            options["title"] = "Open File"
	    options['defaultextension'] = '.txt'
            name_openfle = tkFileDialog.askopenfilename(**options)
        if target == 1: # prediction data file
	    options["filetypes"] = [("CSV", "*.csv"), ("DBF", "*.dbf"), ("TEXT", "*.txt"), ("SHP", "*.shp")]
	    options['defaultextension'] = '.txt'
            options["title"] = "Open File"
            name_openfle = tkFileDialog.askopenfilename(**options)
        if target == 2: # control file
	    options["filetypes"] = [("CONTROL", "*.ctl")]
	    options['defaultextension'] = '.ctl'
            options["title"] = "Save File"
            name_openfle = tkFileDialog.asksaveasfilename(**options)
	if target == 3:  # summary file
            options["filetypes"] = [("TEXT", "*.txt")]
	    options['defaultextension'] = '.txt'
            options["title"] = "Save File"
            name_openfle = tkFileDialog.asksaveasfilename(**options)
        if target == 4 or target == 5: # local estimates and prediction file
            options["filetypes"] = [("CSV", "*.csv"), ("DBF", "*.dbf"), ("TEXT", "*.txt")]
	    options['defaultextension'] = '.csv'
            options["title"] = "Save File"
            name_openfle = tkFileDialog.asksaveasfilename(**options)
        #print name_openfle

        # openfile on own code: input data for regression model
        if name_openfle and target == 0:  #read_FILE = {0: read_CSV, 1: read_DBF, 2: read_TXT, 3: read_SHP, }
	    fType = FILE_TYPE[name_openfle[-3:]]
	    if fType > 3: # open control file
		self.fillGUI(name_openfle)
	    else:
		self.reset()
		self.txt_open.insert(0,name_openfle) # show file name
		self.flepath_open = name_openfle

	    self.fleType = FILE_TYPE[self.flepath_open[-3:]]
	    allData = FileIO.read_FILE[self.fleType](self.flepath_open)
	    if self.fleType == 3: # shapefile
		self.coords = allData[0]
		self.lstFlds = allData[1]
		self.dicData = allData[2]
	    else:
		self.lstFlds = allData[0]
		self.dicData = allData[1]

	    # fill the fields list
	    if fType <= 3:
		#self.lstb_allvars.delete(0,END)
		nflds = len(self.lstFlds)
		for i in range(nflds):
		    if i < 9:
			id_fld = '00' + str(i+1)
		    else:
			if i < 99:
			    id_fld = '0' + str(i+1)
			else:
			    id_fld = str(i+1)
		    self.lstb_allvars.insert(i,id_fld+' '+self.lstFlds[i])

        # openfile on own code: input data for prediction
        if name_openfle and target == 1:  #read_FILE = {0: read_CSV, 1: read_DBF, 2: read_SHP, 3: read_TXT}
	    self.txt_flepred.delete(0,END)
            self.txt_flepred.insert(0,name_openfle) # show file name
            #fleType = FILE_TYPE[name_openfle[-3:]]
            #allData = FileIO.read_FILE[fleType](name_openfle)
            #if fleType == 3: # shapefile
                #self.coords_pred = allData[0]
            #else:
                #self.coords_pred = allData[1]
	    # insert default names for local estimate file
	    de_flepredout = name_openfle[:-3] + 'csv'
	    self.txt_flepredout.delete(0,END)
            self.txt_flepredout.insert(0,de_flepredout)

        # get name for control file
        if name_openfle and target == 2:
	    self.txt_flectrl.delete(0,END)
            self.txt_flectrl.insert(0,name_openfle)
	    # insert default names for summary file and local estimate file
	    de_flesum = name_openfle[:-4] + '_summary.txt'
	    self.txt_flesum.delete(0,END)
            self.txt_flesum.insert(0,de_flesum)
	    de_flebeta = name_openfle[:-4] + '_listwise.csv'
	    self.txt_flebeta.delete(0,END)
            self.txt_flebeta.insert(0,de_flebeta)

        # get name for summary file
        if name_openfle and target == 3:
	    self.txt_flesum.delete(0,END)
            self.txt_flesum.insert(0,name_openfle)

        # get name for local estimates file
        if name_openfle and target == 4:
	    self.txt_flebeta.delete(0,END)
            self.txt_flebeta.insert(0,name_openfle)

        # get name for local estimates file for prediction
        if name_openfle and target == 5:
	    self.txt_flepredout.delete(0,END)
            self.txt_flepredout.insert(0,name_openfle)

    def addVars(self,txt_obj=None,lstb_obj=None):
        """
        add variables
        """
        #self.txt_id.insert(0,self.lstb_allvars.get[self.lstb_allvars.curselection()])
        if not txt_obj is None and self.lstb_allvars.curselection():
            curr_id = self.lstb_allvars.curselection()
            curr_var = self.lstb_allvars.get(curr_id)
	    if txt_obj.get(): # return the variable to the all variable list
		pre_var = txt_obj.get()
		self.lstb_allvars.insert(END,pre_var)
	    txt_obj.delete(0,END)
            txt_obj.insert(0,curr_var)#
            self.lstb_allvars.delete(curr_id)

        if not lstb_obj is None and self.lstb_allvars.curselection():
            curr_id = self.lstb_allvars.curselection()
            curr_var = self.lstb_allvars.get(curr_id)
            lstb_obj.insert(END,curr_var)
            self.lstb_allvars.delete(curr_id)

    def outVars(self,txt_obj=None,lstb_obj=None):
        """
        remove variables
        """
        if not lstb_obj is None and lstb_obj.curselection():
            curr_id = lstb_obj.curselection()
            curr_fld = lstb_obj.get(curr_id)
            lstb_obj.delete(curr_id)
            self.lstb_allvars.insert(END,curr_fld)#int(curr_fld[:3])

        if not txt_obj is None and txt_obj.get():
            curr_fld = txt_obj.get()
            txt_obj.delete(0,len(curr_fld))
            self.lstb_allvars.insert(END,curr_fld)# int(curr_fld[:3])

    def set_pred(self):
        """
        set prediction option
        """
        if self.var_pred.get() == 0:
            self.txt_flepred["state"] = DISABLED
            self.txt_flepredout["state"] = DISABLED
            self.btn_flepred["state"] = DISABLED
            self.btn_flepredout["state"] = DISABLED
        else:
            self.txt_flepred["state"] = NORMAL
            self.txt_flepredout["state"] = NORMAL
            self.btn_flepred["state"] = NORMAL
            self.btn_flepredout["state"] = NORMAL

    def set_modtype(self):
        """
        set model type
        """
        val = self.var_modtype.get()
        if val == 1:
            self.txt_yoff["state"] = NORMAL
        else:
            self.txt_yoff["state"] = DISABLED

    def set_bw(self):
        """
        set bandwidth method
        """
        curval = self.cmb_bdsel.current()
        if curval == 0: # golden section search
            self.txt_bdmax["state"] = NORMAL
            self.txt_bdmin["state"] = NORMAL
            self.txt_bdstep["state"] = DISABLED
            self.txt_bdval["state"] = DISABLED
        else:
            if curval == 1: #interval search
                self.txt_bdmax["state"] = NORMAL
                self.txt_bdmin["state"] = NORMAL
                self.txt_bdstep["state"] = NORMAL
                self.txt_bdval["state"] = DISABLED
            else:
                self.txt_bdmax["state"] = DISABLED
                self.txt_bdmin["state"] = DISABLED
                self.txt_bdstep["state"] = DISABLED
                self.txt_bdval["state"] = NORMAL

    def checkVars(self):
        """
        check validity of model setting before run
        """
        # 0 check open file
        if not self.txt_open.get():
            tkMessageBox.showwarning("Warning", "Please open a data file!")

        # 1 check x,y coords
        if self.fleType <> 3: # shapefile, already have coords
            name_xcoord = self.txt_xcoord.get()
            name_ycoord = self.txt_ycoord.get()
            if not name_xcoord or not name_ycoord:
                tkMessageBox.showwarning("Warning", "Please input x/y coordinate variable!")

        # 2 check y and y_off settings
        if not self.txt_y.get():
            tkMessageBox.showwarning("Warning", "Please input y variable!")

        if not self.txt_yoff.get() and self.var_modtype == 1:
            tkMessageBox.showwarning("Warning", "Please input y offset variable!")

        # 3 check x settings
        if not self.lstb_xglob.get(0) and not self.lstb_xloc.get(0):
            tkMessageBox.showwarning("Warning", "Please input x variables!")

        if not self.lstb_xglob.get(0) and self.var_g2l == 1:
            tkMessageBox.showwarning("Warning", "Please input global x variables!")

        if not self.lstb_xloc.get(0): #and self.var_l2g == 1
            tkMessageBox.showwarning("Warning", "Please input local x variables!")

        # 4 check bandwidth setting
	if self.cmb_bdsel.current() == 2: # single value
	    if not self.txt_bdval.get():
		tkMessageBox.showwarning("Warning", "Please input a bandwidth value!")
        if self.cmb_bdsel.current() == 1: # intervel search
            if not self.txt_bdmax.get():
                tkMessageBox.showwarning("Warning", "Please input maximum value for bandwidth searching!")
            if not self.txt_bdmin.get():
                tkMessageBox.showwarning("Warning", "Please input minimum value for bandwidth searching!")
            if not self.txt_bdstep.get():
                tkMessageBox.showwarning("Warning", "Please input the interval for bandwidth searching!")

        # 5 check output files
        if not self.txt_flectrl.get():
            tkMessageBox.showwarning("Warning", "Please input the control file name!")

        if not self.txt_flesum.get():
            tkMessageBox.showwarning("Warning", "Please input the summary file name!")

        if not self.txt_flebeta.get():
            tkMessageBox.showwarning("Warning", "Please input the local estimates file name!")

        # 6 check prediction files
        if self.var_pred == 1:
            if not self.txt_flepred.get():
                tkMessageBox.showwarning("Warning", "Please input the data file for prediction!")
            if not self.txt_flepredout.get():
                tkMessageBox.showwarning("Warning", "Please input the output file for prediction!")

    def saveFle_ctrl(self,fleName, dic):
	"""
	save control file
	Arguments:
	    fleName  : string
	               path of control file (.txt)
	    dic      : dictionary,
	               key: item, value: model settings
	"""
	with open(fleName, 'w') as txtfile:
	    # 1 input file
	    txtfile.write('Data file:\n') #line 1
	    txtfile.write(''.join([dic["flepath_open"],'\n'])) # line 2

	    # 2 variable names
	    txtfile.write(': '.join(['Name of ID',self.txt_id.get()])) # line 3
	    txtfile.write('\n')
	    txtfile.write(': '.join(['Name of X Coord',self.txt_xcoord.get()])) # line 4
	    txtfile.write('\n')
	    txtfile.write(': '.join(['Name of Y Coord',self.txt_ycoord.get()])) # line 5
	    txtfile.write('\n')

	    # 3 distance type
	    txtfile.write(': '.join(['Type of distance',str(dic["disttype"])])) # line 6
	    txtfile.write('\n')

	    # 4 kernel type
	    txtfile.write(': '.join(['Type of kernel',str(dic["weittype"])])) # line 7
	    txtfile.write('\n')

	    # 5 bandwidh selection method
	    txtfile.write(': '.join(['Bandwidth selection method',str(dic["bdsel"])])) # line 8
	    txtfile.write('\n')
	    txtfile.write(': '.join(['Single bandwidth',str(dic["bd_val"])])) # line 9
	    txtfile.write('\n')
	    txtfile.write(': '.join(['Maximum bandwidth',str(dic["bd_max"])])) # line 10
	    txtfile.write('\n')
	    txtfile.write(': '.join(['Minimum bandwidth',str(dic["bd_min"])])) # line 11
	    txtfile.write('\n')
	    txtfile.write(': '.join(['Bandwidth stepsize',str(dic["bd_step"])])) # line 12
	    txtfile.write('\n')

	    # 6 model type
	    txtfile.write(': '.join(['Model type',str(dic["modtype"])])) # line 13
	    txtfile.write('\n')

	    # 7 advance model options
	    txtfile.write(': '.join(['Variable standardization',str(dic["varstd"])])) # line 14
	    txtfile.write('\n')
	    txtfile.write(': '.join(['Variability test',str(dic["vartest"])])) # line 15
	    txtfile.write('\n')
	    txtfile.write(': '.join(['L->G test',str(dic["l2g"])])) # line 16
	    txtfile.write('\n')
	    txtfile.write(': '.join(['G->L test',str(dic["g2l"])])) # line 17
	    txtfile.write('\n')

	    # 8 optimization criterion
	    txtfile.write(': '.join(['Optimization criterion',str(dic["optcri"])])) # line 18
	    txtfile.write('\n')

	    # 9 variable names
	    # y
	    txtfile.write(': '.join(['Dependent variable',self.txt_y.get()])) # line 19
	    txtfile.write('\n')

	    # yoff
	    txtfile.write(': '.join(['Offset variable',self.txt_yoff.get()])) # line 20
	    txtfile.write('\n')

	    # x local
	    txtfile.write(':'.join(['Independent variables (local)',str(self.n_xloc)])) # line 21
	    txtfile.write('\n')
	    for i in range(self.n_xloc):
		txtfile.write(''.join([self.lstb_xloc.get(i),'\n']))

	    # x global
	    txtfile.write(':'.join(['Independent variables (global)',str(self.n_xglob)])) # line 22
	    txtfile.write('\n')
	    for i in range(self.n_xglob):
		txtfile.write(''.join([self.lstb_xglob.get(i),'\n']))

	    # x unused
	    nflds_unused = self.lstb_allvars.size()
	    txtfile.write(':'.join(['Unused variables',str(nflds_unused)])) # line 23
	    txtfile.write('\n')
	    for i in range(nflds_unused):
		txtfile.write(''.join([self.lstb_allvars.get(i),'\n']))

	    # 10 output file
	    # control file
	    txtfile.write('Path of control file:\n') # line 24
	    txtfile.write(''.join([dic["flepath_ctrl"],'\n']))

	    # summary file
	    txtfile.write('Path of summary file:\n') # line 25
	    txtfile.write(''.join([dic["flepath_sum"],'\n']))

	    # listwise file
	    txtfile.write('Path of local estimates file:\n') # line 26
	    txtfile.write(''.join([dic["flepath_beta"],'\n']))

	    # 11 prediction
	    txtfile.write(':'.join(['Prediction for non-regression points',str(dic["pred"])])) # line 27
	    txtfile.write('\n')

	    txtfile.write('Path of prediction data file:\n') # line 28
	    txtfile.write(''.join([dic["flepath_pred"],'\n']))

	    txtfile.write('Path of prediction output file:\n') # line 28
	    txtfile.write(''.join([dic["flepath_predout"],'\n']))

	    txtfile.close()

    def saveFle_beta(self, GWRMod, flepath, name_id='', val_id=None):
	"""
	save local estimates into a .txt file

	Arguments:
	    GWRMod      : GWR model
	    name_id     : string
			  name of id
	    flepath     : string
	                  file path
	"""
	# 1 header information
	headers = ['Area_num']
	if name_id <> '':
	    headers.append('Area_key')
	headers.append('x_coord')
	headers.append('y_coord')

	# local x
	if hasattr(GWRMod, 'x_loc'):
	    nxloc = GWRMod.nVars_loc
	    names_x = GWRMod.x_name_loc
	else:
	    nxloc = GWRMod.nVars
	    names_x = GWRMod.x_name
	for i in range(nxloc):
	    headers.append('est_' + names_x[i])
	    headers.append('se_' + names_x[i])
	    headers.append('t_' + names_x[i])

	# y and statistics
	headers.append('y')
	headers.append('y_hat')
	headers.append('residual')
	if GWRMod.mType == 0: # Gaussian model
	    headers.append('std_residual')
	    headers.append('localR2')
	    headers.append('influence')
	    headers.append('CooksD')
	else:
	    headers.append('localpdev')

	# 2 add values
	lstBeta = []
	nrec = GWRMod.nObs
	for i in range(nrec):
	    lstBeta.append([])
	    lstBeta[i].append(i)
	    if name_id <> '':
		lstBeta[i].append(val_id[i])
	    lstBeta[i].append(round(self.coords[i][0],6))
	    lstBeta[i].append(round(self.coords[i][1],6))
	    for j in range(nxloc):
		if hasattr(GWRMod, 'x_loc'):
		    lstBeta[i].append(round(GWRMod.Betas_loc[i][j],6))
		    lstBeta[i].append(round(GWRMod.std_err_loc[i][j],6))
		    lstBeta[i].append(round(GWRMod.t_stat_loc[i][j],6))
		else:
		    lstBeta[i].append(round(GWRMod.Betas[i][j],6))
		    lstBeta[i].append(round(GWRMod.std_err[i][j],6))
		    lstBeta[i].append(round(GWRMod.t_stat[i][j],6))
	    lstBeta[i].append(GWRMod.y[i][0])
	    lstBeta[i].append(round(GWRMod.y_pred[i],6))
	    lstBeta[i].append(round(GWRMod.res[i],6))
	    if GWRMod.mType == 0:
		lstBeta[i].append(round(GWRMod.std_res[i],6))
		lstBeta[i].append(round(GWRMod.localR2[i],6))
		lstBeta[i].append(round(GWRMod.influ[i],6))
		lstBeta[i].append(round(GWRMod.CooksD[i],6))
	    else:
		lstBeta[i].append(round(GWRMod.localpDev[i],6))

	# 3 write file
	fleType = FILE_TYPE[flepath[-3:]]
	FileIO.write_FILE[fleType](flepath,headers,lstBeta)

    def saveFle_pred(self, GWRMod, betas, stdErrs, tstats, localR2, flepath):
	"""
	save local estiamtes for prediction data
	Arguments:
	    GWRMod   : GWR model
	    betas    : array
	               local estimates
	    stdErrs  : array
	               std error of beta
	    tstats   : array
	               t statistics
	    localR2  : array
	               local R2 or deviation
	    flepath  : string
		       file path

	"""
	nrec = len(self.coords_pred.keys())
	headers = ['ID']
	headers.append('X')
	headers.append('Y')

	# local x
	if hasattr(GWRMod, 'x_loc'):
	    nxloc = GWRMod.nVars_loc
	    names_x = GWRMod.x_name_loc
	else:
	    nxloc = GWRMod.nVars
	    names_x = GWRMod.x_name
	for i in range(nxloc):
	    headers.append('est_' + names_x[i])
	    headers.append('se_' + names_x[i])
	    headers.append('t_' + names_x[i])

	if GWRMod.mType == 0:
	    headers.append('localR2')
	else:
	    headers.append('localpdev')

	# 2 add values
	lstBeta = []
	for i in range(nrec):
	    lstBeta.append([])
	    lstBeta[i].append(i)
	    lstBeta[i].append(round(self.coords_pred[i][0],6))
	    lstBeta[i].append(round(self.coords_pred[i][1],6))
	    for j in range(nxloc):
		lstBeta[i].append(round(betas[i][j],6))
		lstBeta[i].append(round(stdErrs[i][j],6))
		lstBeta[i].append(round(tstats[i][j],6))
	    lstBeta[i].append(round(localR2[i],6))

	# 3 write file
	fleType = FILE_TYPE[flepath[-3:]]
	FileIO.write_FILE[fleType](flepath,headers,lstBeta)


    def fillGUI(self, flepath):
	"""
	fill GUI using control file
	"""
	with open(flepath, 'rb') as txtfile:

	    # 1 input file
	    txtfile.readline()
	    self.txt_open.delete(0,END) # clear the text first
	    self.flepath_open = txtfile.readline().strip()
	    self.txt_open.insert(0,self.flepath_open)

	    # 2 variable names
	    name_id = txtfile.readline().strip().split(':')[1].strip()
	    self.txt_id.delete(0,END)
	    self.txt_id.insert(0,name_id)
	    name_xcoord = txtfile.readline().strip().split(':')[1].strip()
	    self.txt_xcoord.delete(0,END)
	    self.txt_xcoord.insert(0,name_xcoord)
	    name_ycoord = txtfile.readline().strip().split(':')[1].strip()
	    self.txt_ycoord.delete(0,END)
	    self.txt_ycoord.insert(0,name_ycoord)

	    # 3 distance type
	    type_dist = int(txtfile.readline().strip().split(':')[1])
	    self.var_disttype.set(type_dist)

	    # 4 kernel type
	    type_kernel = int(txtfile.readline().strip().split(':')[1])
	    self.cmb_wtype.current(type_kernel)

	    # 5 bandwidh selection method
	    type_bdsel = int(txtfile.readline().strip().split(':')[1])
	    self.cmb_bdsel.current(type_bdsel)

	    bd_val = txtfile.readline().strip().split(':')[1]
	    self.txt_bdval.delete(0,END)
	    if bd_val <> '':
		self.txt_bdval.insert(0,float(bd_val))

	    bd_max = txtfile.readline().strip().split(':')[1]
	    self.txt_bdmax.delete(0,END)
	    if bd_max <> '':
		self.txt_bdmax.insert(0,float(bd_max))

	    bd_min = txtfile.readline().strip().split(':')[1]
	    self.txt_bdmin.delete(0,END)
	    if bd_min <> '':
		self.txt_bdmin.insert(0,float(bd_min))

	    bd_step = txtfile.readline().strip().split(':')[1]
	    self.txt_bdstep.delete(0,END)
	    if bd_step <> '':
		self.txt_bdstep.insert(0,float(bd_step))

	    # 6 model type
	    type_mod = int(txtfile.readline().strip().split(':')[1])
	    self.var_modtype.set(type_mod)

	    # 7 advance model options
	    varstd = int(txtfile.readline().strip().split(':')[1])
	    self.var_varstd.set(varstd)

	    vartest = int(txtfile.readline().strip().split(':')[1])
	    self.var_vartest.set(vartest)

	    l2g = int(txtfile.readline().strip().split(':')[1])
	    self.var_l2g.set(l2g)

	    g2l = int(txtfile.readline().strip().split(':')[1])
	    self.var_g2l.set(g2l)

	    # 8 optimization criterion
	    optcri = int(txtfile.readline().strip().split(':')[1])
	    self.var_optcri.set(optcri)

	    # 9 variable names
	    # y
	    name_y = txtfile.readline().strip().split(':')[1].strip()
	    self.txt_y.delete(0,END)
	    self.txt_y.insert(0,name_y)

	    # yoff
	    name_yoff = txtfile.readline().strip().split(':')[1].strip()
	    if name_yoff <> '':
		self.txt_yoff["state"] = NORMAL
		self.txt_yoff.delete(0,END)
		self.txt_yoff.insert(0,name_yoff)

	    # x local
	    n_xloc = int(txtfile.readline().strip().split(':')[1])
	    self.lstb_xloc.delete(0,END)
	    for i in range(n_xloc):
		self.lstb_xloc.insert(i,txtfile.readline().strip())

	    # x global
	    n_xglob = int(txtfile.readline().strip().split(':')[1])
	    self.lstb_xglob.delete(0,END)
	    for i in range(n_xglob):
		self.lstb_xglob.insert(i,txtfile.readline().strip())

	    # x unused
	    nflds_unused = int(txtfile.readline().strip().split(':')[1])
	    self.lstb_allvars.delete(0,END)
	    for i in range(nflds_unused):
		self.lstb_allvars.insert(i,txtfile.readline().strip())

	    # 10 output file
	    # control file
	    txtfile.readline()
	    self.txt_flectrl.delete(0,END)
	    self.txt_flectrl.insert(0,txtfile.readline().strip())

	    # summary file
	    txtfile.readline()
	    self.txt_flesum.delete(0,END)
	    self.txt_flesum.insert(0,txtfile.readline().strip())

	    # listwise file
	    txtfile.readline()
	    self.txt_flebeta.delete(0,END)
	    self.txt_flebeta.insert(0,txtfile.readline().strip())

	    # 11 prediction
	    pred = int(txtfile.readline().strip().split(':')[1])
	    self.var_pred.set(pred)

	    self.txt_flepred.delete(0,END)
	    self.txt_flepredout.delete(0,END)

	    if pred == 1:
		txtfile.readline()
		self.txt_flepred.insert(0,txtfile.readline().strip())

		txtfile.readline()
		self.txt_flepredout.insert(0,txtfile.readline().strip())

	txtfile.close()

    def run_mod(self,ref):
        """
        run model
        """
        # 1 check model setting
        self.checkVars()

        #---------------------------------------setting information, stored in control file--------------------------------

        dic_ctrl = {}

	# open data file
	if self.flepath_open <> self.txt_open.get():
	    # read data file
	    self.flepath_open = self.txt_open.get().strip()
	    self.fleType = FILE_TYPE[self.flepath_open[-3:]]
	    allData = FileIO.read_FILE[self.fleType](self.flepath_open)
	    if self.fleType == 3: # shapefile
		self.coords = allData[0]
		self.lstFlds = allData[1]
		self.dicData = allData[2]
	    else:
		self.lstFlds = allData[0]
		self.dicData = allData[1]

        dic_ctrl["flepath_open"] = self.flepath_open
        dic_ctrl["name_id"] = self.txt_id.get()[4:].strip()
        dic_ctrl["name_xcoord"] = self.txt_xcoord.get()[4:].strip()
        dic_ctrl["name_ycoord"] = self.txt_ycoord.get()[4:].strip()

        # 2 get distance type
        dic_ctrl["disttype"] = self.var_disttype.get()

        # 3 get kernel type
        dic_ctrl["weittype"] = self.cmb_wtype.current()

        # 4 get bandwidth selection method
        dic_ctrl["bdsel"] = self.cmb_bdsel.current()
        dic_ctrl["bd_val"] = self.txt_bdval.get().strip()
        dic_ctrl["bd_max"] = self.txt_bdmax.get().strip()
        dic_ctrl["bd_min"] = self.txt_bdmin.get().strip()
        dic_ctrl["bd_step"] = self.txt_bdstep.get().strip()

        # 5 get model type
        dic_ctrl["modtype"] = self.var_modtype.get()

        # 6 get advance model options
        dic_ctrl["varstd"] = self.var_varstd.get()
        dic_ctrl["vartest"] = self.var_vartest.get()
        dic_ctrl["l2g"] = self.var_l2g.get()
        dic_ctrl["g2l"] = self.var_g2l.get()

        # 7 get optimization criterion
        dic_ctrl["optcri"] = self.var_optcri.get()

        # 8 get coords, x,y, yoffset variables
        dic_ctrl["name_y"] = self.txt_y.get()[4:].strip()
        dic_ctrl["name_yoff"] = self.txt_yoff.get()[4:].strip()
        dic_ctrl["name_xloc"] = []
        dic_ctrl["name_xglob"] = []

        self.n_xloc = self.lstb_xloc.size()
        self.n_xglob = self.lstb_xglob.size()
        for i in range(self.n_xloc):
            dic_ctrl["name_xloc"].append(self.lstb_xloc.get(i)[4:].strip())
        for i in range(self.n_xglob):
            dic_ctrl["name_xglob"].append(self.lstb_xglob.get(i)[4:].strip())

        # 9 get output file
        dic_ctrl["flepath_ctrl"] = self.txt_flectrl.get().strip()
        dic_ctrl["flepath_beta"] = self.txt_flebeta.get().strip()
        dic_ctrl["flepath_sum"] = self.txt_flesum.get().strip()

        # 10 get prediction file
        dic_ctrl["pred"] = self.var_pred.get()
        dic_ctrl["flepath_pred"] = self.txt_flepred.get().strip()
        dic_ctrl["flepath_predout"] = self.txt_flepredout.get().strip()

	# get prediction data
	if dic_ctrl["pred"] == 1:
	    flepred = self.txt_flepred.get().strip()
	    fleType_pred = FILE_TYPE[flepred[-3:]]
	    allData_pred = FileIO.read_FILE[fleType_pred](flepred)
	    if fleType_pred == 3: # shapefile
		self.coords_pred = allData_pred[0]
	    else:
		self.coords_pred = allData_pred[1]

        #------------------------------------------run model--------------------------------------------------------
        begin_t = datetime.now()

	# 1 get coords and x,y data
        flds = []
        if self.fleType <> 3:
            flds.append(dic_ctrl["name_xcoord"])
            flds.append(dic_ctrl["name_ycoord"])
            self.coords = FileIO.get_subset(self.lstFlds,self.dicData,flds)
	    if self.fleType == 0 or self.fleType == 2: # .csv or .txt should change from str to float
		# reformat data to float
		for key, val in self.coords.items():
		    self.coords[key] = tuple(float(elem) for elem in val)

	# 0 get id data
	if dic_ctrl["name_id"] <> '':
	    values_id = FileIO.get_subset(self.lstFlds,self.dicData,[dic_ctrl["name_id"]])
	    if len(set(values_id.items())) < len(self.coords.keys()):
		tkMessageBox.showwarning("Warning", "The ID values must be unique!")
	else:
	    values_id = None



        # 2 get y
        data_y = FileIO.get_subset(self.lstFlds,self.dicData,[dic_ctrl["name_y"]])
        lst_data = []
	nObs = len(data_y.keys())
	if self.fleType == 0 or self.fleType == 2: # .csv or .txt should change from str to float
	    # reformat data to float
	    for key, val in data_y.items():
		data_y[key] = tuple(float(elem) for elem in val)
        for i in range(nObs):
	    lst_data.append(data_y[i])
        y = np.reshape(np.array(lst_data),(-1,1))

        # 3 get y_off
        yoff = None
        if self.var_modtype.get() == 1:
            data_yoff = FileIO.get_subset(self.lstFlds,self.dicData,[dic_ctrl["name_yoff"]])
            lst_data = []
	    if self.fleType == 0 or self.fleType == 2: # .csv or .txt should change from str to float
		# reformat data to float
		for key, val in data_yoff.items():
		    data_yoff[key] = tuple(float(elem) for elem in val)
            for i in range(nObs):
                lst_data.append(data_yoff[i])
            yoff = np.reshape(np.array(lst_data),(-1,1))

        # add intercept
        self.lstFlds.append('Intercept')
        for key, val in self.dicData.items():
            lst_val = list(val)
            lst_val.append(1)
            self.dicData[key] = tuple(lst_val)

        # 4 get x_loc
        if self.n_xloc == 0:
            xloc = None
        else:
            lst_data = []
            data_xloc = FileIO.get_subset(self.lstFlds,self.dicData,dic_ctrl["name_xloc"])
	    if self.fleType == 0 or self.fleType == 2: # .csv or .txt should change from str to float
		# reformat data to float
		for key, val in data_xloc.items():
		    data_xloc[key] = tuple(float(elem) for elem in val)
            for i in range(nObs):
                lst_data.append(data_xloc[i])
            xloc = np.array(lst_data)
            if dic_ctrl["varstd"]:
                xloc = M_Utilities.StandardVars(xloc)

        # 5 get x_glob
        if self.n_xglob == 0:
            xglob = None
        else:
            data_xglob = FileIO.get_subset(self.lstFlds,self.dicData,dic_ctrl["name_xglob"])
            lst_data = []
	    if self.fleType == 0 or self.fleType == 2: # .csv or .txt should change from str to float
		# reformat data to float
		for key, val in data_xglob.items():
		    data_xglob[key] = tuple(float(elem) for elem in val)
            for i in range(nObs):
                lst_data.append(data_xglob[i])
            xglob = np.array(lst_data)
            if dic_ctrl["varstd"]:
                xglob = M_Utilities.StandardVars(xglob)

        # 6 create kernel
	if self.txt_bdmax.get():
	    band_max = float(dic_ctrl["bd_max"])
	else:
	    band_max = 0.0
	if self.txt_bdmin.get():
	    band_min = float(dic_ctrl["bd_min"])
	else:
	    band_min = 0.0
	if self.txt_bdstep.get():
	    band_step = float(dic_ctrl["bd_step"])
	else:
	    band_step = 0.0
	if self.txt_bdval.get():
	    band = float(dic_ctrl["bd_val"])
	else:
	    band = 0.0
        if dic_ctrl["bdsel"] == 0 or dic_ctrl["bdsel"] == 1:#
            weit_sele = M_selection.Band_Sel(y,xglob,xloc,self.coords,dic_ctrl["modtype"],yoff,dic_ctrl["weittype"],
                                        dic_ctrl["optcri"],dic_ctrl["bdsel"],band_max,band_min,
                                        band_step,1e-6,200,dic_ctrl["disttype"])
            weit = weit_sele[1]
            weit_output = weit_sele[2]
        else: #single bandwidth
            weit = Kernel.GWR_W(self.coords, band, dic_ctrl["weittype"],None,dic_ctrl["disttype"])

        # 7 create model
        if self.n_xglob == 0: # local model
            gwrmod = M_GWGLM.GWGLM(y,xloc,weit,dic_ctrl["modtype"],yoff,False,1e-6,200,dic_ctrl["name_y"],
                                          dic_ctrl["name_yoff"],dic_ctrl["name_xloc"],self.flepath_open,True)
        else: # mixed model
            gwrmod = M_semiGWR.semiGWR(y,xglob,xloc,weit,dic_ctrl["modtype"],yoff,False,1e-6,200,dic_ctrl["name_y"],
                                          dic_ctrl["name_yoff"],dic_ctrl["name_xglob"],dic_ctrl["name_xloc"],self.flepath_open,True)

        # 8 Variability test or L->G or G->L test
	# 8.1 variability test
        if dic_ctrl["vartest"] == 1:
            rs_varytest = M_Utilities.varyTest(y, xglob, xloc, weit, dic_ctrl["modtype"], yoff, dic_ctrl["optcri"],gwrmod)

	# 8.2 L->G
	cri_l2g = 1e20
	if dic_ctrl["l2g"] == 1:
	    test_l2g = M_Utilities.L2G(y, xglob, xloc, self.coords, dic_ctrl["modtype"], dic_ctrl["weittype"], yoff, gwrmod, dic_ctrl["optcri"], dic_ctrl["bdsel"], band, band_max, band_min, band_step)
	    cri_l2g = test_l2g[4]

	# 8.3 G->L
	cri_g2l = 1e20
	if dic_ctrl["g2l"] == 1:
	    test_g2l = M_Utilities.G2L(y, xglob, xloc, self.coords, dic_ctrl["modtype"], dic_ctrl["weittype"], yoff, gwrmod, dic_ctrl["optcri"], dic_ctrl["bdsel"], band, band_max, band_min, band_step)
            cri_g2l = test_g2l[4]

	# 8.4 decide whether update the model
	new_mod = False
	if dic_ctrl["optcri"] == 0:
	    cri_origin = gwrmod.aicc
	if dic_ctrl["optcri"] == 1:
	    cri_origin = gwrmod.aic
	if dic_ctrl["optcri"] == 2:
	    cri_origin = gwrmod.bic
	if dic_ctrl["optcri"] == 3:
	    cri_origin = gwrmod.cv

	if cri_l2g < cri_g2l:
	    if cri_l2g < cri_origin: # change model to l2g: varsL, varsG, optband, optWeit, cri_old
		if len(test_l2g[1]) > 0: # some xloc -> xglob
		    new_mod = True
		    # get new x local
		    xloc_new = np.delete(xloc,test_l2g[1],1)
		    # get new x global
		    if self.n_xglob == 0:
			xglob_new = xloc[:,test_l2g[1]]
		    else:
			xglob_new = np.hstack((xglob,xloc[:,test_l2g[1]]))
		    # get new kernel
		    weit_new = test_l2g[3]
		    # get new x_glob names
		    for i in test_l2g[1]:
			dic_ctrl["name_xglob"].append(dic_ctrl["name_xloc"][i])
		    # get new x_loc names
		    for i in test_l2g[1]:
			dic_ctrl["name_xloc"].pop(i)
		    # get new model: should be mixed model
		    gwrmod_new = M_semiGWR.semiGWR(y,xglob_new,xloc_new,weit_new,dic_ctrl["modtype"],yoff,False,1e-6,200,dic_ctrl["name_y"],
                                          dic_ctrl["name_yoff"],dic_ctrl["name_xglob"],dic_ctrl["name_xloc"],self.flepath_open,True)
	else:
	    if cri_g2l < cri_origin: # change model to g2l: varsL, varsG, optband, optWeit, cri_old
		if len(test_g2l[0]) > 0: # some xglob -> xloc
		    new_mod = True
		    # get new x local
		    if self.n_xloc == 0:
			xloc_new = xglob[:,test_g2l[0]]
		    else:
			xloc_new = np.hstack((xloc,xglob[:,test_g2l[0]]))
		    # get new x global
		    xglob_new = np.delete(xglob,test_g2l[0],1)
		    # get new kernel
		    weit_new = test_g2l[3]
		    # get new x_loc names
		    for i in test_l2g[0]:
			dic_ctrl["name_xloc"].append(dic_ctrl["name_xglob"][i])
		    # get new x_glob names
		    for i in test_l2g[0]:
			dic_ctrl["name_xglob"].pop(i)
		    # get new model
		    if len(test_g2l[1]) == 0:
			gwrmod_new = M_GWGLM.GWGLM(y,xloc_new,weit_new,dic_ctrl["modtype"],yoff,False,1e-6,200,dic_ctrl["name_y"],
                                          dic_ctrl["name_yoff"],dic_ctrl["name_xloc"],self.flepath_open,True)
		    else:
			gwrmod_new = M_semiGWR.semiGWR(y,xglob_new,xloc_new,weit_new,dic_ctrl["modtype"],yoff,False,1e-6,200,dic_ctrl["name_y"],
                                          dic_ctrl["name_yoff"],dic_ctrl["name_xglob"],dic_ctrl["name_xloc"],self.flepath_open,True)

        # 9 prediction
	if dic_ctrl["pred"] == 1:
	    s2 = 1
	    yfix = None
	    fMat = None
	    if new_mod:
		if dic_ctrl["modtype"] == 0:
		    s2 = gwrmod_new.sigma2
		if dic_ctrl["l2g"] == 1:
		    nlen = len(test_l2g[1]) +  self.n_xglob
		else:
		    nlen = len(test_g2l[1])
		if nlen > 0:  # mixed model
		    yfix = np.dot(gwrmod_new.x_glob, gwrmod_new.Betas_glob)
		    fMat = gwrmod_new.m_glob.FMatrix
		rs_pred = M_Utilities.pred(self.coords_pred, self.coords, weit_new.band, y, xloc_new, gwrmod_new.y_pred, dic_ctrl["weittype"], dic_ctrl["modtype"], dic_ctrl["disttype"], yoff, s2, yfix, fMat)
	    else:
		if dic_ctrl["modtype"] == 0:
		    s2 = gwrmod.sigma2
		if self.n_xglob > 0:  # mixed model
		    yfix = np.dot(xglob, gwrmod.Betas_glob)
		    fMat = gwrmod.m_glob.FMatrix
		rs_pred = M_Utilities.pred(self.coords_pred, self.coords, weit.band, y, xloc, gwrmod.y_pred, dic_ctrl["weittype"], dic_ctrl["modtype"], dic_ctrl["disttype"], yoff, s2, yfix, fMat)

        # 10 output
        end_t = datetime.now()

        # 10.1 running time
        gwrmod.summary["BeginT"] = "%-21s: %s %s\n\n" % ('Program started at', datetime.date(begin_t), datetime.strftime(begin_t,"%H:%M:%S"))
        gwrmod.summary["EndT"] = "%-21s: %s %s\n\n" % ('Program terminated at', datetime.date(end_t), datetime.strftime(end_t,"%H:%M:%S"))

        # 10.2 model setting
	gwrmod.summary['ModSettings'] += "%-45s %s\n" % ('Method for optimal bandwidth search:', self.cmb_bdsel.get())
        if dic_ctrl["bdsel"] == 0 or dic_ctrl["bdsel"] == 1:
            gwrmod.summary['ModSettings'] += "%-45s %s\n" % ('Criterion for optimal bandwidth:', OPT_CRITERIA[dic_ctrl["optcri"]])
        gwrmod.summary['ModSettings'] += "%-45s %s\n" % ('Number of varying coefficients:', self.n_xloc)
        gwrmod.summary['ModSettings'] += "%-45s %s\n" % ('Number of fixed coefficients:', self.n_xglob)
        gwrmod.summary['ModSettings'] += '\n'

        # 10.3 model options
        gwrmod.summary['ModOptions'] += "%-60s %s\n" % ('Standardisation of independent variables:', OPTION[dic_ctrl["varstd"]])
        gwrmod.summary['ModOptions'] += "%-60s %s\n" % ('Testing geographical variability of local coefficients:', OPTION[dic_ctrl["vartest"]])
        gwrmod.summary['ModOptions'] += "%-60s %s\n" % ('Local to Global Variable selection:',OPTION[dic_ctrl["l2g"]])
        gwrmod.summary['ModOptions'] += "%-60s %s\n" % ('Global to Local Variable selection:', OPTION[dic_ctrl["g2l"]])
	gwrmod.summary['ModOptions'] += "%-60s %s\n" % ('Prediction at non-regression points:', OPTION[dic_ctrl["pred"]])
        gwrmod.summary['ModOptions'] += '\n'

        # 10.4 var settings
        gwrmod.summary['VarSettings'] += "%-60s %s\n" % ('Area key:', dic_ctrl["name_id"])
        gwrmod.summary['VarSettings'] += "%-60s %s\n" % ('Easting (x-coord):', dic_ctrl["name_xcoord"])
        gwrmod.summary['VarSettings'] += "%-60s %s\n" % ('Northing (y-coord):', dic_ctrl["name_ycoord"])
        gwrmod.summary['VarSettings'] += "%-60s %s\n" % ('Cartesian coordinates:', DIST_TYPE[dic_ctrl["disttype"]])
	gwrmod.summary['VarSettings'] += '\n'

        # 10.5 bandwidth searching output
	if dic_ctrl["bdsel"] == 0 or dic_ctrl["bdsel"] == 1:
	    gwrmod.summary['GWR_band'] += "%s\n" % ('Bandwidth search process:')
	    for elem in weit_output:
		gwrmod.summary['GWR_band'] += "%-s %20.3f, %s: %20.6f\n" % ("bandwidth:", elem[0], OPT_CRITERIA[dic_ctrl["optcri"]], elem[1])
	    gwrmod.summary['GWR_band'] += '\n'

        # 10.6 variability test
        if dic_ctrl["vartest"] == 1:
            gwrmod.summary['VaryTest'] += "%s\n" % ('Geographical variability tests of local coefficients')
	    gwrmod.summary['VaryTest'] += '-' * 75 + '\n'
	    if dic_ctrl["modtype"] == 0:
		gwrmod.summary['VaryTest'] += "%-20s %20s %20s %20s\n" %('Variable','F','DOF for F test', 'DIFF of Criterion')
		gwrmod.summary['VaryTest'] += "%-20s %20s %20s %20s\n" %('-'*20, '-'*20, '-'*20, '-'*20)
	    else:
		gwrmod.summary['VaryTest'] += "%-20s %20s %20s %20s\n" % ('Variable',"Diff of deviance","Diff of DOF","DIFF of Criterion")
		gwrmod.summary['VaryTest'] += "%-20s %20s %20s %20s\n" %('-'*20, '-'*20, '-'*20, '-'*20)
	    for i in range(self.n_xloc):
		if dic_ctrl["modtype"] == 0:
		    gwrmod.summary['VaryTest'] += "%-20s %20.6f %10.3f %10.3f %20.6f\n" %(dic_ctrl["name_xloc"][i],rs_varytest[i][0],rs_varytest[i][1],rs_varytest[i][2],rs_varytest[i][3])
		else:
		    gwrmod.summary['VaryTest'] += "%-20s %20.6f %20.6f %20.6f\n" %(dic_ctrl["name_xloc"][i],rs_varytest[i][0],rs_varytest[i][1],rs_varytest[i][2])
            gwrmod.summary['VaryTest'] += '\n'

        # 10.7 L->G or G->L test
        if dic_ctrl["l2g"] == 1:
	    gwrmod.summary['l2g'] += "%s\n" % ('(L -> G) Variable selection from varying coefficients to fixed coefficients')
	    gwrmod.summary['l2g'] += '-' * 75 + '\n'
	    for bd_sele in test_l2g[2]:  #varsL, varsG, optband, optWeit, cri_old
		gwrmod.summary['l2g'] += "%s\n" % ('Bandwidth search process:')
		for elem in bd_sele[2]:
		    gwrmod.summary['l2g'] += "%-37s %20.3f, %5s: %20.6f\n" % ("bandwidth:", elem[0], OPT_CRITERIA[dic_ctrl["optcri"]], elem[1])
		gwrmod.summary['l2g'] += '\n'
	    gwrmod.summary['l2g'] += "%s\n" % ('The summary of the L -> G variable selection')
	    gwrmod.summary['l2g'] +=  '-' * 55 + '\n'
	    gwrmod.summary['l2g'] += "%-35s   %20s\n" % ('Model', OPT_CRITERIA[dic_ctrl["optcri"]])
	    gwrmod.summary['l2g'] += "%-35s   %20s\n" % ('-'*35, '-'*20)
	    gwrmod.summary['l2g'] += "%-35s   %20.6f\n" % ('GWR model before L -> G selection', cri_origin)
	    gwrmod.summary['l2g'] += "%-35s   %20.6f\n" % ('GWR model after  L -> G selection', cri_l2g)
	    gwrmod.summary['l2g'] += "%-35s   %20.6f\n" % ('Improvement', cri_origin-cri_l2g)
	    gwrmod.summary['l2g'] += '\n'


        if dic_ctrl["g2l"] == 1:
	    gwrmod.summary['g2l'] += "%s\n" % ('(G -> L) Variable selection from fixed coefficients to varying coefficients')
	    gwrmod.summary['g2l'] += '-' * 75 + '\n'
	    for bd_sele in test_g2l[2]:  #varsL, varsG, optband, optWeit, cri_old
		gwrmod.summary['g2l'] += "%s\n" % ('Bandwidth search process:')
		for elem in bd_sele[2]:
		    gwrmod.summary['g2l'] += "%-37s %20.3f, %5s: %20.6f\n" % ("bandwidth:", elem[0], OPT_CRITERIA[dic_ctrl["optcri"]], elem[1])
		gwrmod.summary['g2l'] += '\n'
	    gwrmod.summary['g2l'] += "%s\n" % ('The summary of the G -> L variable selection')
	    gwrmod.summary['g2l'] +=  '-' * 55 + '\n'
	    gwrmod.summary['g2l'] += "%-35s   %20s\n" % ('Model', OPT_CRITERIA[dic_ctrl["optcri"]])
	    gwrmod.summary['g2l'] += "%-35s   %20s\n" % ('-'*35, '-'*20)
	    gwrmod.summary['g2l'] += "%-35s   %20.6f\n" % ('GWR model before G -> L selection', cri_origin)
	    gwrmod.summary['g2l'] += "%-35s   %20.6f\n" % ('GWR model after  G -> L selection', cri_g2l)
	    gwrmod.summary['g2l'] += "%-35s   %20.6f\n" % ('Improvement', cri_origin-cri_g2l)
	    gwrmod.summary['g2l'] += '\n'

	# 10.8 output to new window
	if new_mod: # find better model
	    gwrmod.summary['newMod'] += "%s\n" % ('Model summary and local stats are being updated by the improved model.')
	    gwrmod.summary['newMod'] += '*' * 75 + '\n' + '\n'
	    gwrmod.summary['newMod'] += gwrmod_new.summary['GWRResult']
	    gwrmod.summary['newMod'] += gwrmod_new.summary['GWR_band']
	    gwrmod.summary['newMod'] += gwrmod_new.summary['GWR_diag']
	    gwrmod.summary['newMod'] += gwrmod_new.summary['GWR_esti_glob']
	    gwrmod.summary['newMod'] += gwrmod_new.summary['GWR_esti']
	    gwrmod.summary['newMod'] += '\n'

        str_out = gwrmod.summaryPrint()

	# 11 save to file
        # 11.1 save control file
        self.saveFle_ctrl(dic_ctrl["flepath_ctrl"],dic_ctrl)

        # 11.2 save summary file
        with open(dic_ctrl["flepath_sum"], 'w') as sumfile:
	    sumfile.write(str_out)
	    sumfile.close()

        # 11.3 save listwise file
	if new_mod:
	    self.saveFle_beta(gwrmod_new, dic_ctrl["flepath_beta"], dic_ctrl["name_id"],values_id)
	else:
	    self.saveFle_beta(gwrmod, dic_ctrl["flepath_beta"], dic_ctrl["name_id"],values_id)

        # 11.4 save prediction file
	if dic_ctrl["pred"] == 1:
	    if new_mod:
		self.saveFle_pred(gwrmod_new, rs_pred[0], rs_pred[1], rs_pred[2], rs_pred[3], dic_ctrl["flepath_predout"])
	    else:
		self.saveFle_pred(gwrmod, rs_pred[0], rs_pred[1], rs_pred[2], rs_pred[3], dic_ctrl["flepath_predout"])

        # 12 output window

	self.infoWin = Info(self, str_out)
        self.infoWin.mainloop()

        #if not self.openInfo:
            #self.openInfo = True
            #self.infoWin = Info(self, str_out)
            #self.infoWin.mainloop()

        #if not self.infoWin.winfo_exists():
            #self.infoWin = Info(self, str_out)
            #self.infoWin.mainloop()

        #if self.infoWin.state == "withdrawn": # if the toplevel window is invisible, make it visible
            #self.infoWin.deiconify()


    def createWidget(self):
        """
        create controls

        """
        #Style().configure("TCanvas", foreground="black", background="white")

        # canvas containing all the widgets
        self.Container = Canvas(self)
        #self.Container["bg"] = "white"
        #self.Container["width"] = 580
        self.Container["height"] = 600
        self.Container.grid(row=0,column=0, columnspan=70, rowspan=80)

        #----------------------------Frame 1----------------------------------------

        # frame 1: open data file
        #ttk.Style().configure("TLabelframe", foreground="black", background="white")

        self.frm_open = ttk.LabelFrame(self)#, style="TLabelframe"
        self.frm_open["text"] = "Data File"
        self.frm_open["height"] = 50
        self.frm_open["width"] = 200
        self.frm_open.grid(row=1,column=1,padx=5,pady=2)
        self.frm_open.columnconfigure(0, weight=9)
        self.frm_open.columnconfigure(1, weight=1)

        # text: open data file
        self.txt_open = ttk.Entry(self.frm_open)#
        self.txt_open["width"] = 24
        #self.txt_open["height"] = 2
        self.txt_open.grid(row=0,column=0, sticky=W, padx=5,pady=2)#

        # button: open file
        self.btn_open = ttk.Button(self.frm_open)#
        self.btn_open["width"] = 3
        #self.btn_open["height"] = 2
        self.img_open = PhotoImage(file=sys.path[0] + "\\img\\openfolder.gif")
        self.btn_open["image"] = self.img_open
        self.btn_open["command"] = lambda arg1=0: self.openFile(arg1) #self.openFile(0)
        self.btn_open.grid(row=0,column=1,padx=5)#

        #-------------------------------Frame 2--------------------------------------------

        # frame 2: choose location variables
        self.frm_locVars = ttk.LabelFrame(self)
        self.frm_locVars["text"] = "Location Variables"
        self.frm_locVars["height"] = 80
        self.frm_locVars["width"] = 200
        self.frm_locVars.grid(row=2,column=1,padx=5)#,pady=2
        self.frm_locVars.columnconfigure(0, weight=1)
        self.frm_locVars.columnconfigure(1, weight=1)
        self.frm_locVars.columnconfigure(2, weight=1)
        self.frm_locVars.columnconfigure(3, weight=1)
        self.frm_locVars.columnconfigure(4, weight=1)
        self.frm_locVars.columnconfigure(5, weight=1)
        self.frm_locVars.columnconfigure(6, weight=1)
        self.frm_locVars.columnconfigure(7, weight=1)
        self.frm_locVars.columnconfigure(8, weight=1)
        self.frm_locVars.columnconfigure(9, weight=1)

        # label: id
        self.lbl_id = ttk.Label(self.frm_locVars)
        self.lbl_id["text"] = "ID"
        self.lbl_id.grid(row=0,column=0,sticky=W,padx=2)#

        # text: for id variable
        self.txt_id = ttk.Entry(self.frm_locVars)#
        #self.txx_id["width"] = 20
        self.txt_id.grid(row=0,column=1,columnspan=7,padx=5)#,  pady=5, sticky=W

        # button: add id
        self.btn_addID = ttk.Button(self.frm_locVars)#
        self.btn_addID["width"] = 2
        self.btn_addID["text"] = '<'
        self.btn_addID.grid(row=0,column=8)#,padx=1

        # button: remove id
        self.btn_outID = ttk.Button(self.frm_locVars)#
        self.btn_outID["width"] = 2
        self.btn_outID["text"] = '>'
        self.btn_outID.grid(row=0,column=9,sticky=E)#,padx=1

        # label: x
        self.lbl_xcoord = ttk.Label(self.frm_locVars)
        self.lbl_xcoord["text"] = "X"
        self.lbl_xcoord.grid(row=1,column=0,sticky=W,padx=2)#,pady=2

        # text: for x coord
        self.txt_xcoord = ttk.Entry(self.frm_locVars)#
        #self.txx_xcoord["width"] = 20
        self.txt_xcoord.grid(row=1,column=1,columnspan=7,padx=5)#, padx=5,pady=5

        # button: add x coord
        self.btn_addxcoord = ttk.Button(self.frm_locVars)#
        self.btn_addxcoord["width"] = 2
        self.btn_addxcoord["text"] = '<'
        self.btn_addxcoord.grid(row=1,column=8)#,padx=1

        # button: remove x coord
        self.btn_outxcoord = ttk.Button(self.frm_locVars)#
        self.btn_outxcoord["width"] = 2
        self.btn_outxcoord["text"] = '>'
        self.btn_outxcoord.grid(row=1,column=9,sticky=E)#,padx=1

        # label: y
        self.lbl_ycoord = ttk.Label(self.frm_locVars)
        self.lbl_ycoord["text"] = "Y"
        self.lbl_ycoord.grid(row=2,column=0,sticky=W,padx=2)#,pady=2

        # text: for y coord
        self.txt_ycoord = ttk.Entry(self.frm_locVars)#
        #self.txx_ycoord["width"] = 20
        self.txt_ycoord.grid(row=2,column=1,columnspan=7,padx=5)#, padx=5,pady=5

        # button: add y coord
        self.btn_addycoord = ttk.Button(self.frm_locVars)#
        self.btn_addycoord["width"] = 2
        self.btn_addycoord["text"] = '<'
        self.btn_addycoord.grid(row=2,column=8)#,padx=1

        # button: remove y coord
        self.btn_outycoord = ttk.Button(self.frm_locVars)#
        self.btn_outycoord["width"] = 2
        self.btn_outycoord["text"] = '>'
        self.btn_outycoord.grid(row=2,column=9,sticky=E)#,padx=1

        # control variable for type of distance
        self.var_disttype = IntVar()

        # radiobutton: projected
        self.rbtn_Eucdist = ttk.Radiobutton(self.frm_locVars)
        self.rbtn_Eucdist["text"] = "Projected"
        self.rbtn_Eucdist["variable"] = self.var_disttype
        self.rbtn_Eucdist["value"] = 0
        self.rbtn_Eucdist.grid(row=3,column=0,columnspan=5,padx=5)#,,sticky=W

        # radiobutton: spherical
        self.rbtn_Sphdist = ttk.Radiobutton(self.frm_locVars)
        self.rbtn_Sphdist["text"] = "Spherical"
        self.rbtn_Sphdist["variable"] = self.var_disttype
        self.rbtn_Sphdist["value"] = 1
        self.rbtn_Sphdist.grid(row=3,column=5,columnspan=5,padx=5)#, ,sticky=W,column=2

        self.var_disttype.set(0)

        #--------------------------------Frame 3-----------------------------------------------

        # frame 3: kernel settings
        #ttk.Style().configure("TLabelframe", foreground="black", background="white")

        self.frm_weit = ttk.LabelFrame(self)#, style="TLabelframe"
        self.frm_weit["text"] = "Kernel"
        #self.frm_weit["height"] = 50
        self.frm_weit["width"] = 200
        self.frm_weit.grid(row=3,column=1,padx=5)#,pady=2
        self.frm_weit.columnconfigure(0, weight=1)
        self.frm_weit.columnconfigure(1, weight=1)
        self.frm_weit.columnconfigure(2, weight=1)
        self.frm_weit.columnconfigure(3, weight=1)
        self.frm_weit.columnconfigure(4, weight=1)
        self.frm_weit.columnconfigure(5, weight=1)

        # Label: kernel type
        self.lbl_wtype = ttk.Label(self.frm_weit)
        self.lbl_wtype["text"] = "Kernel type"
        self.lbl_wtype.grid(row=0,column=0,columnspan=2,padx=2,sticky=W)#,pady=2

        # combox: kernel type
        self.cmb_wtype = ttk.Combobox(self.frm_weit)
        self.cmb_wtype["values"] = ["Fixed Gaussian (distance)","Adaptive Gaussian (NN)", "Fixed bisquare (distance)", "Adaptive bisquare (NN)"]
        self.cmb_wtype.set("Fixed Gaussian (distance)")
        self.cmb_wtype["width"] = 25
        self.cmb_wtype.grid(row=1,column=0,columnspan=6,padx=2,sticky=W)#,

        # label: bandwidth searching method
        self.lbl_bdsel = ttk.Label(self.frm_weit)
        self.lbl_bdsel["text"] = "Bandwidth selection method"
        self.lbl_bdsel.grid(row=2,column=0,columnspan=6,padx=2,pady=2, sticky=W)#

        # combox: bandwidth searching method
        self.cmb_bdsel = ttk.Combobox(self.frm_weit)
        self.cmb_bdsel["values"] = ["Golden section search","Interval search","Predetermined bandwidth"]
        self.cmb_bdsel.set("Golden section search")
        self.cmb_bdsel["width"] = 25
        self.cmb_bdsel.grid(row=3,column=0,columnspan=6,padx=2,pady=2,sticky=W)#,

        # label: single bandwidth value
        self.lbl_bdval = ttk.Label(self.frm_weit)
        self.lbl_bdval["text"] = "Value"
        self.lbl_bdval.grid(row=4,column=0,padx=2,pady=1,sticky=W)#,columnspan=1

        # text: single bandwidth value
        self.txt_bdval = ttk.Entry(self.frm_weit)
        self.txt_bdval["width"]=23
        #self.txt_bdval["state"]=DISABLED
        self.txt_bdval.grid(row=4,column=1,columnspan=5,padx=3,pady=1)#,sticky=E

        # label: max bandwidth value
        self.lbl_bdmax = ttk.Label(self.frm_weit)
        self.lbl_bdmax["text"] = "Max."
        self.lbl_bdmax.grid(row=5,column=0,padx=2,pady=1,sticky=W)#,columnspan=1

        # text: max bandwidth value
        self.txt_bdmax = ttk.Entry(self.frm_weit)
        self.txt_bdmax["width"]=23
        self.txt_bdmax.grid(row=5,column=1,columnspan=5,padx=3,pady=1)#,sticky=E

        # label: min bandwidth value
        self.lbl_bdmin = ttk.Label(self.frm_weit)
        self.lbl_bdmin["text"] = "Min."
        self.lbl_bdmin.grid(row=6,column=0,padx=2,pady=1,sticky=W)#,columnspan=1

        # text: min bandwidth value
        self.txt_bdmin = ttk.Entry(self.frm_weit)
        self.txt_bdmin["width"]=23
        self.txt_bdmin.grid(row=6,column=1,columnspan=5,padx=3,pady=1)#,sticky=E

        # label: interval bandwidth value
        self.lbl_bdstep = ttk.Label(self.frm_weit)
        self.lbl_bdstep["text"] = "Interval"
        self.lbl_bdstep.grid(row=7,column=0,padx=2,pady=1,sticky=W)#,columnspan=1

        # text: interval bandwidth value
        self.txt_bdstep = ttk.Entry(self.frm_weit)
        self.txt_bdstep["width"] = 23
        #self.txt_bdstep["state"] = DISABLED
        self.txt_bdstep.grid(row=7,column=1,columnspan=5,padx=3,pady=1)#,sticky=E

        #self.cmb_bdsel["command"] = self.set_bw
        #--------------------------------Frame 4----------------------------------------------

        # frame 4: list all the variables in the data file
        self.frm_varlist = ttk.LabelFrame(self)#, style="TLabelframe"
        self.frm_varlist["text"] = "Variables"
        #self.frm_varlist["height"] = 600 #self.frm_open["height"]+self.frm_locVars["height"]+self.frm_weit["height"]#50
        self.frm_varlist["width"] = 200
        self.frm_varlist.grid(row=1,column=2,rowspan=3,pady=2)#,padx=2

        # list box: list all the variables
        self.lstb_allvars = ttk.Tkinter.Listbox(self.frm_varlist)
        self.lstb_allvars["height"] = 22
        self.lstb_allvars["width"] = 23
        self.lstb_allvars.grid(row=0,column=0,padx=2,pady=3)

        # scroll bar
        self.scrollY_allvars = ttk.Scrollbar(self.frm_varlist, orient=VERTICAL, command=self.lstb_allvars.yview )
        self.scrollY_allvars.grid(row=0, column=1, sticky=N+S )
        self.scrollX_allvars = ttk.Scrollbar(self.frm_varlist, orient=HORIZONTAL, command=self.lstb_allvars.xview )
        self.scrollX_allvars.grid(row=1, column=0, sticky=E+W )

        self.lstb_allvars["xscrollcommand"] = self.scrollX_allvars.set
        self.lstb_allvars["yscrollcommand"] = self.scrollY_allvars.set

        # btn command
        self.btn_addID["command"] = lambda arg1=self.txt_id, arg2=None: self.addVars(arg1, arg2) #self.addVars(self.txt_id)
        self.btn_outID["command"] = lambda arg1=self.txt_id, arg2=None: self.outVars(arg1, arg2)#self.outVars(self.txt_id)
        self.btn_addxcoord["command"] = lambda arg1=self.txt_xcoord, arg2=None: self.addVars(arg1, arg2)#self.addVars(self.txt_xcoord)
        self.btn_outxcoord["command"] = lambda arg1=self.txt_xcoord, arg2=None: self.outVars(arg1, arg2)#self.outVars(self.txt_xcoord)
        self.btn_addycoord["command"] = lambda arg1=self.txt_ycoord, arg2=None: self.addVars(arg1, arg2)#self.addVars(self.txt_ycoord)
        self.btn_outycoord["command"] = lambda arg1=self.txt_ycoord, arg2=None: self.outVars(arg1, arg2)#self.outVars(self.txt_ycoord)

        #--------------------------------Frame 5----------------------------------------------

        # frame5: regression varaibles
        self.frm_regVar = ttk.LabelFrame(self)
        self.frm_regVar["text"] = "Regression Variables"
        self.frm_regVar["width"] = 200
        self.frm_regVar.grid(row=1,column=3,rowspan=3,padx=5,pady=3)
        self.frm_regVar.rowconfigure(0, weight=1)
        self.frm_regVar.rowconfigure(1, weight=1)
        self.frm_regVar.rowconfigure(2, weight=1)
        self.frm_regVar.rowconfigure(3, weight=1)
        self.frm_regVar.rowconfigure(4, weight=1)
        self.frm_regVar.rowconfigure(5, weight=1)
        self.frm_regVar.rowconfigure(6, weight=1)
        #self.frm_regVar.rowconfigure(7, weight=1)
        #self.frm_regVar.rowconfigure(8, weight=1)
        #self.frm_regVar.rowconfigure(9, weight=1)
        #self.frm_regVar.rowconfigure(10, weight=1)
        #self.frm_regVar.rowconfigure(11, weight=1)
        #self.frm_regVar.rowconfigure(12, weight=1)
        #self.frm_regVar.rowconfigure(13, weight=1)

        # button: remove y
        self.btn_outy = ttk.Button(self.frm_regVar )
        self.btn_outy["text"] = '<'
        self.btn_outy["width"] = 2
        self.btn_outy.grid(row=0,column=0)

        # button: add y
        self.btn_addy = ttk.Button(self.frm_regVar )
        self.btn_addy["text"] = '>'
        self.btn_addy["width"] = 2
        self.btn_addy.grid(row=0,column=1)

        # label: y
        self.lbl_y = ttk.Label(self.frm_regVar)
        self.lbl_y["text"] = 'Y'
        self.lbl_y.grid(row=0,column=2,padx=2,sticky=W)#,pady=2

        # text: y
        self.txt_y = ttk.Entry(self.frm_regVar)
        self.txt_y["width"] = 16
        self.txt_y.grid(row=0,column=3,padx=2)

        self.btn_outy["command"] = lambda arg1=self.txt_y, arg2=None: self.outVars(arg1, arg2)#self.outVars(self.txt_y)
        self.btn_addy["command"] = lambda arg1=self.txt_y, arg2=None: self.addVars(arg1, arg2)#self.addVars(self.txt_y)

        # button: remove y offset
        self.btn_outyoff = ttk.Button(self.frm_regVar )
        self.btn_outyoff["text"] = '<'
        self.btn_outyoff["width"] = 2
        self.btn_outyoff.grid(row=1,column=0)

        # button: add y offset
        self.btn_addyoff = ttk.Button(self.frm_regVar )
        self.btn_addyoff["text"] = '>'
        self.btn_addyoff["width"] = 2
        self.btn_addyoff.grid(row=1,column=1)

        # label: y offset
        self.lbl_yoff = ttk.Label(self.frm_regVar)
        self.lbl_yoff["text"] = 'Offset'
        self.lbl_yoff.grid(row=1,column=2,padx=2,sticky=W)#

        # text: y offset
        self.txt_yoff = ttk.Entry(self.frm_regVar)
        self.txt_yoff["width"] = 16
        self.txt_yoff["state"] = DISABLED
        self.txt_yoff.grid(row=1,column=3,padx=2)

        self.btn_addyoff["command"] = lambda arg1=self.txt_yoff, arg2=None: self.addVars(arg1, arg2)#self.addVars(self.txt_yoff)
        self.btn_outyoff["command"] = lambda arg1=self.txt_yoff, arg2=None: self.outVars(arg1, arg2)#self.outVars(self.txt_yoff)

        # label: x local
        self.lbl_xloc = ttk.Label(self.frm_regVar)
        self.lbl_xloc["text"] = 'Local X'
        self.lbl_xloc.grid(row=2,column=2,padx=2,pady=2,sticky=W) #

        # listbox: x local
        self.lstb_xloc = ttk.Tkinter.Listbox(self.frm_regVar)
        self.lstb_xloc["height"] = 8
        self.lstb_xloc["width"] = 25
        self.lstb_xloc.insert(0,"000 Intercept")
        self.lstb_xloc.grid(row=3,column=2,columnspan=2,padx=2,sticky=W)#,rowspan=2,pady=2

        # scrollbar: x local
        self.scrollY_xloc = ttk.Scrollbar(self.frm_regVar, orient=VERTICAL, command=self.lstb_xloc.yview )
        self.scrollY_xloc.grid(row=3, column=3, sticky=N+S+E )
        self.scrollX_xloc = ttk.Scrollbar(self.frm_regVar, orient=HORIZONTAL, command=self.lstb_xloc.xview )
        self.scrollX_xloc.grid(row=3, column=2, columnspan=2,sticky=E+W+S )

        self.lstb_xloc["xscrollcommand"] = self.scrollX_xloc.set
        self.lstb_xloc["yscrollcommand"] = self.scrollY_xloc.set

        # button: remove x local
        self.btn_outxloc = ttk.Button(self.frm_regVar)
        self.btn_outxloc["text"] = '<'
        self.btn_outxloc["width"] = 2
        self.btn_outxloc["command"] = lambda arg1=None, arg2=self.lstb_xloc: self.outVars(arg1, arg2)#self.outVars(None,self.lstb_xloc)
        self.btn_outxloc.grid(row=3,column=0)

        # button: add x local
        self.btn_addxloc = ttk.Button(self.frm_regVar)
        self.btn_addxloc["text"] = '>'
        self.btn_addxloc["width"] = 2
        self.btn_addxloc["command"] = lambda arg1=None, arg2=self.lstb_xloc: self.addVars(arg1, arg2)#self.addVars(None,self.lstb_xloc)
        self.btn_addxloc.grid(row=3,column=1)

        # label: x global
        self.lbl_xglob = ttk.Label(self.frm_regVar)
        self.lbl_xglob["text"] = 'Global X'
        self.lbl_xglob.grid(row=4,column=2,padx=2,sticky=W) #,pady=3,

        # listbox: x global
        self.lstb_xglob = ttk.Tkinter.Listbox(self.frm_regVar)
        self.lstb_xglob["height"] = 8
        self.lstb_xglob["width"] = 25
        self.lstb_xglob.grid(row=5,column=2,columnspan=2,padx=2,pady=3,sticky=W)#,rowspan=2

        # scrollbar: x global
        self.scrollY_xglob = ttk.Scrollbar(self.frm_regVar, orient=VERTICAL, command=self.lstb_xglob.yview )
        self.scrollY_xglob.grid(row=5, column=3, sticky=N+S+E )
        self.scrollX_xglob = ttk.Scrollbar(self.frm_regVar, orient=HORIZONTAL, command=self.lstb_xglob.xview )
        self.scrollX_xglob.grid(row=5, column=2, columnspan=2,sticky=E+W+S )

        self.lstb_xglob["xscrollcommand"] = self.scrollX_xglob.set
        self.lstb_xglob["yscrollcommand"] = self.scrollY_xglob.set

        # button: remove x global
        self.btn_outxglob = ttk.Button(self.frm_regVar)
        self.btn_outxglob["text"] = '<'
        self.btn_outxglob["width"] = 2
        self.btn_outxglob["command"] = lambda arg1=None, arg2=self.lstb_xglob: self.outVars(arg1, arg2)#self.outVars(None,self.lstb_xglob)
        self.btn_outxglob.grid(row=5,column=0)

        # button: add x global
        self.btn_addxglob = ttk.Button(self.frm_regVar)
        self.btn_addxglob["text"] = '>'
        self.btn_addxglob["width"] = 2
        self.btn_addxglob["command"] = lambda arg1=None, arg2=self.lstb_xglob: self.addVars(arg1, arg2)#self.addVars(None, self.lstb_xglob)
        self.btn_addxglob.grid(row=5,column=1)

        #--------------------------------Frame 6----------------------------------------------

        # frame6: model options
        self.frm_mod = ttk.LabelFrame(self)
        self.frm_mod["text"] = "Model Options"
        #self.frm_mod["width"] = 200
        self.frm_mod.grid(row=4,column=1,columnspan=3,padx=5)
        self.frm_mod.columnconfigure(0, weight=1)
        self.frm_mod.columnconfigure(1, weight=1)
        self.frm_mod.columnconfigure(2, weight=1)

        #------------------------frame 6.1: model type-----------------------
        self.frm_modtype = ttk.LabelFrame(self.frm_mod)
        self.frm_modtype["text"] = "Model type"
        #self.frm_mod["width"] = 200
        self.frm_modtype.grid(row=0,column=0,padx=5,pady=1)

        # control variable for type of models
        self.var_modtype = IntVar()

        # radiobutton: gaussian
        self.rbtn_gau = ttk.Radiobutton(self.frm_modtype)
        self.rbtn_gau["text"] = "Gaussian"
        self.rbtn_gau["variable"] = self.var_modtype
        self.rbtn_gau["value"] = 0
        self.rbtn_gau["width"] = 20
        self.rbtn_gau["command"] = self.set_modtype
        self.rbtn_gau.grid(row=0,column=0,padx=5,pady=5,sticky=W)

        # radiobutton: poisson
        self.rbtn_poson = ttk.Radiobutton(self.frm_modtype)
        self.rbtn_poson["text"] = "Poisson"
        self.rbtn_poson["variable"] = self.var_modtype
        self.rbtn_poson["value"] = 1
        self.rbtn_poson["width"] = 20
        self.rbtn_poson["command"] = self.set_modtype
        self.rbtn_poson.grid(row=1,column=0,padx=5,pady=5,sticky=W)

        # radiobutton: logistic
        self.rbtn_log = ttk.Radiobutton(self.frm_modtype)
        self.rbtn_log["text"] = "Logistic"
        self.rbtn_log["variable"] = self.var_modtype
        self.rbtn_log["value"] = 2
        self.rbtn_log["width"] = 20
        self.rbtn_log["command"] = self.set_modtype
        self.rbtn_log.grid(row=2,column=0,padx=5,pady=5,sticky=W)

        self.var_modtype.set(0)

        #-----------------------frame 6.2: optimization method----------------------
        self.frm_optcri = ttk.LabelFrame(self.frm_mod)
        self.frm_optcri["text"] = "Optimization criterion"
        #self.frm_mod["width"] = 200
        self.frm_optcri.grid(row=0,column=1,padx=2,pady=1)

        # control variable for type of models
        self.var_optcri = IntVar()

        # radiobutton: aicc
        self.rbtn_aicc = ttk.Radiobutton(self.frm_optcri)
        self.rbtn_aicc["text"] = "AICc"
        self.rbtn_aicc["variable"] = self.var_optcri
        self.rbtn_aicc["value"] = 0
        self.rbtn_aicc["width"] = 20
        self.rbtn_aicc.grid(row=0,column=0,padx=5,pady=1,sticky=W)

        # radiobutton: aic
        self.rbtn_aic = ttk.Radiobutton(self.frm_optcri)
        self.rbtn_aic["text"] = "AIC"
        self.rbtn_aic["variable"] = self.var_optcri
        self.rbtn_aic["value"] = 1
        self.rbtn_aic["width"] = 20
        self.rbtn_aic.grid(row=1,column=0,padx=5,pady=1,sticky=W)

        # radiobutton: bic
        self.rbtn_bic = ttk.Radiobutton(self.frm_optcri)
        self.rbtn_bic["text"] = "BIC"
        self.rbtn_bic["variable"] = self.var_optcri
        self.rbtn_bic["value"] = 2
        self.rbtn_bic["width"] = 20
        self.rbtn_bic.grid(row=2,column=0,padx=5,pady=1,sticky=W)

        # radiobutton: cv
        self.rbtn_cv = ttk.Radiobutton(self.frm_optcri)
        self.rbtn_cv["text"] = "CV"
        self.rbtn_cv["variable"] = self.var_optcri
        self.rbtn_cv["value"] = 3
        self.rbtn_cv["width"] = 20
        self.rbtn_cv.grid(row=3,column=0,padx=5,pady=1,sticky=W)

        self.var_optcri.set(0)

        #-----------------------frame 6.3: vairialbe selection----------------------
        self.frm_vartest = ttk.LabelFrame(self.frm_mod)
        self.frm_vartest["text"] = "Advanced options"
        #self.frm_mod["width"] = 200
        self.frm_vartest.grid(row=0,column=2,padx=5,pady=1)

        # checkbutton: var standardisation
        self.var_varstd = IntVar()
        self.chkbtn_varstd = ttk.Checkbutton(self.frm_vartest)
        self.chkbtn_varstd["text"] = "Variable standardisation"
        self.chkbtn_varstd["width"] = 31
        self.chkbtn_varstd["variable"] = self.var_varstd
        self.chkbtn_varstd.grid(row=0,column=0,padx=5,pady=1,sticky=W)
        self.var_varstd.set(0)

        # checkbutton: local test
        self.var_vartest = IntVar()
        self.chkbtn_loctest = ttk.Checkbutton(self.frm_vartest)
        self.chkbtn_loctest["text"] = "Geographical variability test"
        self.chkbtn_loctest["width"] = 31
        self.chkbtn_loctest["variable"] = self.var_vartest
        self.chkbtn_loctest.grid(row=1,column=0,padx=5,pady=1,sticky=W)
        self.var_vartest.set(0)

        # checkbutton: L->G
        self.var_l2g = IntVar()
        self.chkbtn_l2g = ttk.Checkbutton(self.frm_vartest)
        self.chkbtn_l2g["text"] = "L->G variable selection"
        self.chkbtn_l2g["width"] = 31
        self.chkbtn_l2g["variable"] = self.var_l2g
        self.chkbtn_l2g.grid(row=2,column=0,padx=5,pady=1,sticky=W)
        self.var_l2g.set(0)

        # checkbutton: G->L
        self.var_g2l = IntVar()
        self.chkbtn_g2l = ttk.Checkbutton(self.frm_vartest)
        self.chkbtn_g2l["text"] = "G->L variable selection"
        self.chkbtn_g2l["width"] = 31
        self.chkbtn_g2l["variable"] = self.var_g2l
        self.chkbtn_g2l.grid(row=3,column=0,padx=5,pady=1,sticky=W)
        self.var_g2l.set(0)

        #--------------------------------Frame 7----------------------------------------------

        # frame: output
        self.frm_output = ttk.LabelFrame(self)
        self.frm_output["text"] = "Outputs"
        #self.frm_mod["width"] = 200
        self.frm_output.grid(row=5,column=1,columnspan=3,padx=5,pady=3,sticky=W)

        # label: control file
        self.lbl_flectrl = ttk.Label(self.frm_output)
        self.lbl_flectrl["text"] = "Control file"
        self.lbl_flectrl.grid(row=0, column=0,padx=2,sticky=W)

        # text: control file
        self.txt_flectrl = ttk.Entry(self.frm_output)
        self.txt_flectrl["width"] = 55
        self.txt_flectrl.grid(row=0, column=1,padx=2,sticky=W)

        # button: control file
        self.btn_flectrl = ttk.Button(self.frm_output)
        self.btn_flectrl["width"] = 3
        self.btn_flectrl["command"] = lambda arg1=2: self.openFile(arg1)#self.openFile(2)
        self.btn_flectrl["image"] = self.img_open
        self.btn_flectrl.grid(row=0, column=2,padx=5)

        # label: summary file
        self.lbl_flesum = ttk.Label(self.frm_output)
        self.lbl_flesum["text"] = "Summary file"
        self.lbl_flesum.grid(row=1, column=0,padx=2,sticky=W)

        # text: summary file
        self.txt_flesum = ttk.Entry(self.frm_output)
        self.txt_flesum["width"] = 55
        self.txt_flesum.grid(row=1, column=1,padx=2,sticky=W)

        # button: summary file
        self.btn_flesum = ttk.Button(self.frm_output)
        self.btn_flesum["width"] = 3
        self.btn_flesum["image"] = self.img_open
        self.btn_flesum["command"] = lambda arg1=3: self.openFile(arg1) #self.openFile(3)
        self.btn_flesum.grid(row=1, column=2,padx=5)

        # label: parameter file
        self.lbl_flebeta = ttk.Label(self.frm_output)
        self.lbl_flebeta["text"] = "Local estimates file"
        self.lbl_flebeta.grid(row=2, column=0,padx=2,sticky=W)

        # text: parameter file
        self.txt_flebeta = ttk.Entry(self.frm_output)
        self.txt_flebeta["width"] = 55
        self.txt_flebeta.grid(row=2, column=1,padx=2,sticky=W)

        # button: parameter file
        self.btn_flebeta = ttk.Button(self.frm_output)
        self.btn_flebeta["width"] = 3
        self.btn_flebeta["image"] = self.img_open
        self.btn_flebeta["command"] = lambda arg1=4: self.openFile(arg1)#self.openFile(4)
        self.btn_flebeta.grid(row=2, column=2,padx=5)

        # checkbutton: prediction
        self.var_pred = IntVar()

        self.chkbtn_pred = ttk.Checkbutton(self.frm_output)
        self.chkbtn_pred["text"] = "Prediction at non-sample points"
        self.chkbtn_pred["width"] = 31
        self.chkbtn_pred["variable"] = self.var_pred
        self.chkbtn_pred.grid(row=3,column=0,columnspan=2,padx=2,sticky=W)  #,pady=1

        self.var_pred.set(0)

        # label: prediction data file
        self.lbl_flepred = ttk.Label(self.frm_output)
        self.lbl_flepred["text"] = "Data file"
        self.lbl_flepred.grid(row=4, column=0,padx=2,sticky=W)

        # text: prediction data file
        self.txt_flepred = ttk.Entry(self.frm_output)
        self.txt_flepred["width"] = 55
        self.txt_flepred["state"] = DISABLED
        self.txt_flepred.grid(row=4, column=1,padx=2,sticky=W)

        # button: prediction data file
        self.btn_flepred = ttk.Button(self.frm_output)
        self.btn_flepred["width"] = 3
        self.btn_flepred["image"] = self.img_open
        self.btn_flepred["command"] = lambda arg1=1: self.openFile(arg1)#self.openFile(1)
        self.btn_flepred["state"] = DISABLED
        self.btn_flepred.grid(row=4, column=2,padx=5)

        # label: prediction output file
        self.lbl_flepredout = ttk.Label(self.frm_output)
        self.lbl_flepredout["text"] = "Data file"
        self.lbl_flepredout.grid(row=5, column=0,padx=2,sticky=W)

        # text: prediction output file
        self.txt_flepredout = ttk.Entry(self.frm_output)
        self.txt_flepredout["width"] = 55
        self.txt_flepredout["state"] = DISABLED
        self.txt_flepredout.grid(row=5, column=1,padx=2,sticky=W)

        # button: prediction output file
        self.btn_flepredout = ttk.Button(self.frm_output)
        self.btn_flepredout["width"] = 3
        self.btn_flepredout["image"] = self.img_open
        self.btn_flebeta["command"] = lambda arg1=5: self.openFile(arg1)#self.openFile(5)
        self.btn_flepredout["state"] = DISABLED
        self.btn_flepredout.grid(row=5, column=2,padx=3)

        self.chkbtn_pred["command"] = self.set_pred

        #-------------------------------------------run button------------------------------------
        # button: run
        self.btn_run = ttk.Button(self)
        self.btn_run["text"] = "Run"
        self.btn_run["width"] = 10
        self.btn_run["command"] = lambda arg1=1: self.run_mod(1) #
        self.btn_run.grid(row=5, column=3,padx=8,pady=3,sticky=E)





if __name__=='__main__':
    #print sys.path[0] + "\\img\\fleopen.png"
    #print sys.argv[0]

    root = Tk()
    root.title("pySIM: A Spatial Interaction Modelling Framework")

    app = mainGUI(master=root)
    app.mainloop()
    #root.destroy()