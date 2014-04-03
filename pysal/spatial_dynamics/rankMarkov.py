"""
Rank-based Markov chains for spatial distribution
dynamics.
"""

_author_ = "Serge Rey <sjsrey@gmail.com>, Wang Sizhe <wsizhe@asu.edu, saguswang@gmail.com>"

import sys
import copy
import pysal
import numpy as np

class RankMarkov:
	'''
	This is a base class for full rank Markov and geographical rank Markov. 
	It provides some general properties and functions.
	'''
	def __init__(self, stMat, timeRes = 1):
		self._st_matrix = stMat
		self._time_resolution = timeRes
		self._nRanks = stMat.shape[0]
		self._nSample= stMat.shape[1]
		self._rank_matrix = None

	@property
	def rankMat(self):
		'''
		property: retrieve the rank matrix
		'''
		if self._rank_matrix is None:
			self._rank_matrix = self._rankMatrix()
		return self._rank_matrix

	def _rankMatrix(self):
		'''
		This method converts the raw data to a rank matrix
		'''
		data = self._st_matrix.T
		ranks = np.array([pysal.Quantiles(y, self._nRanks).yb for y in data])
		#reverse all ranks, make the higher ranks represent the greater values
		for i in xrange(ranks.shape[0]):
			for j in xrange(ranks.shape[1]):
				ranks[i, j] = self._nRanks - 1 - ranks[i, j]
		return ranks

	def _randList(self):
		'''
		This method makes a list, in which numbers from 1 to n, n is he number of regions
		in data file, are randomly distributed.
		'''
		n = self._nRanks
		ret = np.random.rand(n)
		ret = pysal.Quantiles(ret, n)
		return ret

	def _randTransMat(self, timeSpan):
		'''
		this method makes two random matrices, one is of transition matrix, one is of 
		symmetric transition matrix.
		'''
		nstates = self._nRanks

		#initial two zero matrix 
		transMatrix = np.zeros((nstates,nstates), dtype=int)
		symTransMat = np.zeros((nstates,nstates), dtype=int)

		#timeSpan means the time range in data you provides, for instance the annual data
		#from 1929-2009, the timeSpan=80. It decides how many times we simulate the inter-
		#regional transition
		for n in xrange(timeSpan):
			rs = range(nstates)
			cs = range(nstates)
			index = [0] * nstates
			for i in xrange(nstates - 1, -1, -1):
				#r is for the code of a region that the transition start from in time t
				#c is for the code of a region that the transition end with in time t+1
				r = rs[np.random.random_integers(0, i)]
				rs.remove(r)
				c = cs[np.random.random_integers(0, i)]
				cs.remove(r)
				transMatrix[r, c] += 1
				index[r] = c
				if index[c] == r:
					symTransMat[c, r] += 1
					symTransMat[r, c] += 1
		return {'asym':transMatrix, 'sym':symTransMat}
		
class FullRankMarkov(RankMarkov):	
	'''
	This class derives from class RankMarkov. It generates and holds the matrices 
	of full	rank Markov. 
	'''
	def __init__(self, stMat, timeRes = 1):
		RankMarkov.__init__(self, stMat, timeRes)
		self._f_trans_mat = None
		self._f_fmpt_mat = None

	def _fullRankMarkov(self):
		'''
		This method generates the transition matrix using full rank Markov
		'''
		markov = pysal.Markov(self.rankMat.transpose())
		self._f_trans_mat = markov.p.tolist()

	@property
	def TransMat(self):
		'''
		property: retrieve the transition matrix of full rank Markov
		'''
		if self._f_trans_mat is None:
			self._fullRankMarkov()
		return self._f_trans_mat

	@property
	def FMPTMat(self):
		'''
		property: retrieve the first mean passage times matrix of full rank Markov
		'''
		if self._f_fmpt_mat is None:
			self._f_fmpt_mat = pysal.ergodic.fmpt(np.matrix(self.TransMat)) * self._time_resolution
		return self._f_fmpt_mat

class GeographicRankMarkov(RankMarkov):
	'''
	This class derives from class RankMarkov. It generates and holds the matrices 
	of geographical	rank Markov. It also provide the function to do the spatial 
	dynamics test. 
	'''
	def __init__(self, stMat, timeRes = 1):
		RankMarkov.__init__(self, stMat, timeRes)
		self._g_trans_times_mat = None
		self._g_trans_mat = None
		self._g_fmpt_mat = None
		self._sym_trans_mat = None

	@property
	def TransMat(self):
		'''
		property: retrieve the transition matrix of geographical rank Markov
		'''
		if self._g_trans_mat is None:
			self._geoRankMarkov()
		return self._g_trans_mat

	@property
	def TransTimesMat(self):
		'''
		property: retrieve the raw transition matrix of geographical rank Markov, 
		which isn't divided by the number of columns.
		'''
		if self._g_trans_times_mat is None:
			self._geoRankMarkov()
		return self._g_trans_times_mat

	@property
	def symTransMat(self):
		'''
		property: retrieve the symmetric transition matrix of geographical rank Markov
		'''
		if self._sym_trans_mat is None:
			self._geoRankMarkov()
		return self._sym_trans_mat

	@property
	def FMPTMat(self):
		'''
		property: retrieve the first mean passage times matrix of geographical rank Markov
		'''
		if self._g_fmpt_mat is None:
			self._g_fmpt_mat = pysal.ergodic.fmpt(np.matrix(self.TransMat)) * self._time_resolution
		return self._g_fmpt_mat

	def _geoRankMarkov(self):
		'''
		This method generates the transition matrix, raw transition matrix, and 
		symmetric transition matrix using geographical rank Markov
		'''
		rm = self.rankMat
		nstates = self._nRanks

		#generate raw transition matrix and symmetric transition matrix together
		transMatrix = np.zeros((nstates, nstates))
		self._sym_trans_mat = np.zeros((nstates, nstates))
		for k in xrange(len(rm) - 1):
			for i in xrange(nstates):
				for j in xrange(nstates):
					if rm[k, i] == rm[k+1, j]:
						transMatrix[i, j] += 1
						if rm[k, j] == rm[k+1, i]:
							self._sym_trans_mat[i, j] += 1
						break
		self._g_trans_times_mat = copy.copy(transMatrix)
		
		#generate transition matrix
		for i in transMatrix:
			total = sum(i)
			for j in xrange(len(i)):
				i[j] /= float(total)
		self._g_trans_mat = transMatrix

	@staticmethod
	def _localRankMarkovTest(ttm, nb, normalize=False):
		'''
		This method calculate all the result of local spatial dynamics test, 
		using transition matrix and neighbor data from spatial weight. The 
		results are held in a list.
		'''
		return np.array([sum([ttm[i][j] for j in nb[i]]) / (normalize and len(nb[i]) or 1) for i in xrange(len(ttm))])
	
	def rankMarkovTest(self, w, simTimes=999):
		'''
		This method do the spatial dynamics test using Monte Carlo method
		and four test standards, origin based, destination based, symmetric,
		and hybrid.
		'''

		n = self._nRanks

		#get the spatial weight information
		_nb = w.neighbors
		nb = _nb.values()

		timeSpan = int(sum(self.TransTimesMat[0]))
		
		#get the result of local spatial dynamics test based on origin
		#based standard
		ttm = self.TransTimesMat
		pv_ori  = np.array([1.0] * n)
		rmt_ori = GeographicRankMarkov._localRankMarkovTest(ttm, nb)

		#get the result of local spatial dynamics test based on destinaiton
		#based standard
		ttm = np.array(ttm).transpose()
		pv_dst  = np.array([1.0] * n)
		rmt_dst = GeographicRankMarkov._localRankMarkovTest(ttm, nb)

		#get the result of local spatial dynamics test based on symmetric
		#standard
		pv_sym = np.array([1.0] * n)
		rmt_sym = GeographicRankMarkov._localRankMarkovTest(self.symTransMat, nb)

		#get the result of local spatial dynamics test based on hybrid
		#standard
		pv_hyb  = np.array([1.0] * n)
		rmt_hyb = rmt_dst + rmt_ori - rmt_sym

		#get the global test results based on different standards
		globalRMT = sum(rmt_ori)
		pv_globle = 1.0
		globalRMT_sym = sum(rmt_sym)
		pv_globle_sym = 1.0
		globalRMT_hyb = sum(rmt_hyb)
		pv_globle_hyb = 1.0

		np.random.seed(512)

		#simulation process
		for i in xrange(simTimes):
			rtm = self._randTransMat(timeSpan)
			sym = rtm['sym']
			asym = rtm['asym']

			rtl_o = GeographicRankMarkov._localRankMarkovTest(asym, nb)
			rtl_d = GeographicRankMarkov._localRankMarkovTest(asym.transpose(), nb)
			rtl_s = GeographicRankMarkov._localRankMarkovTest(sym, nb)
			rtl_h = rtl_o + rtl_d - rtl_s

			pv_ori += rtl_o > rmt_ori
			pv_dst += rtl_d > rmt_dst
			pv_sym += rtl_s > rmt_sym
			pv_hyb += rtl_h > rmt_hyb

			pv_globle += sum(rtl_o) > globalRMT
			pv_globle_sym += sum(rtl_s) > globalRMT_sym
			pv_globle_hyb += sum(rtl_h) > globalRMT_hyb

			sys.stdout.write("Rank-Markov-based Test, simulating using Monte Carlo Method: [%2d%%]\r" % ((i + 1) * 100 / simTimes))
			sys.stdout.flush()
		pv_ori /= simTimes + 1
		pv_dst /= simTimes + 1
		pv_sym /= simTimes + 1
		pv_hyb /= simTimes + 1
		pv_globle /= simTimes + 1
		pv_globle_sym /= simTimes + 1
		pv_globle_hyb /= simTimes + 1

		#organize all the result in to dictionary
		_rmt_results = dict([
			('rmt_ori', rmt_ori),
			('pv_ori' , pv_ori),
			('rmt_dst', rmt_dst),
			('pv_dst' , pv_dst),
			('rmt_sym', rmt_sym),
			('pv_sym' , pv_sym),
			('rmt_hyb', rmt_hyb),
			('pv_hyb' , pv_hyb),
			('global' , globalRMT),
			('p-value', pv_globle),
			('global_s' , globalRMT_sym),
			('p-value_s', pv_globle_sym),
			('global_h' , globalRMT_hyb),
			('p-value_h', pv_globle_hyb)
		])

		return _rmt_results

if __name__ == "__main__":

	f = pysal.open(pysal.examples.get_path('usjoin.csv'))
	m = np.array([f.by_row[i][2:] for i in xrange(48)])
	grm = GeographicRankMarkov(m)
	print grm.rankMat
	print grm.FMPTMat
	w = pysal.rook_from_shapefile('../examples/us48.shp')
	print grm.rankMarkovTest(w, 99)
