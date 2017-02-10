from threading import Thread
import numpy as np
class CurveReducer(Thread):
	"""docstring for CurveReducer"""
	def __init__(self, parent=None, curveData=None, reduceTo=500, extra=None):
		super(CurveReducer, self).__init__()
		self.parent = parent
		self.curveData = list(curveData)
		self.reduceTo = reduceTo
		self.extraData = extra

	def run(self):
		newCurve = [[],[],[]]
		if( len(self.curveData) <= self.reduceTo ):
			tmpArray = np.array(self.curveData)
			tmpArray = tmpArray[np.newaxis, ...]
			newCurve = np.concatenate(( tmpArray, tmpArray, tmpArray)).T
			if( not self.parent == None):
				self.parent.thread_callback( curve=newCurve.tolist(), extra=self.extraData)	

		else:
			step = len(self.curveData) / self.reduceTo



			for i in range(self.reduceTo):
				slices = self.curveData[i*step:(i+1)*step]
				mean = np.mean(slices)
				std = np.std(slices)
				newCurve.append( [mean, mean + std, mean - std])

			remainder = len(self.curveData) % self.reduceTo
			if( remainder != 0 ):
				slices = self.curveData[-remainder:]
				mean = np.mean(slices)
				std = np.std(slices)
				newCurve.append( [mean, mean + std, mean - std])



			if( not self.parent == None):
				self.parent.thread_callback( curve=newCurve, extra=self.extraData)

		return


						