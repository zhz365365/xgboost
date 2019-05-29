import xgboost as xgt
import numpy as np
class DMatrix:
	def __init__(self, _file=None):
		if _file == None:
			self._matrix = xgt.DMatrix(data=np.arange(0, 12).reshape((4, 3)), 
									   label=[1, 2, 3, 4], weight=[0.5, 0.4, 0.3, 0.2], silent=False, 
									   feature_names=['a', 'b', 'c'], feature_types=['int','int','float'],
									   nthread=2)
		else:
			self._matrix = xgt.DMatrix(_file)

	def print_features(self):
		print('feature_names:%s' % self._matrix.feature_names)
		print('feature_types:%s' % self._matrix.feature_types)

	def run_get(self):
		print('get_base_margin():', self._matrix.get_base_margin())
		print('get_label():', self._matrix.get_label())
		print('get_weight():', self._matrix.get_weight())
		print('num_col():', self._matrix.num_col())
		print('num_row():', self._matrix.num_row())

def test():
	# matrix = DMatrix('../data/train.svm.txt')
	matrix = DMatrix()
	print('查看 matrix :')
	matrix.print_features()
	# feature_names:['f0', 'f1', 'f2']
	# feature_types:None
	# or
	# feature_names:['a', 'b', 'c']
	# feature_types:['int', 'int', 'float']

	print('\n查看 matrix get:')
	matrix.run_get()
	# get_base_margin(): []
	# get_label(): [1. 1. 1. 1. 0. 0. 0. 0.]
	# get_weight(): []
	# num_col(): 3
	# num_row(): 8
	# or
	# get_base_margin(): []
	# get_label(): [1. 2. 3. 4.]
	# get_weight(): [0.5 0.4 0.3 0.2]
	# num_col(): 3
	# num_row(): 4

if __name__ == '__main__':
	test()
