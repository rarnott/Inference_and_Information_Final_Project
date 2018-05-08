from itertools import permutations
import numpy as np
import random
import math

def binom(n,m):
	return math.factorial(n)/(math.factorial(m)*math.factorial(n-m))

class DDist:
	#	Specifies a Discrete probability distribution
	def __init__(self, dic):
		self.dic = dic
		self.alphabet = list(dic.keys())

	def prob(self,x):
		return self.dic[x] if self.dic[x] else 0

	def sample(self):
		return np.random.choice(self.alphabet, p = list(self.dic.values()))

	def to_dictionary(self):
		return self.dic.copy()