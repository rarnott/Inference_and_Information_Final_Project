from itertools import permutations
import numpy as np
import random
from dist2 import DDist,binom

#########################################################################################################################################################

# Swaps values of a map
def swap(f, first_key, second_key):
	f_new = f.copy();
	f_new[first_key] = f[second_key]
	f_new[second_key] = f[first_key]
	return f_new

#########################################################################################################################################################

class Epsilon_Transition:
	#	Specifies a transition distribution of form:
	#		V(f'|f) = eps/(m choose 2) if f' differs by
	#			2 assignments from f, and 1-eps if f' = f
	#	Assumes f is a dictionary
	def __init__(self, alphabet, eps = 0.5):
		self.alphabet = alphabet
		self.eps = eps
		self.m = len(alphabet)
		self.p = eps/binom(self.m,2)

	#	Generate a sample from the transition distribution from f
	#	Uses transition dist and sample O(m^2)
	def sim_transition(self,f):
		x = random.random()
		if x <= self.eps:
			swap_keys = np.random.choice(self.alphabet, 2)
			f_prime = swap(f,swap_keys[0],swap_keys[1])
			return f_prime, self.p,self.p #probabilities symmetric under this model
		return f, 1-self.eps, 1-self.eps

#########################################################################################################################################################

class Uniform_Transition:
	#	Specifies a transition distribution of form:
	#		V(f'|f) = 1/(m choose 2) if f' differs by
	#			2 assignments from f
	#	Assumes dictionary
	def __init__(self, alphabet):
		self.alphabet = alphabet
		self.m = len(alphabet)
		self.p = 1./binom(self.m,2)

	#	Generate a sample from the transition distribution from f
	#	Uses transition dist and sample O(m^2)
	def sim_transition(self,f):
		swap_keys = np.random.choice(self.alphabet, 2)
		f_prime = swap(f,swap_keys[0],swap_keys[1])
		return f_prime, self.p,self.p #probabilities symmetric under this model