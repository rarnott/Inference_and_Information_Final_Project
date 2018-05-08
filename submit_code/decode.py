from itertools import permutations
import numpy as np
import random
from transitions2 import Epsilon_Transition,Uniform_Transition
from dist2 import DDist,binom
from multiprocessing import Pool, TimeoutError

#########################################################################################################################################################

#	Helper function to generate a mapping describing a permutation,
#		from the alphabet and the permutation
def perm_to_mapping(permutation, alphabet):
	return {alphabet[i]:permutation[i] for i in range(len(alphabet))}

#	Return the inverse of the permutation f
#	Naive: iterate over letters, find what they map to, create the reverse mapping, O(m)
def invert_mapping(f, alphabet):
	inverted_mapping = {}
	for a in alphabet:
		a_prime = f[a]
		inverted_mapping[a_prime]=a
	return inverted_mapping

#	Helper function to get the accuracy of a mapping on a sequence
def accuracy(f, alphabet, encoded_seq, decoded_seq):
	assert len(encoded_seq) == len(decoded_seq)
	correct = 0
	for i in range(len(encoded_seq)):
		if encoded_seq[i] == f[decoded_seq[i]]:
			correct += 1
	return correct/len(encoded_seq)

#########################################################################################################################################################

class AlphabetDist:
	# 	Defines a distribution over elements in the alphabet,
	#		a conditional distribution conditioned on a letter
	#		in the alphabet, and a likelihood function for an 
	#		encoded string given an encoding
	#	Assumes f is a table
	def __init__(self, floor=-10, letter_dist=None, transition_matrix=None):
		self.letter_dist = letter_dist
		self.tranisition_matrix = transition_matrix #function
		self.alphabet = letter_dist.alphabet() if letter_dist else None
		self.letter_to_index = {self.alphabet[i]:i for i in range(len(self.alphabet))} if letter_dist else None
		self.floor = floor

	def handle_inf(self,x):
		return x if x != -np.inf else self.floor

	#	Generates the letter_dist from a csv
	def generate_dists(self, alphabet_file, letter_file, matrix_file):
		alphabet = np.genfromtxt(alphabet_file,delimiter=",",dtype="unicode")
		letter_prob = np.genfromtxt(letter_file,delimiter=",")
		matrix = np.genfromtxt(matrix_file,delimiter=",")
		self.alphabet = alphabet
		self.letter_to_index = {self.alphabet[i]:i for i in range(len(self.alphabet))}
		self.letter_dist = DDist({alphabet[i]:letter_prob[i] for i in range(len(alphabet))})
		self.transition_matrix = matrix

	#	Returns the probability of observing x
	def prob_letter(self, x):
		return self.handle_inf(np.log(self.letter_dist.prob(x)))

	#	Returns the probability of observing x_prime after x
	def conditional_prob(self, x, x_prime):
		return self.handle_inf(np.log(self.transition_matrix[self.letter_to_index[x_prime],self.letter_to_index[x]]))

	#	Returns the conditional distribution of x_prime given x
	#	O(m)
	def conditional_dist(self,x):
		index = self.letter_to_index[x]
		probabilities = self.transition_matrix[:,index]
		return DDist({self.alphabet[i]:probabilities[i] for i in range(len(self.alphabet))})

	#	Returns the likelihood of an encoded sequence given the encoding table f
	#	creates the inverted table, computes product of conditional probabilities
	# 	O(max(m, |encoded_seq|))
	def log_likelihood(self, encoded_seq, f):
		n = len(encoded_seq)
		f_inv = invert_mapping(f,self.alphabet)
		decoded_seq = [f_inv[x] for x in encoded_seq]
		output = self.prob_letter(decoded_seq[0])
		for i in range(1,n):
			output = output + self.conditional_prob(decoded_seq[i-1],decoded_seq[i])
		return output

	#	Returns a mapping that results from doing frequency analysis on the ciphertext
	def frequency_analysis(self,ciphertext):
		n = len(ciphertext)
		empirical_counts = {letter:0 for letter in self.alphabet}
		for letter in ciphertext:
			empirical_counts[letter] += 1
		empirical_distribution = sorted(list({letter:float(empirical_counts[letter])/n for letter in self.alphabet}.items()), 
			key=lambda x: x[1])
		true_distribution = sorted(list(self.letter_dist.to_dictionary().items()), 
			key=lambda x: x[1])
		frequency_mapping = {}
		for i in range(len(true_distribution)):
			frequency_mapping[true_distribution[i][0]] = empirical_distribution[i][0]
		return frequency_mapping


#########################################################################################################################################################

class MCMC:
	# - Either need an initial state, and a set of states
	#		or an alphabet (to pick a random state, and generate states)
	# - Storing all possible states is computationally intensive, instead 
	#		have a function that generates possible transitions
	#
	#	State representation: table
	def __init__(self, target, proposal_dist, alphabet, floor=-10):
		self.target = target
		self.proposal_dist = proposal_dist
		self.alphabet = alphabet
		self.m = len(self.alphabet)
		self.log_likelihood_floor = floor

	#	Performs 1 step of the metropolis-hastings algorithm
	#	returns the next state in the sequence, true if the sample was accepted, false otw,
	#		and the log_likelihood 
	def step(self, sequence, state, verbose=False):
		proposal, prob1, prob2 = self.proposal_dist.sim_transition(state)
		p_x = self.target.log_likelihood(sequence, state)
		p_xp = self.target.log_likelihood(sequence, proposal)
		if verbose:
			print state, proposal, prob1, prob2, p_x, p_xp
		log_a = np.log(prob1) + p_xp - p_x - np.log(prob2)
		a = min(0, log_a)
		if np.log(random.random()) < a:
			return proposal, True, p_xp
		else:
			return state, False, p_x

	# 	Performs the metropolis hastings algorithm for n_iters iterations,
	#		discarding the first burn_in samples.
	#	Takes encoded sequence as input
	#	Returns samples
	def metropolis_hastings(self, ciphertext, sequence_length=250, n_epochs=10, n_iters = 100):
		# assert burn_in <= n_iters
		def gen_sequence():
			if sequence_length != len(ciphertext):
				start_index = np.random.choice(len(ciphertext)-sequence_length+1)
				return ciphertext[start_index:start_index+sequence_length+1]
			return ciphertext
		current_state=None
		sequence=gen_sequence()
		current_state = self.target.frequency_analysis(ciphertext)
		# perm = np.random.permutation(self.alphabet).tolist()
		# current_state = {self.alphabet[i]:perm[i] for i in range(self.m)}
		samples=[]
		acceptances=[]
		log_likelihoods=[]
		max_likelihood_sample = current_state
		for epoch in range(n_epochs):
			n_accept = 0
			for t in range(n_iters):
				sequence=gen_sequence()
				verbose=False
				current_state,accepted,log_likelihood = self.step(sequence, current_state,verbose)
				samples.append(current_state)
				acceptances.append(accepted)
				log_likelihoods.append(log_likelihood)
				n_accept += accepted
				if current_state == max_likelihood_sample: continue
				if log_likelihood > self.target.log_likelihood(ciphertext,max_likelihood_sample):
				# if self.target.log_likelihood(ciphertext,current_state) > max_likelihood:
					max_likelihood_sample=current_state
					max_likelihood = log_likelihood
				
		overall_likelihood = self.target.log_likelihood(ciphertext,max_likelihood_sample)
		print "Overall Max Likelihood:", overall_likelihood
		return samples, acceptances, max_likelihood_sample, overall_likelihood

def unwrapped_metropolis_hastings(obj, ciphertext,sequence_length=250, n_epochs=10, n_iters = 100):
	return MCMC.metropolis_hastings(obj,ciphertext, sequence_length, n_epochs, n_iters)

#########################################################################################################################################################

def decode(ciphertext, output_file_name):
	floor=-10
	english = AlphabetDist(floor=floor)
	english.generate_dists("alphabet.csv", "letter_probabilities.csv", "letter_transition_matrix.csv")
	m = len(english.alphabet)
	e = 1.-1./binom(m,2)
	eps_trans = Epsilon_Transition(english.alphabet,eps=e)
	mcmc_test = MCMC(english,eps_trans,english.alphabet,floor=floor)
	n_iters=10
	n_epochs=300
	p = 0.1
	n = len(ciphertext)
	n_starts = 3
	pool = Pool(processes=5)
	multi_start = [pool.apply_async(unwrapped_metropolis_hastings, (mcmc_test,ciphertext, 
			int(p*n), n_epochs, n_iters)) for i in range(n_starts)]
	results = [res.get(timeout=10000) for res in multi_start]
	best_result = max(results,key=lambda x: x[3])
	decoder = invert_mapping(best_result[2],english.alphabet)
	best_ll = best_result[3]
	decoded_text = []
	for w in ciphertext:
		decoded_text.append(decoder[w])
	f = open(output_file_name,'w')
	f.write("".join(decoded_text))
	f.close()

#########################################################################################################################################################

if __name__ == "__main__":
	output = open("output.txt","w+")
	output.close()
	cipher_file = open("../ciphertext.txt","r")
	ciphertext = cipher_file.read().replace("\n", "")
	decode(ciphertext,"output.txt")
	plain_file = open("../plaintext.txt","r")
	plaintext = plain_file.read().replace("\n","")
	
	#find accuracy
	decoded = open("output.txt","r")
	decoded_text = decoded.read().replace("\n","")
	decoded.close()
	n_correct = 0
	for i in range(len(decoded_text)):
		if decoded_text[i] == plaintext[i]:
			n_correct += 1
	print float(n_correct)/len(decoded_text)