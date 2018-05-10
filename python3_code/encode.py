from numpy.random import shuffle

#Generate a random mapping
def random_mapping(alphabet):
	perm = alphabet.copy()
	shuffle(perm)
	return {alphabet[i]:perm[i] for i in range(len(alphabet))}

def random_encoding(plaintext, alphabet):
	mapping = random_mapping(alphabet)
	output = []
	for c in plaintext:
		output.append(mapping[c])
	return "".join(output)