import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import gensim
import itertools

### LOADING THINGS ###
# Load the word embeddings
def load_embeddings(file_path):
    embeddings = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    
    return embeddings

def get_embedding(word, embeddings, word2vec_model):
    if word in embeddings:
        # Return pre-existing embedding from the dictionary for single words
        return embeddings[word]
    else:
        # For multi-word phrases, split and average the embeddings of individual words
        words_in_phrase = word.split()  # Split phrase into words
        embeddings_list = []
        for w in words_in_phrase:
            if w in word2vec_model:  # Check if the word exists in the Word2Vec model
                embeddings_list.append(word2vec_model[w])
        if embeddings_list:
            # Average the embeddings of the words in the phrase
            return np.mean(embeddings_list, axis=0)
        else:
            # If no embeddings are found (e.g., out-of-vocabulary), return a zero vector
            return np.zeros(word2vec_model.vector_size)

def compute_density(G, group):
    pairs = itertools.combinations(group, 2)  # Generate all unique pairs of nodes in the group
    similarities = []
    for word1, word2 in pairs:
        if G.has_edge(word1, word2):
            similarities.append(G[word1][word2]['weight'])
    return sum(similarities) / len(similarities) if similarities else 0

def compute_conductance(G, group, all_words):
    outsiders = [word for word in all_words if word not in group]  # Get nodes not in the group
    conductances = []
    for word_in_group in group:
        for word_outside in outsiders:
            if G.has_edge(word_in_group, word_outside):
                conductances.append(G[word_in_group][word_outside]['weight'])
    return sum(conductances) / len(conductances) if conductances else 0

def compute_rank(G, combo, connections):
    return compute_density(G, combo) + (1 - compute_conductance(G, combo, connections))


# Generate guesses
def generateGuesses(G, connections):
    combinations_of_4 = list(itertools.combinations(connections, 4))
    combos = []
    # Print the combos
    for combo in combinations_of_4:
        combos.append(combo)
    
    # Rank combinations
    combinations = dict()

    for combo in combos:
        # compute rank
        rank = compute_rank(G, combo, connections)

        #add to dictionary
        combinations[combo] = rank

    sorted_combinations = dict(sorted(combinations.items(), key=lambda item: item[1], reverse=True))

    return sorted_combinations

### LOADING MORE THINGS ###
# path to ConceptNet Numberbatch embeddings file
embeddings_file = '../datasets/numberbatch.txt'
# Load the embeddings into a dictionary
embeddings = load_embeddings(embeddings_file)
	
# Load Word2Vec Model
model_path='../GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
	"""
	_______________________________________________________
	Parameters:
	words - 1D Array with 16 shuffled words
	strikes - Integer with number of strikes
	isOneAway - Boolean if your previous guess is one word away from the correct answer
	correctGroups - 2D Array with groups previously guessed correctly
	previousGuesses - 2D Array with previous guesses
	error - String with error message (0 if no error)

	Returns:
	guess - 1D Array with 4 words
	endTurn - Boolean if you want to end the puzzle
	_______________________________________________________
	"""

	# Your Code here
    
	### CREATE GRAPH OF WORDS
	# Create an empty graph
	G = nx.Graph()

	# Add nodes (words) to the graph
	G.add_nodes_from(words)

	# Compute the embeddings for each word or phrase
	embedding_matrix = np.array([get_embedding(word, embeddings, word2vec_model) for word in words])

	# Compute the cosine similarity matrix between word embeddings
	cos_sim_matrix = cosine_similarity(embedding_matrix)

	# Add edges between words based on cosine similarity
	for i, word1 in enumerate(words):
		for j, word2 in enumerate(words):
			if i < j:  # Ensure each pair is considered once
				similarity = cos_sim_matrix[i, j]
				print(word1, word2)
				print(similarity)
				G.add_edge(word1, word2, weight=similarity)
                    
	# ### GAME START ###
    
	# # If it's one away, find the best combination and rank by adding a new word
	# if isOneAway:
	# 	# Find the weakest link and the best new combination
	# 	best_subset, weakest_link = find_highest_rank(G, words)
	# 	best_combination, rank = find_best_addition(G, best_subset, words, weakest_link, previousGuesses)
	# 	guess = best_combination
	
	# # If it's not one away, just make the best guess from the highest ranked combinations
	# else:
	# 	sorted_combinations = generateGuesses(G, words)  # Get the ranked combinations of 4 words
	# 	highest_rank_combo = list(sorted_combinations.keys())[0]  # Get the highest rank combination
	# 	guess = list(highest_rank_combo)  # Convert tuple to list for the final guess
	# 	endTurn = False  # You can set endTurn logic if needed, based on game state (e.g., if strikes are hit)

	# # Check if the correct group has been guessed already (optional)
	# if correctGroups:
	# 	endTurn = True  # If guessed correctly, end the turn	



	# Good Luck!

	# Example code where guess is hard-coded
	combinations = generateGuesses(G, words)
	guess = next(iter(combinations))
	#guess = ["apples", "bananas", "oranges", "grapes"] # 1D Array with 4 elements containing guess
	endTurn = False # True if you want to end puzzle and skip to the next one

	return guess, endTurn


### FUNCTIONS ###
# Find the subset with the highest rank when one word is removed
def find_highest_rank(G, connections):
    max_density = float('-inf')
    max_conductance = float('-inf')
    best_subset = None
    weakest_link = None
    
    # Generate all combinations of removing one word (subsets of 3)
    for subset in itertools.combinations(connections, len(connections) - 1):
        density = compute_density(G, subset)
        conductance = compute_conductance(G, subset, connections)
        if density > max_density or conductance > max_conductance:
            max_density = density
            max_conductance = conductance
            best_subset = subset
            # The weakest link is the word removed to form the current subset
            weakest_link = list(set(connections) - set(subset))[0]
    
    return best_subset, weakest_link

# Function to find the highest rank by adding one unused word to the best subset
def find_best_addition(G, best_subset, connections, weakest_link, previous_guesses):
    unused_words = [word for word in connections if word not in best_subset and word != weakest_link]
    
    max_density = float('-inf')
    max_conductance = float('-inf')
    best_combination = None
    
    for word in unused_words:
        new_group = list(best_subset) + [word]
        density = compute_density(G, best_subset)
        conductance = compute_conductance(G, best_subset, connections)
        if (density > max_density or conductance > max_conductance) and (sorted(new_group) not in [sorted(previous_guess) for previous_guess in previous_guesses]):
            max_density = density
            max_conductance = conductance
            max_rank = density + (1-conductance)
            best_combination = new_group
    
    return best_combination, max_rank

def isOneAway(G, guess, new_words, previous_guesses):
    weakest_link_removed, weakest_link = find_highest_rank(G, next(iter(guess)))
    best_combination, rank = find_best_addition(G, weakest_link_removed, new_words, weakest_link, previous_guesses)

    return best_combination, rank