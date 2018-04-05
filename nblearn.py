import sys
import json

""" Class to represent a Naive Bayes model. """
class NaiveBayesModel():

	def __init__(self, lines):
		self.lines = lines
		self.labels = ["True Pos", "True Neg", "Fake Pos", "Fake Neg"]
		self.vocabulary = set()		# Known vocabulary
		self.X = []								# Training examples
		self.Y = []								# Training labels
		self.smoothing_value = 1	# Smoothing probabilities
		self.priors = {}					# P(c)
		self.likelihoods = {}			# P(w|c)
		self.stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', \
			'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', \
			'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', \
			"didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', \
			'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', \
			'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', \
			'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', \
			'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", \
			'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', \
			'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', \
			"she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', \
			"that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", \
			'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', \
			'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", \
			"we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', \
			"where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", \
			'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', \
			'yourself', 'yourselves']

	# Tokenize and preprocess training data
	def parseTrainingData(self):
		# Translator to remove punctuation
		punctuation = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
		translator = str.maketrans('', '', punctuation)

		# Parse lines from input file
		for line in self.lines:
			tokens = line.split()
			label = tokens[1] + " " + tokens[2]
			sentence = tokens[3:]

			# Remove punctuation, convert to lowercase, exclude stop words
			words = []
			for word in sentence:
				processed = word.translate(translator).lower().strip()
				if processed is "" or processed.isdigit() or processed in self.stopwords:
					continue
				words.append(processed)
				self.vocabulary.add(processed)

			self.X.append(words)
			self.Y.append(label)

	# Compute priors and likelihoods
	def train(self):
		num_labels = len(self.labels)
		num_vocab = len(self.vocabulary)
		num_examples = len(self.X)

		# Initialize counts
		label_counts = {}
		for label in self.labels:
			label_counts[label] = 0
		word_counts = {}
		for label in self.labels:
			word_counts[label] = {}
			for word in self.vocabulary:
				word_counts[label][word] = self.smoothing_value

		# Count word and class occurrences
		max_count = 0
		for i in range(num_examples):
			sentence = self.X[i]
			label = self.Y[i]
			label_counts[label] += 1
			for word in sentence:
				word_counts[label][word] += 1
				if word_counts[label][word] > max_count:
					max_count = word_counts[label][word]

		# Compute priors
		for label in self.labels:
			self.priors[label] = label_counts[label] / num_examples

		# Compute likelihoods
		for label in self.labels:
			total_count = sum(word_counts[label].values())
			self.likelihoods[label] = {}
			for word in self.vocabulary:
				self.likelihoods[label][word] = word_counts[label][word] / total_count

	# Writes the final model parameters to file
	def writeModelToFile(self):
		model = {'vocabulary': list(self.vocabulary), \
			'priors': self.priors, 'likelihoods': self.likelihoods}
		with open("nbmodel.txt", "w", encoding="utf-8") as file_object:
			file_object.write(json.dumps(model, ensure_ascii=False))

def main():
	# Get training data from file
	filename = sys.argv[1]
	with open(filename, "r", encoding="utf-8") as file_object:
		lines = file_object.readlines()

	# Train Naive Bayes model
	model = NaiveBayesModel(lines)
	model.parseTrainingData()
	model.train()
	model.writeModelToFile()


if __name__=="__main__":
	main()