import sys
import json

""" Class to classify text using Naive Bayes model. """
class NaiveBayesClassifier():

	def __init__(self, lines):
		self.lines = lines
		self.labels = ["True Pos", "True Neg", "Fake Pos", "Fake Neg"]
		self.vocabulary = set()		# Known vocabulary
		self.priors = {}					# P(c)
		self.likelihoods = {}			# P(w|c)
		self.identifiers = []			# Test example identifiers
		self.X = []								# Test examples
		self.Y = []								# Test classifications
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

	# Read model parameters from file
	def readModelFromFile(self):
		with open("nbmodel.txt", "r", encoding="utf-8") as file_object:
			model = json.load(file_object)
			self.vocabulary = model['vocabulary']
			self.priors = model['priors']
			self.likelihoods = model['likelihoods']

	# Tokenize and preprocess training data
	def parseTestData(self):
		# Translator to remove punctuation
		punctuation = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
		translator = str.maketrans('', '', punctuation)

		# Parse lines from input file
		for line in self.lines:
			tokens = line.split()
			identifier = tokens[0]
			sentence = tokens[1:]

			# Remove punctuation, convert to lowercase, exclude stop words
			words = []
			for word in sentence:
				processed = word.translate(translator).lower().strip()
				if processed is "" or processed.isdigit() or processed in self.stopwords:
					continue
				words.append(processed)

			self.identifiers.append(identifier)
			self.X.append(words)

	# Classify given test examples
	def classify(self):
		num_examples = len(self.X)

		# Classify each test example
		for i in range(num_examples):
			sentence = self.X[i]

			# Compute label probabilities
			label_probabilities = {}
			for label in self.labels:
				label_probabilities[label] = self.priors[label]
				for word in sentence:
					if word not in self.vocabulary:
						continue
					label_probabilities[label] *= self.likelihoods[label][word]

			# Assign label of maximum probability
			max_label = max(label_probabilities, key=lambda k: label_probabilities[k])
			self.Y.append(max_label)

	# Writes the final classifications to file
	def writeResultsToFile(self):
		num_examples = len(self.X)
		with open("nboutput.txt", "w", encoding="utf-8") as output_file:
			for i in range(num_examples):
				output_file.write(self.identifiers[i] + " " + self.Y[i] + "\n")


def main():
	# Get test data from file
	filename = sys.argv[1]
	with open(filename) as file_object:
		lines = file_object.readlines()

	# Test Naive Bayes model
	classifier = NaiveBayesClassifier(lines)
	classifier.readModelFromFile()
	classifier.parseTestData()
	classifier.classify()
	classifier.writeResultsToFile()


if __name__=="__main__":
	main()