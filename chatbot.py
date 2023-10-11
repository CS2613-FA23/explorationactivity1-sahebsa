# To run the code - python chatbot.py METHOD
# where METHOD is one of overlap or w2v. Program will need to read guteberg.txt and cc.en.300.vec.10k. It should look for these files in
#the same directory as the source code.

#Implementing a corpus-based chatbot based on ideas from information retrieval and vector semantics to analyze chatbot’s behavior
#Information about the corpus - gutenburg.txt Each line in this file is a sentence from a book in the NLTK Gutenberg corpus.It includes works from Jane Austen, Lewis Carroll, and William Shakespeare, among others.The sentences are pruned to avoid very short, and very long, sentences. There are roughly 15k lines in the file. It is UTF-8 encoded.
#cc.en.300.vec.10k fastText word embeddings for English. These are vectors for the 10k most frequent alphabetic, lower-case words.

from __future__ import print_function, division
from collections import defaultdict as ddict
from math import log, exp, sqrt
from sys import argv as arg
import re,sys

class Chatbot(object):
    # A simple tokenizer. Applies case folding  
    @staticmethod
    def tokenize(s):
        tokens = s.lower().split()
        trimmed_tokens = []
        for t in tokens:
            if re.search('\w', t):
                # t contains at least 1 alphanumeric character
                t = re.sub('^\W*', '', t)  # Leading non-alphanumeric chars
                t = re.sub('\W*$', '', t)  # Trailing non-alphanumeric chars
            trimmed_tokens.append(t)
        return trimmed_tokens

class Overlap(Chatbot):
    # Find the most similar response in terms of token overlap.
    def __init__(self, responses, vectors=None):
        self.responses_types = []
        with open(responses) as f:
            for res in f:
                self.responses_types.append((res, set(self.tokenize(res))))

    def most_sim_overlap(self, query):
        q_tokenized = self.tokenize(query)
        max_sim = 0
        max_resp = "Sorry, I don't understand"
        for res, r_tokenized_set in self.responses_types:
            sim = len(r_tokenized_set.intersection(q_tokenized))
            if sim > max_sim:
                max_sim, max_resp = sim, res
        return max_resp

class W2V(Chatbot):
    def __init__(self, responses, vectors):
        self.type_vectors = ddict(int)  
        self.res_nvec = {} 
        self.load_vectors(vectors)
        self.normalize_responses(responses)

    def load_vectors(self, fname):
        # Code for loading the fasttext (word2vec) vectors from here (lightly
        # modified): https://fasttext.cc/docs/en/crawl-vectors.html
        for line in open(fname):
            tkns = line.rstrip().strip().split(' ')
            self.type_vectors[tkns[0]] = tuple((float(i) for i in tkns[1:]))

    def mag(self, vec):
        return sqrt(sum((x * x for x in vec)))

    def sum_vectors(self, vecs):
        return tuple((sum(i) for i in zip(*vecs)))

    def mul_vectors(self, vec1, vec2):
        return sum((i * j for i, j in zip(vec1, vec2)))

    def div_vectors(self, vec, denom):
        return tuple((i / denom for i in vec))

    def normalize_doc(self, doc):
        tok_vec = []
        n = 0
        for tok in self.tokenize(doc):
            vec = self.type_vectors[tok]
            n += 1
            if vec != 0:
                mag = self.mag(vec) 
                nvec = self.div_vectors(vec, mag)
                tok_vec.append(nvec)
        if len(tok_vec) != 0:
            return self.div_vectors(self.sum_vectors(tok_vec), n)
        else:
            return False

    def normalize_responses(self, responses):
        with open(responses) as f:
            for res in f:
                n_doc = self.normalize_doc(res)
                if n_doc:
                    self.res_nvec[res] = n_doc

    def cosine(self, res_vec, query_vec):
        numer = self.mul_vectors(res_vec, query_vec)
        denom = self.mag(res_vec) * self.mag(query_vec)
        cos = numer / denom
        assert -1.0 <= round(cos, 2) <= 1.0, "Cosine value error..!!"
        return cos

    def most_sim_overlap(self, query):
        qVec = self.normalize_doc(query)
        if not qVec:
            return "Sorry, I don't understand", 0.0
        most_sim_overlap = {}
        for res, nvec in self.res_nvec.items():
            most_sim_overlap[res] = self.cosine(nvec, qVec)
        return max(most_sim_overlap, key=lambda x: most_sim_overlap[x]), max(most_sim_overlap.values())


if __name__ == "__main__":
    # Method will be one of 'overlap' or 'w2v'
    method = sys.argv[1]
    responses = 'gutenberg.txt'
    vectors = 'cc.en.300.vec.10k'
    user = input("Please Enter your name: \n")
    if method == "both":
        overlap = Overlap(responses, vectors)
        w2v = W2V(responses, vectors)
        print("\nBot: Hi! Let's chat\n")
        while True:
            query = input(user+ ": ")
            omost_sim_overlap = overlap.most_sim_overlap(query)
            vq, vr = w2v.normalize_doc(query), w2v.normalize_doc(omost_sim_overlap)
            ocos = w2v.cosine(vr, vq) if vq else 0.0
            print("Overlap: %s -(%s)" % (omost_sim_overlap.strip(), round(ocos, 3)))

            w2vmost_sim_overlap, w2vcos = w2v.most_sim_overlap(query)
            print("W2V: %s -(%s)" % (w2vmost_sim_overlap.strip(), round(w2vcos, 3)))
            print()

    elif method == "overlap":
        bot = Overlap(responses, vectors)
        print("\nOverlap: Hi! Let's chat")
        while True:
            query = input(user+ ": ")
            most_sim_overlap = bot.most_sim_overlap(query)
            print("%s: %s" % (arg[1].capitalize(), most_sim_overlap.strip()))

    # Initialization for the w2v method if we're using it
    elif method == "w2v":
        bot = W2V(responses, vectors)
        print("\nw2v: Hi! Let's chat")
        # Building response vectors here
        while True:
            query = input(user+ ": ")
            most_sim_overlap, cos = bot.most_sim_overlap(query)
            print("%s: %s" % (arg[1].capitalize(), most_sim_overlap.strip()))
    print()

# Given a user turn (i.e., a “user query” as in 24.1 of the textbook), chatbot will respond with the most similar line (i.e., response / system turn) in the Guteberg corpus (i.e., guteberg.txt). Represents the user turn, and all of the lines from the Guteberg corpus, using the vector-based approach to turn representation described below.
# Turn Representations- Approaches to learning word embeddings, like word2vec and fastText, give us a vector for each
# word (type). One way to represent the meaning of a (short) document is to simply add together
# the vectors for the words in the document. In this approach it’s often helpful to normalize the length of the vectors before adding them. 
# If a word (token) in the text does not have a corresponding vector, just ignore it in forming the
# representation of the text. If all words in the user query do not have a vector, and as such a vec#tor
# representing this turn cannot be formed, the output is a default response (e.g., “I’m sorry?”, “I don’t understand”). 
