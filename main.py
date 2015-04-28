import time
import csv
import nltk
import numpy as np

terms = []
docs = []
nlimit = 1000

termsFilename = "data/ajeoterms.txt"
docsFilename = "data/dblp.csv"

# Reading terms
t1 = time.time()
with open(termsFilename, 'r') as termsFile:
    for term in termsFile.readlines():
        term = term.split("\n")[0].split("\r")[0]
        terms.append(term)
t2 = time.time()

# Reading docs
with open(docsFilename, 'r') as docsFile:
    publicationsReader = csv.reader(docsFile, delimiter=',')
    for row in publicationsReader:
        docs.append(row[1])
t3 = time.time()

q = 0
counting_ones = 0
counting_zeros = 0
C = np.zeros(len(terms), dtype=np.int)
for doc in docs:
    ocurrenceLine = []
    tokens = nltk.wordpunct_tokenize(doc.lower())

    bigrams = list(nltk.FreqDist(nltk.bigrams(tokens)).keys())
    for term in terms:
        if "_" in term:
            bigram = (term.split("_")[0], term.split("_")[1])
            if bigram in bigrams:
                ocurrenceLine.append(1)
                counting_ones += 1
            else:
                ocurrenceLine.append(0)
                counting_zeros += 1
        else:
            if term in tokens:
                ocurrenceLine.append(1)
                counting_ones += 1
            else:
                ocurrenceLine.append(0)
                counting_zeros += 1
    C = np.vstack([C, ocurrenceLine])
    q += 1
    if q == nlimit:
        break

t4 = time.time()

# Output results
print "Execution Summary [Python]"
print "+++++++++++++++++++++++++++++++++++++++++++++++++"
print "\tTerms:                             %d" % len(terms)
print "\tDocs:                              %d" % len(docs)
print "\tFormal Context (#docs,#terms):     (%d, %d)" % (C.shape[0], C.shape[1])
print "\tOnes found:                        %d" % counting_ones
print "\tZeros found:                       %d" % counting_zeros
print "\tTime reading terms file:           %d seconds" % int(t2 - t1)
print "\tTime reading docs file:             %d seconds" % int(t3 - t2)
print "\tTime generating formal context:    %d seconds" % int(t4 - t3)
print "\tTotal time:                        %d seconds" % int(t4 - t1)
print "+++++++++++++++++++++++++++++++++++++++++++++++++"