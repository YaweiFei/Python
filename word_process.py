
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import string
import pandas
import random

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

file = 'abcnews-date-text.csv'
file1 = pandas.read_csv(file)
nrows = file1.shape[0]
m = round(0.01*nrows)
skip = sorted(random.sample(range(1,nrows+1),nrows-m))
new = pandas.read_csv(file,skiprows=skip)
export_csv=new.to_csv(r'C:\Users\Feiya\Desktop\AIAssignment2\sampletext.csv')
in_filename='sampletext.csv'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding="utf8")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):

	# replace '--' with a space ' '
    bad_chars = ['\n',',']
    for i in range(len(bad_chars)):
       doc = doc.replace(bad_chars[i], ' ')
	# split into tokens by white space
    tokens = doc.split(' ')
	# remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
	# make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load document
doc = load_doc(in_filename)
doc= doc[27:]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))


# organize into sequences of tokens
length = 100 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# save sequences to file
out_filename = 'sample.txt'
save_doc(sequences, out_filename)