import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import concatenate, recurrent, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
import re
import numpy as np

np.random.seed(1337)  # for reproducibility
from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import sys
import subprocess
import shutil

if len(sys.argv) == 4:
    cumle1 = sys.argv[1]
    cumle2 = sys.argv[2]
    modelIsim = sys.argv[3]
    sentence1 = cumle1.replace('_', ' ')
    sentence2 = cumle2.replace('_', ' ')

if len(sys.argv) == 3:
    cumle1 = sys.argv[1]
    modelIsim = sys.argv[2]
    sentence1 = cumle1.replace('_', ' ')
    sentence2 = ' '

if len(sys.argv) == 2:
    f = open('results.txt', 'w')
    f.write('0')
    f.close()
    sys.exit(0)




def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']));
    s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)
	
def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]
  print(max(len(x.split()) for x in left))
  print(max(len(x.split()) for x in right))

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))

  return left, right, Y
now_path = os.getcwd()
print(now_path)
os.chdir(now_path)

training = get_data('snli_1.0_train.jsonl')
tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}


from keras.models import load_model
from operator import itemgetter
RNN = None
LAYERS = 1
USE_GLOVE = False
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 512
PATIENCE = 4 # 8
MAX_EPOCHS = 40
MAX_LEN = 42
DP = 0.2
L2 = 4e-6

def extract_tokens_from_binary_parse(parse):
    parse = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+', "", parse)
    parse = re.sub(r"[^\x00-\x7F]+", " ", parse)
    parse = re.sub('[ ]+', ' ', parse)
    parse = re.sub(r"[^\x00-\x7F]+", " ", parse)
    return parse.replace('(', ' ').replace(')', ' ').replace('.', ' ').replace(';', ' ').replace("'", ' ').replace(
        """\t""", ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()
gloveIsim = modelIsim.split(".")[0]
print('glove name : =======>>>>>>'+ gloveIsim)
GLOVE_STORE = gloveIsim

now_path = os.getcwd()
print('path2=======>>>>>'+ now_path)
if USE_GLOVE:
  if not os.path.exists(GLOVE_STORE + '.npy'):
    print('Computing GloVe')
  
    embeddings_index = {}
    f = open('cc.tr.300.vec',encoding="utf8")
    for line in f:
      values = line.split(' ')
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
    f.close()
    
    # prepare embedding matrix
    embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
    for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
      else:
        print('Missing from GloVe: {}'.format(word))
  
    np.save(GLOVE_STORE, embedding_matrix)

  print('Loading GloVe')
  embedding_matrix = np.load(GLOVE_STORE + '.npy')

  print('Total number of null word embeddings:')
  print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

  embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
else:
  embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)

rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)
SumEmbeddings = keras.layers.core.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))


now_path = os.getcwd()
os.chdir(now_path + "/models")
model = load_model(modelIsim)
os.chdir(now_path)
test_sentence_pairs = ([sentence1],[sentence2])

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]))
test_sentence_pairs = prepare_data(test_sentence_pairs)
print(sentence1)
print(sentence2)
s1_sequences=keras.preprocessing.text.text_to_word_sequence(sentence1)
s1_tokenzed=tokenizer.texts_to_sequences(s1_sequences)
#print(s1_sequences)
#print(s1_tokenzed)
#test_sentence_pairs[0] = to_seq(test_sentence_pairs[0])
#test_sentence_pairs[1] = to_seq(test_sentence_pairs[1])
array1 = test_sentence_pairs[0]
array2 = test_sentence_pairs[1]
#print(array1)
prediction = model.predict([np.array(array1),np.array(array2)])
print(prediction)
pred = np.argmax(prediction,axis=1)
if(pred == 0):
    print('Contradiction')
elif(pred == 1):
    print("Neutral")
else:
    print("Entailment")

f = open('results.txt', 'w')
f.write('1'+'\n')
f.write(str(prediction) + '\n')
if(pred == 0):
    f.write('Contradiction')
elif(pred == 1):
    f.write('Neutral')
else:
    f.write('Entailment')
f.close()