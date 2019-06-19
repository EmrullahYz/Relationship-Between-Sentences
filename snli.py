from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile

import numpy as np
np.random.seed(1337)  # for reproducibility

'''
300D Model - Train / Test (epochs)
=-=-=
Batch size = 512
Fixed GloVe
- 300D SumRNN + Translate + 3 MLP (1.2 million parameters) - 0.8315 / 0.8235 / 0.8249 (22 epochs)
- 300D GRU + Translate + 3 MLP (1.7 million parameters) - 0.8431 / 0.8303 / 0.8233 (17 epochs)
- 300D LSTM + Translate + 3 MLP (1.9 million parameters) - 0.8551 / 0.8286 / 0.8229 (23 epochs)

Following Liu et al. 2016, I don't update the GloVe embeddings during training.
Unlike Liu et al. 2016, I don't initialize out of vocabulary embeddings randomly and instead leave them zeroed.

The jokingly named SumRNN (summation of word embeddings) is 10-11x faster than the GRU or LSTM.

Original numbers for sum / LSTM from Bowman et al. '15 and Bowman et al. '16
=-=-=
100D Sum + GloVe - 0.793 / 0.753
100D LSTM + GloVe - 0.848 / 0.776
300D LSTM + GloVe - 0.839 / 0.806
''' 

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

def extract_tokens_from_binary_parse(parse):
	parse = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+',"",parse)
	parse = re.sub(r"[^\x00-\x7F]+"," ", parse)
	parse = re.sub('[ ]+',' ', parse)
	parse = re.sub(r"[^\x00-\x7F]+"," ", parse)
	return parse.replace('(', ' ').replace(')', ' ').replace('.', ' ').replace(';', ' ').replace("'", ' ').replace("""\t""", ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def cleanText(s):
        
        s = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+',"",s)
        s = re.sub(r'( [0-9,\.]+)',r"\1 ", s)
        s = re.sub(r'\$'," $ ", s)
        s = re.sub('[ ]+',' ', s)
        return s.lower()

def yield_turkish_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn,encoding="utf8")):
    if limit and i > limit:
      break
    l=line.strip().split(",")
    if len(l)<6 or len(l)>6:
        continue
    data = l
    label = data[3]
    s1 = ' '.join(extract_tokens_from_binary_parse(data[5]));
    s2 = ' '.join(extract_tokens_from_binary_parse(data[4]))
    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2)



def get_data(fn, limit=None):
  raw_data = list(yield_turkish_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]

  print(max(len(x.split()) for x in right))
  print(max(len(x.split()) for x in left))

  LABELS = {'çelişki': 0, 'nötr': 1, 'Vasiyetiniz': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))

  return left, right, Y
  

def yield_validation_data_turkish(fn, skip_no_majority=True, limit=None):
  num=0
  for i, line in enumerate(open(fn,encoding="utf8")):
    if limit and i > limit:
      break
    l=line.strip().split(",")
    if len(l)<6 or len(l)>6:
      continue
    if num>10000:
      break
    data = l
    label = data[3]
    s1 = ' '.join(extract_tokens_from_binary_parse(data[5]));
    s2 = ' '.join(extract_tokens_from_binary_parse(data[4]))
    if skip_no_majority and label == '-':
      continue
    num=num+1
    yield (label, s1, s2)


def get_data_validation(fn, limit=None):
  raw_data = list(yield_validation_data_turkish(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]

  print(max(len(x.split()) for x in right))
  print(max(len(x.split()) for x in left))

  LABELS = {'çelişki': 0, 'nötr': 1, 'Vasiyetiniz': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))
  return left, right, Y


training = get_data('translated_SNLI.csv')
test = get_data('10k_test(550k).csv')
validation = get_data_validation('20kValidate(540k).csv')


tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])
# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'çelişki': 0, 'nötr': 1, 'Vasiyetiniz': 2}
#RNN = recurrent.LSTM
#RNN = lambda *args, **kwargs: Bidirectional(recurrent.LSTM(*args, **kwargs))
#RNN = recurrent.GRU
#RNN = lambda *args, **kwargs: Bidirectional(recurrent.GRU(*args, **kwargs))
# Summation of word embeddings
RNN = None
LAYERS = 1
USE_GLOVE = True
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
BATCH_SIZE = 512
PATIENCE = 4 # 8
MAX_EPOCHS = 1
MAX_LEN = 42
DP = 0.2
L2 = 4e-6
ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'
print('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_GLOVE, TRAIN_EMBED))

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

print('Build model...')
print('Vocab size =', VOCAB)

GLOVE_STORE = 'precomputed_glove.weights'
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

translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')
prem = embed(premise)
hypo = embed(hypothesis)
prem = translate(prem)
hypo = translate(hypo)

if RNN and LAYERS > 1:
  for l in range(LAYERS - 1):
    rnn = RNN(return_sequences=True, **rnn_kwargs)
    prem = BatchNormalization()(rnn(prem))
    hypo = BatchNormalization()(rnn(hypo))
rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
prem = rnn(prem)
hypo = rnn(hypo)
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)

joint = concatenate([prem, hypo])
joint = Dropout(DP)(joint)
for i in range(3):
  joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, W_regularizer=l2(L2) if L2 else None)(joint)
  joint = Dropout(DP)(joint)
  joint = BatchNormalization()(joint)

pred = Dense(len(LABELS), activation='softmax')(joint)
model = Model(input=[premise, hypothesis], output=pred)
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

print('Training')
#_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
#callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
#model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

# Restore the best found model during validation
#model.load_weights(tmpfn)


#save model.h5

#model.fit("Today weather is realy good","Weather is good for today")
#model = load_model('my_model.h5')

text1='ali evde ders çalışıyor'
text2='ali evde ders çalışıyor'
from keras.models import load_model
model = load_model('model_translated_turkish.h5')
text1 = ' '.join(text1.split())
text2 = ' '.join(text2.split())
print('sentences ===> ',text1, text2)

text_ints1 = [ord(x) for x in text1 if ord(x)<128]
onehot1 = keras.utils.to_categorical(text_ints1, num_classes=128)
print(onehot1.shape)

text_ints2 = [ord(x) for x in text2 if ord(x)<128]
onehot2 = keras.utils.to_categorical(text_ints2, num_classes=128)
print('onehot2 ===>',onehot2.shape)












