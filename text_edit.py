from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import numpy as np
np.random.seed(1337)
from keras.utils import np_utils
np.set_printoptions(threshold=np.nan)

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

a='20'

def yield_examples(fn, skip_no_majority=True, limit=None):
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
  

def yield_examples_turkish(fn, skip_no_majority=True, limit=None):
  print("Loading testing/labelled data from "+fn)
  s1=[]
  s2=[]
  label=[]
  # positive samples from file
  for line in open(fn,encoding="utf8"):
    l=line.strip().split(",")
    if len(l)<6 or len(l)>6:
        continue
    #print(l[4])
    s1.append(l[4].lower()) 
    s2.append(l[5].lower())
    label.append(l[3].lower())
    if skip_no_majority and label == '-':
      continue
  yield (label, s1, s2) 

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  left = [s1 for _, s1, s2 in raw_data]
  right = [s2 for _, s1, s2 in raw_data]

  print(max(len(x.split()) for x in right))
  print(max(len(x.split()) for x in left))

  LABELS = {'çelişki': 0, 'nötr': 1, 'Vasiyetiniz': 2}
  Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
  Y = np_utils.to_categorical(Y, len(LABELS))

  return left, right, Y
  

def write_turkish_txt(fn, skip_no_majority=True, limit=None):
  print("Loading testing/labelled data from "+fn)
  s1=[]
  s2=[]
  label=[]
  # positive samples from file
  for line in open(fn,encoding="utf8"):
    l=line.strip().split(",")
    if len(l)<6 or len(l)>6:
        continue
    #print(l[4])
    s1.append(l[4].lower()) 
    s2.append(l[5].lower())
    label.append(l[3].lower())
    if skip_no_majority and label == '-':
      continue
  yield (label, s1, s2) 


data = get_data('train'+a+'k.xlsx - Sayfa1.csv')
print(data[2])