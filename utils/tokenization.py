import re
import pandas as pd
import os
import sys

_WORD_SPLIT1 = re.compile(b"([.,!?\"':;)(])")
_WORD_SPLIT = re.compile(" ")
def basic_tokenizer(sentence, convert_lower=True):
    if convert_lower:
        sentence = str(sentence).lower()
    words = []
    for space_separated_fragment in sentence.strip().split():
        #space_separated_fragment = space_separated_fragment.strip().strip(',')
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]
    


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]

############## Used for tokenizing training Data #Currently not used ################################

def tokenize(x,convert_lower=True,space_replace_chars=None,remove_chars=None,replace_digits=True, return_list = True):
    """
    Cleans the input string according to pre-defined set of rules.
    
    Input: 
        x: str
            word or description which needs to be tokenized
        
        convert_lower: Boolean, default False
            boolean flag for connverting input string to lower case
        space_replace_chars: list, if None then default as described below
            list of characters to be replaced by space
        remove_chars: list, if None then default as described below
            list of characters to be removed
        replace_digits: Boolean, default True
            Boolean flag for replacing digits with their consecutive counts
    
    Returns:
        tokenized version of the input string
   
    """
    tk_x = str(x)
    #import pdb; pdb.set_trace()
    
    
    # removes both lowercase and upper case verison of www. from the word
    tk_x = tk_x.replace('www.','')
    tk_x = tk_x.replace('WWW.','')
    
    # removes both lowercase and upper case verison of .com from the word
    tk_x = tk_x.replace('.com','')
    tk_x = tk_x.replace('.COM','')
    
    # repacing string '&amp'  which gets induced due to movement across various filesystems with '&'
    tk_x = tk_x.replace('&amp;','&')
    
    
    # replace all 3 or more consecutive occurences of the * with the ' <M> ' 
    regex = re.compile('\*{3,}')
    tk_x= regex.sub(' <M> ',tk_x)
    
    # replace all 3 or more consecutive occurences of the X with the ' <M> ' 
    regex = re.compile('X{3,}')
    tk_x= regex.sub(' <M> ',tk_x)
    
    # replace all 3 or more consecutive occurences of the x with the ' <M> ' 
    regex = re.compile('x{3,}')
    tk_x= regex.sub(' <M> ',tk_x)
    
    # list of characters which needs to be replaced with space
    if space_replace_chars is None:
        space_replace_chars_ = [':',',','"','[',']','~','*',';', '!', '?', '(', ')','@','&']
    else:
        space_replace_chars_ = space_replace_chars
    tk_x = tk_x.translate({ord(x): ' ' for x in space_replace_chars_})
    
    # list of characters which needs to be removed
    #remove_chars = ['-',"'",'.']
    # keeping - as it is for now
    if remove_chars is None:
        remove_chars_ = ["'",'.','|'] 
    else:
        remove_chars_ = remove_chars
    
    tk_x = tk_x.translate({ord(x): '' for x in remove_chars_})
    
    # replace all consecutive spaces with one space
    tk_x = re.sub( '\s+', ' ', tk_x).strip()
    
    # find all consecutive numbers present in the word, first converted numbers to * to prevent conflicts while replacing with numbers
    if replace_digits:
        regex = re.compile(r'([\d])')
        tk_x = regex.sub('*',tk_x)
        nos = re.findall(r'([\*]+)',tk_x)
    
    # replace the numbers with the corresponding count like 123 by 3
        for no in nos:
            tk_x = tk_x.replace(no,str(len(no)),1)
        
    if convert_lower:
        tk_x = tk_x.lower()
        
    if(return_list):
        return basic_tokenizer(tk_x)
    return tk_x
    

