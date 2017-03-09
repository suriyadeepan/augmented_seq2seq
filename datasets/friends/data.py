FILENAME = 'sequences_full.csv'
VOCAB_SIZE = None
UNK = 'UNK'

POS_TAGS = { 'CC' : '<CC>', 'CD' : '<CD>', 'DT' : '<DT>', 'EX' : '<EX>', 'FW' : '<FW>', 'IN' : '<IN>', 'JJ' : '<JJ>', 'JJR' : '<JJR>', 'JJS' : '<JJS>', 'LS' : '<LS>', 'MD' : '<MD>', 'NN' : '<NN>', 'NNS' : '<NNS>', 'NNP' : '<NNP>', 'NNPS' : '<NNPS>', 'PDT' : '<PDT>', 'POS' : '<POS>', 'PRP' : '<PRP>', 'PRP' : '<PRP>', 'RB' : '<RB>', 'RBR' : '<RBR>', 'RBS' : '<RBS>', 'RP' : '<RP>', 'SYM' : '<SYM>', 'TO' : '<TO>', 'UH' : '<UH>', 'VB' : '<VB>', 'VBD' : '<VBD>', 'VBG' : '<VBG>', 'VBN' : '<VBN>', 'VBP' : '<VBP>', 'VBZ' : '<VBZ>', 'WDT' : '<WDT>', 'WP' : '<WP>', 'WP$' : '<WP$>', 'WRB' : '<WRB>' }


# imports : in the order of usage
import itertools
import nltk

import random
import sys

import pickle


'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
    return fix_win_encode(open(filename).read()).split('\n')[1:-1]

def fix_win_encode(text):
    return text.replace('\x92', "'").replace('\x97', ' ').replace('\x91', '').replace('_b_','').replace('*','').replace('\x93','')


'''
 split each row of form "query |respect| response"
  to [ query, response, respect ]

'''
def split_row(lines):
    q,r,respect = [], [], []
    for line in lines:
        line = line.split('|')
        r.append(split_and_tag(line[0]))
        q.append(split_and_tag(line[-1]))
        respect.append(int(line[1]))
    return q,r,respect


'''
 split sentences into words and tags with nltk
  replace foreign words and numbers 
   into <FW> and <CD> tags
    
'''
def split_and_tag(line):
    wtags = nltk.pos_tag(nltk.word_tokenize(line.strip()))
    words = []
    for w,t in wtags:
        if t == 'CD' or t == 'FW':
            w = t
        words.append(w)
    return words
    
    
'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    vocab = [ item for item in vocab if item[1] > 1 ]
    # index2word
    index2word = ['_'] + ['UNK'] + list(POS_TAGS.keys()) + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 There will be no zero padding!
 
'''
def encode(q, r, w2idx):
    # num of rows
    data_len = len(q)

    idx_q, idx_r = [], []

    for i in range(data_len):
        idx_q.append(encode_seq(q[i], w2idx))
        idx_r.append(encode_seq(r[i], w2idx))

    return idx_q, idx_r


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def encode_seq(seq, lookup):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            tag = nltk.pos_tag([word])[-1][-1]
            if tag in lookup:
                indices.append(lookup[tag])
            else:
                indices.append(lookup[UNK])
    return indices


def process_data():

    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)

    # change to lower case
    lines = [ line.lower() for line in lines ]

    print('>> [read_lines] {} lines;\nexamples\n{}'.
            format(len(lines), lines[121:125]))

    # split row into query, response and respect
    q, r, respect = split_row(lines)

    print('\n>> [split_row] \n{} {} {}'.
            format( q[121:125], r[121:125], respect[121:125]))

    #############
    # NL pipeline
    ####

    ##
    # [1] Spell Check
    #
    # [2] POS tagging

    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_(q+r, vocab_size=None)

    idx_q, idx_r = encode(q, r, w2idx)

    data = {
        'q' : idx_q, 
        'r' : idx_r, 
        'respect' : respect
            }

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'freq_dist' : freq_dist,
            'respect_size' : max(respect) + 1
                }

    # write to disk : data control dictionaries
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)



def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    with open(PATH + 'data.pkl', 'rb') as f:
        data = pickle.load(f)

    return data, metadata


if __name__ == '__main__':
    process_data()
