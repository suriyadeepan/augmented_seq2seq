import numpy as np
from random import sample, shuffle, randint


'''
 create batches

'''
def create_batches(data_):
    qr = list(zip(data_['q'], data_['r'], data_['respect']))
    batches = {}
    for qi,ri,respecti in qr:
        lqi, lri = len(qi), len(ri)
        if (lqi,lri) in batches:
            batchi = batches[(lqi,lri)]
        else:
            batchi = []
        batchi += [(qi, ri, respecti)]
        batches[(lqi,lri)] = batchi
    return [ batches[k] for k in batches ]


'''
 split data into train (80%), test (20%)

'''
def split_dataset(batches, ratio = [0.8, 0.2] ):

    nbatches = len(batches)
    num_train = int(ratio[0]*nbatches)

    # shuffle batches
    shuffle(batches) # why do i even bother to write comments!

    trainset = batches[:num_train]
    testset  = batches[num_train:]

    return trainset, testset


'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(dataset):
    while True: # use try catch here; just repeat idx=.. and batch=...
        #  why do you us try-catch? clip the idx value based on the data at hand. can't we do that?
        #   i did that.. but not working; weird exception - out of bounds; you try fixing it then!
        # shall I run this once?sure
        idx = randint(0, len(dataset)-1) # choose a random batch id
        batch = dataset[idx] # fetch the batch
        bx = [bi[0] for bi in batch]
        by = [bi[1] for bi in batch]
        br = [bi[2] for bi in batch]
        yield ( np.array(bx, dtype=np.int32).reshape([len(bx), len(bx[0])]), 
                np.array(by, dtype=np.int32).reshape([len(by), len(by[0])]), 
                np.array(br, dtype=np.int32) )


'''
 a generic decode function 
    inputs : sequence, lookup

'''
def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])
