from Bio import SeqIO
import numpy as np
import torch

import os
import re
import random

base = ["A", "T", "G", "C", "N"]

def to_tensor(string, unknown_as_label=False):
    seq = sanitize(string)
    idx = list(map(lambda b: base.index(b), seq))
    idx = torch.tensor(idx)
    n_symbols = len(base) if unknown_as_label else len(base) - 1
    seq = torch.eye(5, n_symbols)[idx].T

    return seq.unsqueeze(0)
    
def sanitize(seq, unknown_as_label=False):
    seq = re.sub(r"R", ["A", "G"][random.randint(0, 1)], seq)
    seq = re.sub(r"Y", ["T", "C"][random.randint(0, 1)], seq)
    seq = re.sub(r"K", ["G", "T"][random.randint(0, 1)], seq)
    seq = re.sub(r"S", ["G", "C"][random.randint(0, 1)], seq)
    seq = re.sub(r"W", ["A", "T"][random.randint(0, 1)], seq)
    seq = re.sub(r"B", ["T", "G", "C"][random.randint(0, 2)], seq)
    seq = re.sub(r"D", ["A", "G", "T"][random.randint(0, 2)], seq)
    seq = re.sub(r"H", ["A", "C", "T"][random.randint(0, 2)], seq)
    seq = re.sub(r"M", ["A", "C"][random.randint(0, 1)], seq)

    if unknown_as_label:
        seq = re.sub(r"N", base[:-1][random.randint(0, 3)], seq)

    return seq

def read_one(path, unknown_as_label=False):
    seq = None
    name = ""

    # the file specified by "path" must contain genome of one species
    for record in SeqIO.parse(path, "fasta"):
        seq = to_tensor(str(record.seq), unknown_as_label)
        name = record.name

    return name, seq

def read_all(dir):
    species = []
    seqs = []
    labels = []

    for i, filename in enumerate(os.listdir(dir)):
        print("\rLoading... {:0=3}/{:0=3}".format(i+1, len(os.listdir(dir))), end="")
        name, seq = read_one(dir+"/"+filename)
        species.append(name)
        seqs.append(seq)
        labels.append(i)

    return species, seqs, labels

class DataLoader():
    def __init__(self, batch_size, how="random", length=1024, is_train=True, use_all=False, test_size=0.3, n_batch=1000):
        self.is_train = is_train # if true, loader splits each genome sequence into two parts, train set and test set. 
        self.batch_size = batch_size
        self.how = how # sampling method
        self.n_batch = n_batch # number of iterations per epoch
        self._i = 0
        self.length = length # sequence length
        self.use_all = use_all # if true, loader takes whole genome sequence as a train set.
        self.test_size = test_size # proportion of test set

    def __call__(self, species, seqs, labels):
        self.species, self.seqs, self.labels = species, seqs, labels
        self.labels = np.array(self.labels)
        self.lens = np.array([x.size(2) for x in seqs])
        self.bounds = self.lens if self.use_all else np.array([int(x.size(2) * self.test_size) for x in self.seqs])

        # raise an error if you have sequences in "seqs" shorter than sampling length
        if self.lens.min() < self.length:
            raise Exception("Your dataset contain sequences shorter than sampling length({} bp).".format(self.length))

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self.n_batch:
            self._i = 0
            raise StopIteration()

        self._i += 1

        X_batch, y_batch = self.get_batch()
        
        X_batch = torch.tensor(X_batch).float()
        y_batch = torch.tensor(y_batch).long()

        return X_batch, y_batch

    def __len__(self):
        return self.n_batch

    def get_batch(self):
        if self.how=="random":
            idx = np.random.randint(len(self.bounds), size=self.batch_size)

            def crop(arr, is_train):
                label = arr[0]

                if is_train:
                    start = random.randint(0, self.bounds[label] - self.length)
                else:
                    start = random.randint(self.bounds[label], self.lens[label] - self.length)

                return self.seqs[label][0,:,start:start+self.length].numpy()

            return np.apply_along_axis(crop, 1, idx.reshape(-1, 1), self.is_train), self.labels[idx]

    # sample sequences randomly from specified species
    def get_one(self, ind):
        start = np.random.randint(self.lens[ind] - self.length + 1)
        return self.seqs[ind][:,:,start:start+self.length]