import pickle
import os
import re

def find_ckpt_filename(path_or_file):
    if os.path.isfile(path_or_file):
        epoch = int(re.search('epoch_([0-9]*).pkl', path_or_file).group(1))
        return path_or_file, epoch
    files = [f for f in os.listdir(path_or_file) if ('pkl' in f)]
    for f in sorted(files, reverse=True):
        fname = os.path.join(path_or_file, f)
        try:
            with open(fname, "rb") as f:
                pickle.load(f)
            epoch = int(re.search('epoch_([0-9]*).pkl', fname).group(1))
            return fname, epoch
        except (OSError, EOFError):
            print('Error loading checkpoint. Trying next checkpoint...', fname)
    return None, 0

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
