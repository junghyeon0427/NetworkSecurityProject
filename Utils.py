import pickle

def open_pickle(path: str):
    with open(path, 'rb') as f:
        target = pickle.load(f)
    return target
