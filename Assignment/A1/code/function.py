def read_file(path: str):
    with open(path) as f:
        texts_train = f.read()
    f.close()
    return texts_train
