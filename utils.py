import numpy as np

import os
import codecs
import glob2
import json
import tempfile
import pandas as pd


def to_string(char_list, lowercase):
    if lowercase:
        return u''.join(char_list).lower()
    return u''.join(char_list)


def create_training_data(text_in, text_out, char_to_int, n_vocab,
                         seq_length=25, batch_size=100, padding_char=u'\n',
                         lowercase=False, predict_chars=0, step=1,
                         char_embedding=False):
    """Create padded one-hot encoded data sets from aligned text.

    A sample consists of seq_length characters from text_in (e.g., the ocr
    text) (may include empty characters), and seq_length characters from
    text_out (e.g., the gold standard text) (may include empty characters).
    text_in and text_out contain aligned arrays of characters.
    Because of the empty characters ('' in the character arrays), the input
    and output sequences may not have equal length. Therefore input and
    output are padded with a padding character (default: newline).

    Returns:
      int: the number of samples in the dataset
      generator: generator for one-hot encoded data (so the data doesn't have
        to fit in memory)
    """
    dataX = []
    dataY = []
    text_length = len(text_in)
    for i in range(0, text_length-seq_length-predict_chars+1, step):
        seq_in = text_in[i:i+seq_length]
        seq_out = text_out[i:i+seq_length+predict_chars]
        dataX.append(to_string(seq_in, lowercase))
        dataY.append(to_string(seq_out, lowercase))
    if char_embedding:
        data_gen = data_generator_embedding(dataX, dataY, seq_length,
                                            predict_chars, n_vocab,
                                            char_to_int, batch_size,
                                            padding_char)
    else:
        data_gen = data_generator(dataX, dataY, seq_length, predict_chars,
                                  n_vocab, char_to_int, batch_size,
                                  padding_char)

    return len(dataX), data_gen


def data_generator(dataX, dataY, seq_length, predict_chars, n_vocab,
                   char_to_int, batch_size, padding_char):
    while 1:
        for batch_idx in range(0, len(dataX), batch_size):
            X = np.zeros((batch_size, seq_length, n_vocab), dtype=np.bool)
            Y = np.zeros((batch_size, seq_length+predict_chars, n_vocab),
                         dtype=np.bool)
            sliceX = dataX[batch_idx:batch_idx+batch_size]
            sliceY = dataY[batch_idx:batch_idx+batch_size]
            for i, (sentenceX, sentenceY) in enumerate(zip(sliceX, sliceY)):
                for j, c in enumerate(sentenceX):
                    X[i, j, char_to_int[c]] = 1
                for j in range(seq_length-len(sentenceX)):
                    X[i, len(sentenceX) + j, char_to_int[padding_char]] = 1
                for j, c in enumerate(sentenceY):
                    Y[i, j, char_to_int[c]] = 1
                for j in range(seq_length+predict_chars-len(sentenceY)):
                    Y[i, len(sentenceY) + j, char_to_int[padding_char]] = 1
            yield X, Y


def data_generator_embedding(dataX, dataY, seq_length, predict_chars, n_vocab,
                             char_to_int, batch_size, padding_char):
    while 1:
        for batch_idx in range(0, len(dataX), batch_size):
            X = np.zeros((batch_size, seq_length), dtype=np.int)
            Y = np.zeros((batch_size, seq_length+predict_chars, n_vocab),
                         dtype=np.bool)
            sliceX = dataX[batch_idx:batch_idx+batch_size]
            sliceY = dataY[batch_idx:batch_idx+batch_size]
            for i, (sentenceX, sentenceY) in enumerate(zip(sliceX, sliceY)):
                for j, c in enumerate(sentenceX):
                    X[i, j] = char_to_int[c]
                for j in range(seq_length-len(sentenceX)):
                    X[i, len(sentenceX) + j] = char_to_int[padding_char]
                for j, c in enumerate(sentenceY):
                    Y[i, j, char_to_int[c]] = 1
                for j in range(seq_length+predict_chars-len(sentenceY)):
                    Y[i, len(sentenceY) + j, char_to_int[padding_char]] = 1
            yield X, Y


def read_texts(data_files, data_dir):
    raw_text = []
    gs = []
    ocr = []
    #print('ok')
    for df in data_files:
        #print(df)
        if data_dir is None:
            fi = df
        else:
            fi = os.path.join(data_dir, df)
            print(fi)
        with codecs.open(fi, encoding='utf-8') as f:
        #with codecs.open(fi, 'r', encoding='utf-8') as f:
            aligned = json.load(f)
    #print('ok')
        ocr.append(aligned['ocr'])
        ocr.append([' '])             # add space between two texts
        gs.append(aligned['gs'])
        gs.append([' '])              # add space between two texts

        raw_text.append(''.join(aligned['ocr']))
        raw_text.append(''.join(aligned['gs']))

    # Make a single array, containing the character-aligned text of all data
    # files
    gs_text = [y for x in gs for y in x]
    ocr_text = [y for x in ocr for y in x]

    return ' '.join(raw_text), gs_text, ocr_text

def read_text_to_predict(text, seq_length, lowercase, n_vocab,
                         char_to_int, padding_char, predict_chars=0, step=1,
                         char_embedding=False):
    dataX = []
    text_length = len(text)
    for i in range(0, text_length-seq_length-predict_chars+1, step):
        seq_in = text[i:i+seq_length]
        dataX.append(to_string(seq_in, lowercase))

    #print(dataX)

    if char_embedding:
        X = np.zeros((len(dataX), seq_length), dtype=np.int)
    else:
        X = np.zeros((len(dataX), seq_length, n_vocab), dtype=np.bool)

    for i, sentenceX in enumerate(dataX):
        for j, c in enumerate(sentenceX):
            if char_embedding:
                X[i, j] = char_to_int[c]
            else:
                try:
                    X[i, j, char_to_int[c]] = 1
                except Exception as e:
                    #print('read_text_to_predict, character no encontrado')
                    #print(e)
                    X[i, j, char_to_int[' ']] = 1

        for j in range(seq_length-len(sentenceX)):
            if char_embedding:
                X[i, len(sentenceX) + j] = char_to_int[padding_char]
            else:
                X[i, len(sentenceX) + j, char_to_int[padding_char]] = 1
    #print(X)
    return X

def read_text_to_predict_mod(text, seq_length, lowercase, n_vocab, diff_n_vocab,
                         char_to_int,diff_char_to_int, diff_int_to_char, padding_char, predict_chars=0, step=1,
                         char_embedding=False):
    print('estoy aqui realmente ')
    dataX = []
    text_length = len(text)
    for i in range(0, text_length-seq_length-predict_chars+1, step):
        seq_in = text[i:i+seq_length]
        dataX.append(to_string(seq_in, lowercase))

    #print(dataX)

    if char_embedding:
        X = np.zeros((len(dataX), seq_length), dtype=np.int)
    else:
        X = np.zeros((len(dataX), seq_length, n_vocab), dtype=np.bool)
        X_diff = np.zeros((len(dataX), seq_length, diff_n_vocab), dtype=np.bool)

    for i, sentenceX in enumerate(dataX):
        for j, c in enumerate(sentenceX):
            if char_embedding:
                #print('char_embedding')
                X[i, j] = char_to_int[c]
            else:
                #print('no char_embedding')
                try:
                    X[i, j, char_to_int[c]] = 1
                except Exception as e:
                    print('read_text_to_predict, character no encontrado, usar el vocab_out')
                    print(e)
                    #X[i, j, char_to_int[' ']] = 1
                    try:
                        X_diff[i, j, diff_char_to_int[c]] = 1
                        print("%i int para %s char"%(diff_char_to_int[c], c))
                    except Exception as e:
                        print('no fue encontrado ni en el vocab_out')
                        print(e)

        for j in range(seq_length-len(sentenceX)):
            if char_embedding:
                #print('char_embedding')
                X[i, len(sentenceX) + j] = char_to_int[padding_char]
            else:
                #print('no char_embedding')
                X[i, len(sentenceX) + j, char_to_int[padding_char]] = 1
    #print(X)
    return X, X_diff





def get_char_to_int(chars):
    return dict((c, i) for i, c in enumerate(chars))


def get_int_to_char(chars):
    return dict((i, c) for i, c in enumerate(chars))


def save_charset(weights_dir, chars, lowercase):
    if lowercase:
        fname = 'chars-lower.txt'
    else:
        fname = 'chars.txt'
    chars_file = os.path.join(weights_dir, fname)
    with codecs.open(chars_file, 'wb', encoding='utf-8') as f:
        f.write(u''.join(chars))


def get_chars(raw_val, raw_test, raw_train, lowercase, padding_char=u'\n',
              oov_char='@'):
    raw_text = ''.join([raw_val, raw_test, raw_train])
    if lowercase:
        raw_text = raw_text.lower()

    chars = sorted(list(set(raw_text)))
    chars.append(padding_char)
    chars.append(oov_char)
    char_to_int = get_char_to_int(chars)

    return chars, len(chars), char_to_int


def cwl_path():
    """Return the path to the directory containing CWL steps.
    """
    module_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(module_path, 'cwl'))


def merge_wordmappings(wm_improved_ocr, wm_original_ocr):
    def to_list_of_dfs(df):
        dfs = []
        start = 0
        for i in df.index[1:]:
            res = df.loc[i]
            print(res)
            if res['word_index'] == 0:
                dfs.append(df.loc[start:i-1])
                start = i
        dfs.append(df.loc[start:])
        return dfs
    io_dfs = to_list_of_dfs(wm_improved_ocr)
    oo_dfs = to_list_of_dfs(wm_original_ocr)

    res = []
    for d1, d2 in zip(io_dfs, oo_dfs):
        d1 = d1.reset_index()
        del d1['index']
        d2 = d2.reset_index()
        del d2['index']
        r = pd.concat([d1, d2], axis=1)
        r.columns = ['wordindex1', 'gs', 'corrected-ocr', 'wordindex2', 'gs2',
                     'original-ocr']
        res.append(r[['gs', 'corrected-ocr', 'original-ocr']])
    df = pd.concat(res)
    df = df.reset_index()
    return df


def merge_wordmappings2(list_wm_improved_ocr, list_wm_original_ocr):
    res = []
    for d1, d2 in zip(list_wm_improved_ocr, list_wm_original_ocr):
        d1 = d1.reset_index()
        del d1['index']
        d2 = d2.reset_index()
        del d2['index']
        r = pd.concat([d1, d2], axis=1)
        r.columns = ['wordindex1', 'gs', 'corrected-ocr', 'wordindex2', 'gs2',
                     'original-ocr']
        res.append(r[['gs', 'corrected-ocr', 'original-ocr']])
    df = pd.concat(res)
    df = df.reset_index()
    return df



def match(name, beginnings):
    for b in beginnings:
        if name.startswith(b) and name[len(b)] in ('.', '_', '-'):
            return True
    return False


def get_files(in_dir, div, name):
    files_out = []

    files = [os.path.splitext(os.path.basename(f))[0] for f in div.get(name, [])]

    for f in os.listdir(in_dir):
        fi = os.path.join(in_dir, f)
        if os.path.isfile(fi) and match(f, files):
            files_out.append(fi)
    files_out.sort()
    return files_out


def get_sequences(gs, ocr, length):
    gs_ngrams = zip(*[gs[i:] for i in range(length)])
    ocr_ngrams = zip(*[ocr[i:] for i in range(length)])

    return [''.join(n) for n in gs_ngrams], [''.join(n) for n in ocr_ngrams]


def get_temp_file():
    """Create a temporary file and return the path.

    Returns:
        Path to the temporary file.
    """
    (fd, fname) = tempfile.mkstemp()
    os.close(fd)

    return fname


def to_space_tokenized(string):
    result = []
    for c in string:
        if c != ' ':
            result.append(c)
        else:
            result.append('<SPACE>')
    return ' '.join(result)
