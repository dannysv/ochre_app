import codecs
import os
import numpy as np 
import tqdm
from collections import Counter
import os
import nltk


from keras.models import load_model

from utils import get_char_to_int, get_int_to_char, read_text_to_predict
from edlibutils import align_output_to_input
import argparse


def lstm_synced_correct_ocr(model, charset, text):
    # load model
    conf = model.get_config()
    conf_result = conf['layers'][0].get('config').get('batch_input_shape')
    seq_length = conf_result[1]
    #print(seq_length)
    char_embedding = False
    if conf['layers'][0].get('class_name') == u'Embedding':
        char_embedding = True
    with codecs.open(charset, 'r') as f:
        charset = f.read()
    n_vocab = len(charset)
    char_to_int = get_char_to_int(charset)
    int_to_char = get_int_to_char(charset)
    lowercase = True
    for c in u'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        if c in charset:
            lowercase = False
            break

    pad = u'\n'
    #print(charset)
    to_predict = read_text_to_predict(text, seq_length, lowercase,
                                      n_vocab, char_to_int, padding_char=pad,
                                      char_embedding=char_embedding)

    outputs = []
    inputs = []

    predicted = model.predict(to_predict, verbose=0)
    for i, sequence in enumerate(predicted):
        #for p in sequence:
            #print("aqui")
        predicted_indices = [np.random.choice(n_vocab, p=p) for p in sequence]
        pred_str = u''.join([int_to_char[j] for j in predicted_indices])
        outputs.append(pred_str)

        if char_embedding:
            indices = to_predict[i]
        else:
            indices = np.where(to_predict[i:i+1, :, :] == True)[2]
        inp = u''.join([int_to_char[j] for j in indices])
        inputs.append(inp)

    idx = 0
    counters = {}

    for input_str, output_str in zip(inputs, outputs):
        if pad in output_str:
            output_str2 = align_output_to_input(input_str, output_str)
        else:
            output_str2 = output_str
        for i, (inp, outp) in enumerate(zip(input_str, output_str2)):
            if not idx + i in counters.keys():
                counters[idx+i] = Counter()
            counters[idx+i][outp] += 1

        idx += 1

    agg_out = []
    for idx, c in counters.items():
        agg_out.append(c.most_common(1)[0][0])

    corrected_text = u''.join(agg_out)
    corrected_text = corrected_text.replace(pad, u'')
    #print(corrected_text)
    return corrected_text 

def corrigir(model, charset, word):
    resp = {word:0}
    for i in range(10):
        word_corr = lstm_synced_correct_ocr(model, charset,word)
        try:
            resp[word_corr]+=1
        except Exception as e:
            resp.update({word_corr:1})
    return resp 

def corrigir_sent(model, charset, sent):
    resp = corrigir(model, charset, sent)
    #key, _ = max(resp.iteritems(), key=lambda x:x[1])
    key = max(resp, key=lambda key: resp[key])
    return key 

def corrigir_line(model, charset, line):
    sents = nltk.sent_tokenize(line)
    sents_new = []
    for sent in sents:
        sent_new = corrigir_sent(model, charset, sent)
        sents_new.append(sent_new)
    return ' '.join(sents_new)

def read_file(path):
    try:
        with open(path, 'r') as f:
            resp = f.readlines()
        return resp 
    except Exception as e:
        print('errorrrr')
        return None

def processar_onefile(pathin, pathout, txtfile, model, charset):
    lines = read_file(os.path.join(pathin, txtfile))
    out = codecs.open(os.path.join(pathout, txtfile), 'w')
    for line in tqdm.tqdm(lines):
        line = line.strip()
        line_new = corrigir_line(model, charset, line)
        out.write(line_new+'\n')
    out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('scrip para correção com ochre')
    parser.add_argument('--folderin', 
                        type=str,
                        required=False,
                        help='caminho para a pasta de entrada')
    parser.add_argument('--folderout', 
                        required=False,
                        default='ochre',
                        type=str,
                        help='caminho para a pasta de saida')

    args = parser.parse_args()
    folderin=args.folderin 
    folderout=args.folderout
    
    if folderin is not None:
        if os.path.exists(folderin):
            print("carregar modelo")
            model_path = './models/0.1241-88.hdf5'
            model = load_model(model_path)
            charset = './models/chars-lower.txt'

            #listar os arquivos de txt
            files = os.listdir(folderin)
            files = [f for f in files if str(f).endswith('.txt') and 'readme' not in str(f)]

            #pasta saida
            if folderout == 'ochre':
                if folderin[-1]=='/':
                    folderout=folderin[:-1].split('/')[-1]+'_ochre'
                else:
                    folderout= folderin.split('/')[-1]+'_ochre'
            print('pasta de saida: %s'%folderout)
 

            if os.path.exists(folderout)==False:
                os.mkdir(folderout)
            #obter arquivos já processados
            files_ok = os.listdir(folderout)
            #print("processar todos los archivos de una carpeta in folder: %s" %files)
            for fil in files:
                try:
                    if fil not in files_ok:
                        processar_onefile(folderin, folderout, fil, model, charset)
                    else:
                        print("arquivo %s já processado"%fil)
                except Exception as e:
                    print(e)
        else:
            print("folder de entrada no existe")
    else:
        model_path = './models/0.1241-88.hdf5'
        model = load_model(model_path)
        charset = './models/chars-lower.txt'

        word1 = 'pertuguesa'
        word2 = 'locaimente'
        word3 = 'petrolee'
        word4 = 'dlariamante'
        word5 = 'ótime dla'
        word6 = '□       controle'

        resp1 = corrigir(model, charset, word1)
        resp2 = corrigir(model, charset, word2)
        resp3 = corrigir(model, charset, word3)
        resp4 = corrigir(model, charset, word4)
        resp5 = corrigir(model, charset, word5)
        resp6 = corrigir(model, charset, word6)

        print('OCR: %s'%word1)
        print('SUGESTÕES: %s'%resp1)
        print('OCR: %s'%word2)
        print('SUGESTÕES: %s'%resp2)
        print('OCR: %s'%word3)
        print('SUGESTÕES: %s'%resp3)
        print('OCR: %s'%word4)
        print('SUGESTÕES: %s'%resp4)
        print('OCR: %s'%word5)
        print('SUGESTÕES: %s'%resp5)
        print('SUGESTÕES: %s'%resp6)
