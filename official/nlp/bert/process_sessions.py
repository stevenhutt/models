import pandas as pd
import numpy as np
from pathlib import Path

import socket, struct

def dottedQuadToNum(ip):
    "convert decimal dotted quad string to long integer"
    return struct.unpack('>L',socket.inet_aton(ip))[0]

def numToDottedQuad(n):
    "convert long int to dotted quad string"
    return socket.inet_ntoa(struct.pack('>L',n))

def split_ip(ip):
    return numToDottedQuad(ip).split('.') 

def make_list(x):
    return [x]

def list_to_str(x):
    return ' '.join(x)

def load_raw_df():
    data_dir = Path('/') / 'home/shutt/data/lcm_last_30_days/raw'
    raw_dfs = []
    num_raws = 1
    for idx, raw_filename in enumerate(data_dir.iterdir()):
        if idx < num_raws:
            raw_df = pd.read_pickle(raw_filename)
            raw_dfs.append(raw_df)
        else:
            pass
    raw_df = pd.concat(raw_dfs, axis=0)
    raw_df = raw_df.sort_values(by=['stop'])
    # select only ipv4
    raw_df = raw_df[(raw_df['layerTransport']==1) & (raw_df['layerNetwork']==1)]
    return raw_df


def get_sentences(raw_df):
    s_1 = raw_df['ipSource'].apply(split_ip).to_numpy()
    s_2 = raw_df['ipDest'].apply(split_ip).to_numpy()
    s_3 = raw_df['portSource'].apply(str).apply(make_list).to_numpy()
    s_4 = raw_df['portDest'].apply(str).apply(make_list).to_numpy()

    s = list(s_1 + s_3 + s_2 + s_4)

    sentences = [list_to_str(x) for x in s]

    tokens = set()
    for i_1 in s:
        for i_2 in i_1:
            tokens.add(i_2) 

    tokens = list(tokens)
    tokens = tokens + ['[UNK]', '[CLS]', '[SEP]', '[MASK]']

    with open('sessions_med.txt', 'w') as filehandle:
        for sentence in sentences:
            filehandle.write('%s\n' % sentence)

    with open('vocab_med.txt', 'w') as filehandle:
        for token in tokens:
            filehandle.write('%s\n' % token)
    return 



        