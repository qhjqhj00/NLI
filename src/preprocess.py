import csv
from d import Dict
import argparse

dic = Dict('../data')
parser = argparse.ArgumentParser()
parser.add_argument("--replace", default=True, type=str)
args = parser.parse_args()

def load(path):
    data = []
    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            data.append(line)
    data.pop(0)
    return data

def replace(sentence):
    candicates = dic.multi_max_match(sentence)
    for k in candicates:
        placeholder = list(candicates[k]['value'].values())[0]
        sentence = sentence.replace(k, placeholder)
    return sentence
        
def process_data(path, save_path, split_dev = False, replace = True):
    data = load(path)
    with open(save_path, 'w') as f:
        for s in data:
            if s[-1] == '': continue
            if replace:
                s2 = replace(s[2])
                s1 = replace(s[1])
            else:
                s2 = s[2]
                s1 = s[1]
            f.write(f'{s[-1]}\t{s1}\t{s2}\n')
        f.truncate()
    if split_dev:
        l = len(data)
        with open(save_path.replace('train', 'dev'), 'w') as f:
            for s in data[:int(0.2*l)]:
                if s[-1] == '': continue
                if replace:
                    s2 = replace(s[2])
                    s1 = replace(s[1])
                else:
                    s2 = s[2]
                    s1 = s[1]
                f.write(f'{s[-1]}\t{s1}\t{s2}\n')
            f.truncate()
process_data('../raw/dev.csv', '../data/processed_test.tsv', replace=args.replace)
process_data('../raw/train.csv', '../data/processed_train.tsv', split_dev = True, replace=args.replace)
