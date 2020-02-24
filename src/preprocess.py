from d import Dict

dic = Dict('../data')

def load(path):
    data = open(path).read().strip().split('\n')[1:]
    data = [s.split(',') for s in data]
    return data

def replace(sentence):
    candicates = dic.multi_max_match(sentence)
    for k in candicates:
        placeholder = list(candicates[k]['value'].values())[0]
        sentence = sentence.replace(k, placeholder)
    return sentence
        
def process_data(path, save_path, split_dev = False):
    data = load(path)
    with open(save_path, 'w') as f:
        for s in data:
            s2 = replace(s[2])
            s1 = replace(s[1])
            f.write(f'{s[-1]}\t{s1}\t{s2}\n')
    if split_dev:
        l = len(data)
        with open(save_path.replace('train', 'dev'), 'w') as f:
            for s in data(:int(0.2*l)):
                s2 = replace(s[2])
                s1 = replace(s[1])
                f.write(f'{s[-1]}\t{s1}\t{s2}\n')

process_data('../data/dev.csv', '../data/processed_test.tsv')
process_data('../data/train.csv', '../data/processed_train.tsv', split_dev = True)
