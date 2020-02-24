from d import Dict

def load(path):
    data = open(path).read().strip().split('\n')[1:]
    data = [s.split(',') for s in data]
    return data

med = open('./data/MED.txt').read().strip().split('\n')
ill = open('./data/ILL.txt').read().strip().split('\n')
dev = load('data/dev.csv')
train = load('data/train.csv')

d = Dict("data")
for m in med:
    d[m] = {'med': '药品'}
for i in ill:
    d[i] = {'ill': '疾病'}
    
for s in dev+train:
    d[s[0]] = {'ill': '疾病'}
d.save()