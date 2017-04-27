import json
import random
import numpy
import math
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve
from collections import OrderedDict
import matplotlib.pyplot as plt
def encoded(a):
    return str(bin(a[0])).split('b')[1].zfill(8) + str(bin(a[1])).split('b')[1].zfill(8)

def uniform_crossover(a, b):
    assert len(a) == len(b)
    res = ''
    for i in range(len(a)):
        mutate = random.random() < 0.1
        if random.random() > 0.5:
            if mutate:
                if b[i] == '0':
                    res += '1'
                else:
                    res += '0'
            else:
                res += b[i]
        else:
            if mutate:
                if a[i] == '0':
                    res += '1'
                else:
                    res += '0'
            else:
                res += a[i]
    return res


def decoded(a):
    return (int(eval('0b' + a[:8])), int(eval('0b' + a[8:])))
    
def train_test(a, b):
    dist = math.sqrt( ((a - 150)**2) + ((b - 150)**2) )
    if dist == 0:
        return 1
    return 1.0/dist
        
sample_size = 40
x_cases = range(20, 256)
y_cases = range(20, 256)

first_generation_x = numpy.random.choice(x_cases, sample_size)
first_generation_y = numpy.random.choice(y_cases, sample_size)

gen = []
for i in range(sample_size):
    gen.append((first_generation_x[i], first_generation_y[i]))

trained_aucs = {}
for gen_number in range(30):
    sum_aucs = 0
    gen_result = []
    case_number = 0
    for case in gen:
        case_number += 1
        auc = train_test(case[0], case[1])
        trained_aucs[case] = auc
        gen_result.append((case, auc))
        sum_aucs += auc
    aucs_sorted = OrderedDict(sorted(trained_aucs.items(), key=lambda t: t[1], reverse=True))
    print('SUM', sum_aucs, gen_number)
    new_gen = []
    cases, aucs = zip(*gen_result)
    for i in range(int(sample_size / 2)):
        parents = numpy.random.choice(range(sample_size), 2, replace=False, p= [x / sum_aucs for x in aucs])
        parent1 = encoded(cases[parents[0]])
        parent2 = encoded(cases[parents[1]])
        child1 = decoded(uniform_crossover(parent1, parent2))
        child2 = decoded(uniform_crossover(parent1, parent2))
        new_gen.append(child1)
        new_gen.append(child2)
    gen = new_gen