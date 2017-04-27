import json
import random
import numpy
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve
from collections import OrderedDict
import matplotlib.pyplot as plt

class DataSet(object):
    def __init__(self, cases, labels):
        self._num_examples = cases.shape[0]
        self._cases = cases
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def cases(self):
        return self._cases
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._cases = self._cases[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._cases[start:end], self._labels[start:end]

with open('enwiki.features_damaging.20k_2015.tsv', 'r') as f:
    res = [i.split('\t') for i in f.read().split('\n')]
new_res = []
labels = []
test_features = []
test_labels = []
falses = 0
trues = 0
for case in res:
    new_case = []
    for i in case[1:-1]:
        if i == 'False':
            new_case.append(0)
        elif i == 'True':
            new_case.append(1)
        else:
            new_case.append(float(i))
    label = case[-1]
    if label == 'False':
        falses += 1
        label = [0, 1]
    elif label == 'True':
        trues += 1
        label = [1, 0]
    else:
        continue 
    if random.random() < 0.2:
        test_features.append(new_case)
        test_labels.append(label)
    else:
        labels.append(label)
        new_res.append(new_case)
ratio = int((falses / trues)* 0.8)
length = len(labels)
for i in range(length):
    label = labels[i]
    new_case = new_res[i]
    if label == [1, 0]:
        for i in range(ratio):
            if random.random() < 0.2:
                test_features.append(new_case)
                test_labels.append(label)
            else:
                labels.append(label)
                new_res.append(new_case) 

shuffled_res = []
shuffled_labels = []
for i in range(len(new_res)):
    i = int(len(new_res) * random.random())
    shuffled_labels.append(labels[i])
    shuffled_res.append(new_res[i])

dataset = DataSet(numpy.array(shuffled_res), numpy.array(shuffled_labels))
test = DataSet(numpy.array(test_features), numpy.array(test_labels))
# Parameters
learning_rate = 0.0001
training_epochs = 20
batch_size = 50
display_step = 1

# Network Parameters
n_input = len(res[0]) - 2
n_classes = 2
train_size = len(res)
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

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
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def train_test(n_hidden_1, n_hidden_2):
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    softmax_pred = tf.nn.softmax(pred)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(train_size/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = dataset.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                            y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
        # Test model
        #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        res = softmax_pred.eval({x: test.cases, y: test.labels})
        prrr = []
        test_labell = []
        for case in res:
            prrr.append(case[0])
        for case in test.labels:
            test_labell.append(case[0])
        auc = roc_auc_score(numpy.array(test_labell), numpy.array(prrr))
        return auc
        
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
    print(gen)
    sum_aucs = 0
    gen_result = []
    case_number = 0
    for case in gen:
        case_number += 1
        print(case_number, end=", ", flush=True)
        auc = train_test(case[0], case[1])
        trained_aucs[case] = auc
        gen_result.append((case, auc))
        sum_aucs += auc
    aucs_sorted = OrderedDict(sorted(trained_aucs.items(), key=lambda t: t[1], reverse=True))
    print('SUM', sum_aucs, gen_number, list(aucs_sorted.items())[:int(sample_size/10)])
    new_gen = []
    cases, aucs = zip(*gen_result)
    for i in range(int(sample_size / 2)):
        parents = numpy.random.choice(range(sample_size), 2, replace=False, p=[x / sum_aucs for x in aucs])
        parent1 = encoded(cases[parents[0]])
        parent2 = encoded(cases[parents[1]])
        child1 = decoded(uniform_crossover(parent1, parent2))
        child2 = decoded(uniform_crossover(parent1, parent2))
        new_gen.append(child1)
        new_gen.append(child2)
    gen = new_gen
with open('resss', 'w') as f:
    f.write(json.dumps(aucs_sorted))