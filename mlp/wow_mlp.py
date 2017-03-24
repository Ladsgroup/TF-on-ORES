import json
import random
import numpy
import tensorflow as tf

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
for case in res:
    new_case = []
    for i in case[:-1]:
        if i == 'False':
            new_case.append(0)
        elif i == 'True':
            new_case.append(1)
        else:
            new_case.append(float(i))
    label = case[-1]
    if label == 'False':
        label = [0, 1]
    elif label == 'True':
        label = [1, 0]
    else:
        continue
    if random.random() < 0.2:
        test_features.append(new_case)
        test_labels.append(label)
    else:
        labels.append(label)
        new_res.append(new_case)
dataset = DataSet(numpy.array(new_res), numpy.array(labels))
print(new_res[0], labels[0])
test = DataSet(numpy.array(test_features), numpy.array(test_labels))
# Parameters
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = len(res[0]) - 1
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


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
        total_batch = int(len(res)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = dataset.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test.cases, y: test.labels}))
