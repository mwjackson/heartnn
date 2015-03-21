import csv
from pprint import pprint
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.structure import TanhLayer, SoftmaxLayer

from pybrain.utilities import percentError
import random

from sklearn.metrics import precision_score, recall_score, confusion_matrix

ds = ClassificationDataSet(13, nb_classes=2, class_labels=['healthy', 'heart disease'])

with open('heart_data_norm.txt', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader, None)  # skip header
    rows = [r for r in reader]
    random.shuffle(rows)  # randomly shuffle the data

# inspect a row from file
pprint(rows[:1])

# add rows to dataset
for row in rows:
    ds.appendLinked(row[:13], row[13:])

# convert from single bool, to 2 mutually exclusive categories ['not_heartdisease', 'heartdisease']
ds._convertToOneOfMany()

# check what we have
print ds.calculateStatistics()
print ds.getLinked(0)

# build network
# by default hidden layer is sigmoid
net = buildNetwork(ds.indim, 10, ds.outdim, bias=True, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
test_ds, train_ds = ds.splitWithProportion(0.15)

# create backprop trainer
trainer = BackpropTrainer(net, train_ds, learningrate=0.01, lrdecay=1.0, momentum=0.1, verbose=True, batchlearning=False, weightdecay=0.0)

# train until end
errors = trainer.trainUntilConvergence(maxEpochs=1000, verbose=True, continueEpochs=10, validationProportion=0.15)

train_result = trainer.testOnClassData()
test_result = trainer.testOnClassData(dataset=test_ds)

print 'total epochs: {0} training error: {1}% test error: {2}%'.format(
    trainer.totalepochs, percentError(train_result, [0, 1]), percentError(test_result, [0, 1]))

predicated_values = test_result
actual_values = [int(v[1]) for v in list(test_ds['target'])]  # convert categorical back to single var

print 'actual:'
print actual_values
print 'predicted:'
print predicated_values
print 'precision: {0}'.format(str(precision_score(actual_values, predicated_values)))
print 'recall: {0}'.format(str(recall_score(actual_values, predicated_values)))
print 'confusion matrix:'
print confusion_matrix(actual_values, predicated_values)