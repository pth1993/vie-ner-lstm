import utils
import network
import argparse
import numpy as np
from datetime import datetime
from keras.callbacks import EarlyStopping
import subprocess
import shlex

parser = argparse.ArgumentParser()
parser.add_argument("--word_dir", help="word surface dict directory")
parser.add_argument("--vector_dir", help="word vector dict directory")
parser.add_argument("--train_dir", help="training directory")
parser.add_argument("--dev_dir", help="development directory")
parser.add_argument("--test_dir", help="testing directory")
parser.add_argument("--num_lstm_layer", help="number of lstm layer")
parser.add_argument("--num_hidden_node", help="number of hidden node")
parser.add_argument("--dropout", help="dropout number: between 0 and 1")
parser.add_argument("--batch_size", help="batch size for training")
parser.add_argument("--patience", help="patience")
args = parser.parse_args()

word_dir = args.word_dir
vector_dir = args.vector_dir
train_dir = args.train_dir
dev_dir = args.dev_dir
test_dir = args.test_dir
num_lstm_layer = int(args.num_lstm_layer)
num_hidden_node = int(args.num_hidden_node)
dropout = float(args.dropout)
batch_size = int(args.batch_size)
patience = int(args.patience)

startTime = datetime.now()

print 'Loading data...'
input_train, output_train, input_dev, output_dev, input_test, output_test, alphabet_tag, max_length = \
    utils.create_data(word_dir, vector_dir, train_dir, dev_dir, test_dir)
print 'Building model...'
time_step, input_length = np.shape(input_train)[1:]
output_length = np.shape(output_train)[2]
ner_model = network.building_ner(num_lstm_layer, num_hidden_node, dropout, time_step, input_length, output_length)
print 'Model summary...'
print ner_model.summary()
print 'Training model...'
early_stopping = EarlyStopping(patience=patience)
history = ner_model.fit(input_train, output_train, batch_size=batch_size, epochs=1000,
                        validation_data=(input_dev, output_dev), callbacks=[early_stopping])
print 'Testing model...'
answer = ner_model.predict_classes(input_test, batch_size=batch_size)
utils.predict_to_file(answer, output_test, alphabet_tag, 'out.txt')
input = open('out.txt')
p1 = subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input)
p1.wait()
endTime = datetime.now()
print "Running time: "
print (endTime - startTime)
