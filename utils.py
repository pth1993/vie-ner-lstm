import codecs
from alphabet import Alphabet
import numpy as np
import cPickle as pickle


def read_conll_format(input_file):
    with codecs.open(input_file, 'r', 'utf-8') as f:
        word_list = []
        chunk_list = []
        pos_list = []
        tag_list = []
        words = []
        chunks = []
        poss = []
        tags = []
        num_sent = 0
        max_length = 0
        for line in f:
            line = line.split()
            if len(line) > 0:
                words.append(map_number_and_punct(line[0].lower()))
                poss.append(line[1])
                chunks.append(line[2])
                tags.append(line[3])
            else:
                word_list.append(words)
                pos_list.append(poss)
                chunk_list.append(chunks)
                tag_list.append(tags)
                sent_length = len(words)
                words = []
                chunks = []
                poss = []
                tags = []
                num_sent += 1
                max_length = max(max_length, sent_length)
    return word_list, pos_list, chunk_list, tag_list, num_sent, max_length


def map_number_and_punct(word):
    if any(char.isdigit() for char in word):
        word = u'<number>'
    elif word in [u',', u'<', u'.', u'>', u'/', u'?', u'..', u'...', u'....', u':', u';', u'"', u"'", u'[', u'{', u']',
                  u'}', u'|', u'\\', u'`', u'~', u'!', u'@', u'#', u'$', u'%', u'^', u'&', u'*', u'(', u')', u'-', u'+',
                  u'=']:
        word = u'<punct>'
    return word


def map_string_2_id_open(string_list, name):
    string_id_list = []
    alphabet_string = Alphabet(name)
    for strings in string_list:
        ids = []
        for string in strings:
            id = alphabet_string.get_index(string)
            ids.append(id)
        string_id_list.append(ids)
    alphabet_string.close()
    return string_id_list, alphabet_string


def map_string_2_id_close(string_list, alphabet_string):
    string_id_list = []
    for strings in string_list:
        ids = []
        for string in strings:
            id = alphabet_string.get_index(string)
            ids.append(id)
        string_id_list.append(ids)
    return string_id_list


def map_string_2_id(pos_list_train, pos_list_dev, pos_list_test, chunk_list_train, chunk_list_dev, chunk_list_test,
                    tag_list_train, tag_list_dev, tag_list_test):
    pos_id_list_train, alphabet_pos = map_string_2_id_open(pos_list_train, 'pos')
    pos_id_list_dev = map_string_2_id_close(pos_list_dev, alphabet_pos)
    pos_id_list_test = map_string_2_id_close(pos_list_test, alphabet_pos)
    chunk_id_list_train, alphabet_chunk = map_string_2_id_open(chunk_list_train, 'chunk')
    chunk_id_list_dev = map_string_2_id_close(chunk_list_dev, alphabet_chunk)
    chunk_id_list_test = map_string_2_id_close(chunk_list_test, alphabet_chunk)
    tag_id_list_train, alphabet_tag = map_string_2_id_open(tag_list_train, 'tag')
    tag_id_list_dev = map_string_2_id_close(tag_list_dev, alphabet_tag)
    tag_id_list_test = map_string_2_id_close(tag_list_test, alphabet_tag)
    return pos_id_list_train, pos_id_list_dev, pos_id_list_test, chunk_id_list_train, chunk_id_list_dev, chunk_id_list_test, \
           tag_id_list_train, tag_id_list_dev, tag_id_list_test, alphabet_pos, alphabet_chunk, alphabet_tag


def padding(input_sequence, max_len):
    pad_sequence = sequence.pad_sequences(input_sequence, maxlen=max_len, padding='post', value=0)
    return pad_sequence


def padding_data(pos_id_list_train, pos_id_list_dev, pos_id_list_test, chunk_id_list_train, chunk_id_list_dev, 
                 chunk_id_list_test, tag_id_list_train, tag_id_list_dev, tag_id_list_test, max_length):
    pos_id_list_train_pad = padding(pos_id_list_train, max_length)
    pos_id_list_dev_pad = padding(pos_id_list_dev, max_length)
    pos_id_list_test_pad = padding(pos_id_list_test, max_length)
    chunk_id_list_train_pad = padding(chunk_id_list_train, max_length)
    chunk_id_list_dev_pad = padding(chunk_id_list_dev, max_length)
    chunk_id_list_test_pad = padding(chunk_id_list_test, max_length)
    tag_id_list_train_pad = padding(tag_id_list_train, max_length)
    tag_id_list_dev_pad = padding(tag_id_list_dev, max_length)
    tag_id_list_test_pad = padding(tag_id_list_test, max_length)
    return pos_id_list_train_pad, pos_id_list_dev_pad, pos_id_list_test_pad, chunk_id_list_train_pad, \
           chunk_id_list_dev_pad, chunk_id_list_test_pad, tag_id_list_train_pad, tag_id_list_dev_pad, \
           tag_id_list_test_pad


def construct_tensor_word(word_sentences, unknown_embedd, embedd_words, embedd_vectors, embedd_dim, max_length):
    X = np.empty([len(word_sentences), max_length, embedd_dim])
    for i in range(len(word_sentences)):
        words = word_sentences[i]
        length = len(words)
        for j in range(length):
            word = words[j].lower()
            try:
                embedd = embedd_vectors[embedd_words.index(word)]
            except:
                embedd = unknown_embedd
            X[i, j, :] = embedd
        # Zero out X after the end of the sequence
        X[i, length:] = np.zeros([1, embedd_dim])
    return X


def construct_tensor_onehot(feature_sentences, max_length, dim):
    X = np.zeros([len(feature_sentences), max_length, dim])
    for i in range(len(feature_sentences)):
        for j in range(len(feature_sentences[i])):
            if feature_sentences[i][j] > 0:
                X[i, j, feature_sentences[i][j]] = 1
    return X


def create_vector_data(word_list_train, word_list_dev, word_list_test, pos_id_list_train, pos_id_list_dev,
                       pos_id_list_test, chunk_id_list_train, chunk_id_list_dev, chunk_id_list_test, tag_id_list_train,
                       tag_id_list_dev, tag_id_list_test, unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                       max_length, dim_pos, dim_chunk, dim_tag):
    word_train = construct_tensor_word(word_list_train, unknown_embedd, embedd_words, embedd_vectors, embedd_dim, 
                                       max_length)
    word_dev = construct_tensor_word(word_list_dev, unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                                     max_length)
    word_test = construct_tensor_word(word_list_test, unknown_embedd, embedd_words, embedd_vectors, embedd_dim,
                                      max_length)
    pos_train = construct_tensor_onehot(pos_id_list_train, max_length, dim_pos)
    pos_dev = construct_tensor_onehot(pos_id_list_dev, max_length, dim_pos)
    pos_test = construct_tensor_onehot(pos_id_list_test, max_length, dim_pos)
    chunk_train = construct_tensor_onehot(chunk_id_list_train, max_length, dim_chunk)
    chunk_dev = construct_tensor_onehot(chunk_id_list_dev, max_length, dim_chunk)
    chunk_test = construct_tensor_onehot(chunk_id_list_test, max_length, dim_chunk)
    tag_train = construct_tensor_onehot(tag_id_list_train, max_length, dim_tag)
    tag_dev = construct_tensor_onehot(tag_id_list_dev, max_length, dim_tag)
    tag_test = construct_tensor_onehot(tag_id_list_test, max_length, dim_tag)
    input_train = word_train
    input_train = np.concatenate((input_train, pos_train), axis=2)
    input_train = np.concatenate((input_train, chunk_train), axis=2)
    output_train = tag_train
    input_dev = word_dev
    input_dev = np.concatenate((input_dev, pos_dev), axis=2)
    input_dev = np.concatenate((input_dev, chunk_dev), axis=2)
    output_dev = tag_dev
    input_test = word_test
    input_test = np.concatenate((input_test, pos_test), axis=2)
    input_test = np.concatenate((input_test, chunk_test), axis=2)
    output_test = tag_test
    return input_train, output_train, input_dev, output_dev, input_test, output_test


def create_data(word_dir, vector_dir, train_dir, dev_dir, test_dir):
    embedd_vectors = np.load(vector_dir)
    with open(word_dir, 'rb') as handle:
        embedd_words = pickle.load(handle)
    embedd_dim = np.shape(embedd_vectors)[1]
    unknown_embedd = np.random.uniform(-0.01, 0.01, [1, embedd_dim])
    word_list_train, pos_list_train, chunk_list_train, tag_list_train, num_sent_train, max_length_train = \
        read_conll_format(train_dir)
    word_list_dev, pos_list_dev, chunk_list_dev, tag_list_dev, num_sent_dev, max_length_dev = \
        read_conll_format(dev_dir)
    word_list_test, pos_list_test, chunk_list_test, tag_list_test, num_sent_test, max_length_test = \
        read_conll_format(test_dir)
    pos_id_list_train, pos_id_list_dev, pos_id_list_test, chunk_id_list_train, chunk_id_list_dev, chunk_id_list_test, \
    tag_id_list_train, tag_id_list_dev, tag_id_list_test, alphabet_pos, alphabet_chunk, alphabet_tag = \
        map_string_2_id(pos_list_train, pos_list_dev, pos_list_test, chunk_list_train, chunk_list_dev, chunk_list_test,
                        tag_list_train, tag_list_dev, tag_list_test)
    max_length = max(max_length_train, max_length_dev, max_length_test)
    input_train, output_train, input_dev, output_dev, input_test, output_test = \
        create_vector_data(word_list_train, word_list_dev, word_list_test, pos_id_list_train, pos_id_list_dev,
                           pos_id_list_test, chunk_id_list_train, chunk_id_list_dev, chunk_id_list_test,
                           tag_id_list_train, tag_id_list_dev, tag_id_list_test, unknown_embedd, embedd_words,
                           embedd_vectors, embedd_dim, max_length, alphabet_pos.size(), alphabet_chunk.size(),
                           alphabet_tag.size())
    return input_train, output_train, input_dev, output_dev, input_test, tag_id_list_test, alphabet_tag, max_length


def predict_to_file(predicts, tests, alphabet_tag, output_file):
    with codecs.open(output_file, 'w', 'utf-8') as f:
        for i in range(len(tests)):
            for j in range(len(tests[i])):
                predict = alphabet_tag.get_instance(predicts[i][j])
                if predict == None:
                    predict = alphabet_tag.get_instance(predicts[i][j] + 1)
                test = alphabet_tag.get_instance(tests[i][j])
                f.write('_' + ' ' + predict + ' ' + test + '\n')
            f.write('\n')


if __name__ == "__main__":
    create_data('../vie-nlp/vie-nlp/embedding/words.pl', '../vie-nlp/vie-nlp/embedding/vectors.npy', 'data/train.txt',
                'data/dev.txt', 'data/dev.txt')
