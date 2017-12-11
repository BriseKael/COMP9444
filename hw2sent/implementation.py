import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile

lstm_size = 40
batch_size = 100

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    if os.path.exists(os.path.join(os.path.dirname(__file__), "Data.npy")):
        print("loading saved parsed data, to reparse, delete 'Data.npy'")
        data = np.load("Data.npy")
    else:
        # 1. check_file (modified from imdb_sentiment_data.py in stage1)
        filename = "reviews.tar.gz"
        expected_bytes = 14839260
        if not os.path.exists(filename):
            print("please make sure {0} exists in the current directory".format(filename))
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print("Found and verified", filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                "File {0} didn't have the expected size. Please ensure you have downloaded the assignment files correctly".format(filename))
        
        # 2. extract_data (modified from imdb_sentiment_data.py in stage1)
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "data/")):
            with tarfile.open(filename, "r") as tarball:
                dir = os.path.dirname(__file__)
                tarball.extractall(os.path.join(dir, "data/"))
                
        # 3. read_data (modified from imdb_sentiment_data.py in stage1)
        print("READING data")
        dir = os.path.dirname(__file__)
        file_paths = glob.glob(os.path.join(dir, "data/pos/*"))
        file_paths.extend(glob.glob(os.path.join(dir, "data/neg/*")))
        print("Parsing %s files" % len(file_paths))

        reviews = list()
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                words = [word.lower() for word in file.read().split()]
                v_words = [glove_dict[word] if word in glove_dict else glove_dict['UNK'] for word in words]

                if len(v_words) < 40:
                    v_words += [glove_dict['UNK']] * (40 - len(v_words))
                    
                reviews.append(v_words[:40])

        data = np.array(reviews)
        np.save("Data", data)
        
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    if os.path.exists(os.path.join(os.path.dirname(__file__), "Embeddings.npy")):
        print("loading saved parsed glove_embeddings, to reparse, delete 'Embeddings.npy'")
        embeddings = np.load("Embeddings.npy").tolist()
        word_index_dict = np.load("WordIndexDict.npy").item()
    else:
        # 1. read glove embeddings in to a list and dict
        print("READING glove_embeddings")
        embeddings = list()
        word_index_dict = dict()

        with open("glove.6B.50d.txt", "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                word = line.split()[0]
                vector = line.split()[1:]

                embeddings.append([float(v) for v in vector])
                word_index_dict[word] = len(word_index_dict) + 1
                
            # 2. add 'UNK' to the first, with index as 0
            # word_index_dict['unk'] = 201535
            # all words in this dict are not isupper() expect this 'UNK'
            embeddings = [[0] * len(embeddings[0])] + embeddings
            word_index_dict['UNK'] = 0

            # 3. save embeddings and word_index_dict
            print("glove_embeddings lines", len(word_index_dict))
            np.save("Embeddings", embeddings)
            np.save("WordIndexDict", word_index_dict)
    
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    # placeholders
    input_data = tf.placeholder(dtype=tf.int32, shape=[batch_size, 40], name="input_data")
    labels = tf.placeholder(dtype=tf.float32,shape=[batch_size, 2], name="labels")
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")

    # embeds
    embedding = tf.convert_to_tensor(glove_embeddings_arr, dtype=tf.float32)
    embeds = tf.Variable(tf.zeros([batch_size, len(glove_embeddings_arr[0])], dtype=tf.float32))
    embeds = tf.nn.embedding_lookup(embedding, input_data)

    # LSTM cell
    # basic
    # add dropout
##    ## complex model
##    def lstm_cell():
##        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
##        return tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout_keep_prob)
##    
##    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])

    ## simple model
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    cell = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=dropout_keep_prob)
    # output
    output, _ = tf.nn.dynamic_rnn(cell, embeds, dtype=tf.float32)
    output = tf.transpose(output, [1,0,2])
    
    last = tf.gather(output, int(output.get_shape()[0]) - 1)

    # prediction
    weight = tf.Variable(tf.truncated_normal(shape=[lstm_size, 2], stddev=0.01))
    bias = tf.Variable(tf.constant(0.1, shape=[2]))
    logits = tf.matmul(last, weight) + bias
    prediction = tf.nn.softmax(logits)

    # accuracy
    predict_labels = tf.argmax(logits, 1)
    real_labels = tf.argmax(labels, 1)
    correct_predict = tf.equal(predict_labels, real_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32), name="accuracy")
    
    # loss
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
    loss = tf.reduce_mean(xentropy, name="loss")
    
    # optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
