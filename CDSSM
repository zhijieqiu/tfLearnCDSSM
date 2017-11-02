import tensorflow as tf
import numpy as np
import re

UNKNOWNKEY = "UNKNOWN"
MAX_LEN = 100
NEG_SAMPLES = 10
CORPUS = "I come from China, I love my country very much."
tri_letter_dict = None
class TrainSample(object):
    def __init__(self, src, target):
        self.src = src
        self.target = target

def build_tri_letter_dict(corpus):
    words = re.split(r'(?:,|;|\s)\s*', corpus)
    words = [w for w in words if w.strip()!=""]
    print(words)
    tri_letter_dict = {UNKNOWNKEY:0}
    for word in words:
        tw = "${}$".format(word)
        for i in range(len(tw) - 3+1):
            tri = tw[i:i+3]
            if tri not in tri_letter_dict:
                tri_letter_dict[tri] = len(tri_letter_dict)

    return tri_letter_dict

#print(build_tri_letter_dict("hello,world"))
tri_letter_dict = build_tri_letter_dict(CORPUS)
#print(tri_letter_dict)
def sentence_to_tri_letter(sentence,tri_letter_dict,max_len = None):
    #tri_letter_dict = {}
    words = re.split(r'(?:,|;|\s)\s*', sentence)
    if max_len != None:
        words = words[:max_len]
    sentence_matrix = []
    for word in words:
        tw = "${}$".format(word)
        init_row = [0]*len(tri_letter_dict)
        for i in range(len(tw)-3+1):
            tri = tw[i:i+3]
            id = tri_letter_dict[UNKNOWNKEY]
            if tri in tri_letter_dict:
                id  = tri_letter_dict[tri]
            init_row[id] += 1

        sentence_matrix.append(init_row)
    for i in range(MAX_LEN-len(sentence_matrix)+1):
        sentence_matrix.append([0]*len(tri_letter_dict))
    sentence_matrix = sentence_matrix[:MAX_LEN]
    return sentence_matrix
sentence_matrix = sentence_to_tri_letter(CORPUS, tri_letter_dict,100)

def corpus_to_features(srcs, targets, tri_letter_dict=None):
    src_features  = []
    tgt_features = []
    if tri_letter_dict is None:
        tri_letter_dict = build_tri_letter_dict(' '.join(srcs)+" "+ ' '.join(targets))
    for s in srcs:
        src_features.append(sentence_to_tri_letter(s, tri_letter_dict,MAX_LEN))
    for t in targets:
        tgt_features.append(sentence_to_tri_letter(t, tri_letter_dict, MAX_LEN))
    return src_features, tgt_features

src_features,tgt_features = corpus_to_features(["I come from China","I love my country very much."],
                                               ["I come from China", "I love my country very much."])



class DataGenerator(object):
    def __init__(self,src_features,tgt_features,batch_size=20,n_echo=5):
        self.src_features = src_features
        self.tgt_features = tgt_features
        #self.train_data = np.array()
        self.feature_length = len(src_features[0])
        self.batch_size = batch_size
        self.train_data = None
        self.train_one_line = None
        self.train_data_set = None
        self.batch_iterator = None

    def generate_train_data(self):
        train_src_all_data = []
        train_tgt_all_data = []
        labels = []
        one_line_features = []
        for i,f in enumerate(self.src_features):
            train_src_all_data.append(f)
            train_tgt_all_data.append(self.tgt_features[i])
            labels.append(1)
            one_line_features.append(f + self.tgt_features[i] + [1])
            for j in range(NEG_SAMPLES):
                #np.random.randint()
                next_int = np.random.randint(0,len(self.src_features),1)[0]
                if next_int == i:
                    continue
                print(next_int)
                train_src_all_data.append(self.src_features[next_int])
                train_tgt_all_data.append(self.tgt_features[next_int])
                labels.append(0)
                one_line_features.append(self.src_features[next_int]+self.tgt_features[next_int]+[0])

        self.train_data = train_src_all_data,train_tgt_all_data,labels,one_line_features

    def WrapperToTFDataSet(self):
        src_data_set = tf.contrib.data.Dataset.from_tensor_slices(np.array(self.train_data[0]))
        src_data_set = src_data_set.batch(self.batch_size)
        tgt_data_set = tf.contrib.data.Dataset.from_tensor_slices(np.array(self.train_data[1]))
        tgt_data_set = tgt_data_set.batch(self.batch_size)
        labels_data_set = tf.contrib.data.Dataset.from_tensor_slices(np.array(self.train_data[2]))
        labels_data_set = labels_data_set.batch(self.batch_size)
        one_line_data_set = tf.contrib.data.Dataset.from_tensor_slices(np.array(self.train_data[3]))
        one_line_data_set = one_line_data_set.batch(self.batch_size)
        self.train_data_set = src_data_set,tgt_data_set,labels_data_set,one_line_data_set

    def NextBatch(self):
        if self.batch_iterator is None:
            one_line_data_set = self.train_data_set[3]
            self.batch_iterator = one_line_data_set.make_one_shot_iterator()
        next_element = self.batch_iterator.get_next()
        return next_element

def cnn_layer(padding,input_layer,kernel_size,activation=None):
    if activation is not None:
        return tf.layers.conv2d(inputs=input_layer,filters=10,kernel_size=kernel_size,padding='valid')
    if activation == 'relu':
        return tf.layers.conv2d(inputs=input_layer,filters=10,kernel_size=kernel_size,padding='valid', activation=tf.nn.relu)
    elif activation == 'tanh':
        return tf.layers.conv2d(inputs=input_layer, filters=10, kernel_size=kernel_size, padding='valid',
                                activation=tf.nn.tanh)
    elif activation == 'sigm':
        return tf.layers.conv2d(inputs=input_layer, filters=10, kernel_size=kernel_size, padding='')
#tf.layers.conv2d()


data_generactor = DataGenerator(src_features,tgt_features,2)
data_generactor.generate_train_data()
data_generactor.WrapperToTFDataSet()
batch_data = data_generactor.NextBatch()
print(batch_data)
print("finished")