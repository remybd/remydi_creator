import tensorflow as tf
import numpy as np
import os
import utils_nn as unn
import sys
import neural_network_train as nn_train
import data_to_midi as dtm

ALPHA_SIZE = unn.ALPHA_SIZE
CELL_SIZE = nn_train.CELL_SIZE # size of neuron layers in a cell : 512 neuron by layers
N_LAYERS = 3


#one hot encoded vector
#   [0<->127,0,1,2,3,4,5,6,7,8,9,c,i,n,,]
#   0<->127 : for instrument and note
#   128<->137 : for 1,2,3,4,5,6,7,8,9
#   138 : c
#   139 : i
#   140 : n
#   141 : ,
#alphabet size = 141 + 1 = 142


#checkpoints
DEFAULT_CHECKPOINT = "checkpoints/rnn_train_1501793298"
DEFAULT_NB_ITER = "32000000"
MAX_CHAR_GENERATED = 5000

def generate_char(sess, y, h):
    yo, h = sess.run(['Softmax:0', 'rnn/while/Exit_2:0'], feed_dict={'X:0': y, 'Hin:0': h, 'batchsize:0': 1})
    # yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'Hin:0': h, 'batchsize:0': 1})

    # If sampling is be done from the topn most likely characters, the generated text
    # is more credible. If topn is not set, it defaults to the full
    # distribution (ALPHASIZE)

    c = unn.sample_from_probabilities(yo, topn=5)
    y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
    c = unn.decode_char(c)

    return y, h, c


def run_trained_rnn(checkpoint=DEFAULT_CHECKPOINT, nb_iter=DEFAULT_NB_ITER):

    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(checkpoint + '-0.meta')
        new_saver.restore(sess, checkpoint + "-" + nb_iter)
        y = np.array([[unn.encode_char("1")]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

        # initial values
        h = np.zeros([1, CELL_SIZE * N_LAYERS], dtype=np.float32)  # [ BATCHSIZE, CELL_SIZE * NLAYERS]

        #search the begining of a track
        print("searching a track ...")
        c = ""
        while(c != "c"):
            y, h, c = generate_char(sess, y, h)

        c = ""
        while (c != "c"):
            y, h, c = generate_char(sess, y, h)

        #generate while we have notes
        print("start generating a true track")
        track = "c"
        c = ""
        nb_char = 1
        already_an_i = False
        while (c != "c" and nb_char < MAX_CHAR_GENERATED):
            track += c
            y, h, c = generate_char(sess, y, h)
            nb_char += 1

            if nb_char % 200 == 0:
                print(str(nb_char) + " chars generated")

            #if we find another instrument before the start of a new track
            if c == "i":
                if already_an_i:
                    break
                else:
                    already_an_i = True

        track = track[:track.rfind('n')]
        print(track)
        dtm.data_to_midi(track)




def main():
    if len(sys.argv) > 2 and os.path.isfile(sys.argv[1] + "-0.meta")\
            and os.path.isfile(sys.argv[1] + "-" + sys.argv[2] + ".meta"):
        checkpoint = sys.argv[1]
        nb_iter = sys.argv[2]
        run_trained_rnn(checkpoint, nb_iter)
    else:
        print("no checkpoints given, use : " + DEFAULT_CHECKPOINT)
        run_trained_rnn()


if __name__ == "__main__":
    main()


