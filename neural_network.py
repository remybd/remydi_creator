import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import os
import time
import math
import utils_nn as unn
import sys

ALPHA_SIZE = unn.ALPHA_SIZE
CELL_SIZE = 512 # size of neuron layers in a cell : 512 neuron by layers
SEQ_LEN = 40
N_LAYERS = 3
BATCH_SIZE = 100
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCH_SIZE * SEQ_LEN


learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout


# one hot encoded vector
#   [0,1,2,3,4,5,6,7,8,9,c ,i ,n ,,]
# =>[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
#===== ALGO FOR NEURAL NETWORK =====


def create_placeholders():
    print("Create placeholders")
    #create placeholders for nn parameters
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    # create placeholder for data
    X = tf.placeholder(tf.uint8, [None, None], name="X")  # entry : batch de séquences de 30 char
    Xo = tf.one_hot(X, ALPHA_SIZE, 1.0, 0.0)  # one hot encoded : bacth de séquence de 30 char chacun sur ALPHASIZE

    Y_ = tf.placeholder(tf.uint8, [None, None], name="Y_")  # output / sortie
    Yo_ = tf.one_hot(Y_, ALPHA_SIZE, 1.0, 0.0)

    Hin = tf.placeholder(tf.float32, [None, CELL_SIZE * N_LAYERS], name="Hin")  # entry state because recurent network

    return batchsize, X, Y_, Hin, Xo, Yo_



def create_network_model(batchsize, Xo, Yo_, Hin):
    print("Create network model")
    # === Grid Cell ===
    # create the grid of GRU CELLS
    cells = []
    for i in range(N_LAYERS):
        cell = tf.nn.rnn_cell.GRUCell(num_units=CELL_SIZE)
        cells.append(cell)

    mcell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=False)
    Yr, H = tf.nn.dynamic_rnn(mcell, Xo, initial_state=Hin)

    # === Softmax layer ===
    # add the softmax layer at the output
    # one softmax to manage only one time BATCH_SIZE * SEQ_LEN (100 * 40) char
    # simple than 40 softmax who manage each 100 characters
    Yf = tf.reshape(Yr, [-1, CELL_SIZE])
    Ylogits = layers.linear(Yf, ALPHA_SIZE)
    Yf_ = tf.reshape(Yo_, [-1, ALPHA_SIZE])

    Yo = tf.nn.softmax(Ylogits)

    # take the higher prediction of the softmax and give the index (come back from one hot encoding)
    Ypredictions = tf.arg_max(Yo, 1)
    # reshape to separate the sequences fro mthe softmax
    Ypredictions = tf.reshape(Ypredictions, [batchsize, -1])

    # === Loss and Gradient ===
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yf_)
    loss = tf.reshape(loss, [batchsize, -1])
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return train_step, Ypredictions, H, loss, Yo



def create_stats_for_display(loss, Y_, Y):
    print("Create stats for display")
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)

    #input_tensor = tf.cast(tf.equal(Y_, Y), tf.float32)
    #accuracy = tf.reduce_mean(input_tensor=input_tensor)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))

    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])

    return seqloss, batchloss, accuracy, summaries


def init_tensorboard():
    print("Init tensorboard")
    # Init Tensorboard stuff. This will save Tensorboard information into a different
    # folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
    # you can compare training and validation curves visually in Tensorboard.
    timestamp = str(math.trunc(time.time()))
    summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
    validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")
    return summary_writer, validation_writer



def drive_tensorflow(midi_directory="./"):
    #load data
    code_text, valid_text = unn.load_midi_files_to_data(midi_directory, BATCH_SIZE, SEQ_LEN)

    batchsize, X, Y_, Hin, Xo, Yo_ = create_placeholders()
    train_step, Y, H, loss, Yo = create_network_model(batchsize, Xo, Yo_, Hin)#put all the args in the network model

    # Init for saving models. They will be saved into a directory named 'checkpoints'.
    # Only the last checkpoint is kept.
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    saver = tf.train.Saver(max_to_keep=1000)

    summary_writer, validation_writer = init_tensorboard()

    #stats for display
    seqloss, batchloss, accuracy, summaries = create_stats_for_display(loss, Y_, Y)

    #initialize the variables and the tf session
    print("initialize tf session")
    timestamp = str(math.trunc(time.time()))
    inH = np.zeros([BATCH_SIZE, CELL_SIZE * N_LAYERS])  # init all neurons of each cells to 0
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    step = 0

    #training loop
    print("Start training loop")
    for x, y_, epoch in unn.rnn_minibatch_sequencer(code_text, BATCH_SIZE, SEQ_LEN, nb_epochs=10):
        #train the model
        dic = {X:x, Y_:y_, Hin: inH, batchsize: BATCH_SIZE} #x = batch entry; y = batch output, inH = batch initial states

        _, y, outH =  sess.run([train_step, Y, H], feed_dict=dic)

        # log training data for Tensorboard display a mini-batch of sequences (every 50 batches)
        if step % _50_BATCHES == 0:
            feed_dict = {X: x, Y_: y_, Hin: inH, batchsize: BATCH_SIZE}  # no dropout for validation
            y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
            summary_writer.add_summary(smm, step)

        # run a validation step every 50 batches
        # The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
        # so we cut it up and batch the pieces (slightly inaccurate)
        # tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
        if step % _50_BATCHES == 0 and len(valid_text) > 0:

            VALID_SEQLEN = 1 * 1024  # Sequence length for validation. State will be wrong at the start of each sequence.
            bsize = len(valid_text) // VALID_SEQLEN

            print("validation at epoch :" + str(epoch))

            vali_x, vali_y, _ = next(unn.rnn_minibatch_sequencer(valid_text, bsize, VALID_SEQLEN, 1))  # all data in 1 batch
            vali_nullstate = np.zeros([bsize, CELL_SIZE * N_LAYERS])

            feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, batchsize: bsize}
            ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
            #log validation
            unn.print_validation_stats(ls, acc)

            #log tensorboard
            validation_writer.add_summary(smm, step)


        # display a short text generated with the current weights and biases (every 150 batches)
        if step // 3 % _50_BATCHES == 0:
            print("generation txt at epoch :" + str(epoch))
            ry = np.array([[unn.encode_char("0")]])
            rh = np.zeros([1, CELL_SIZE * N_LAYERS])
            for k in range(1000):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, Hin: rh, batchsize: 1})
                rc = unn.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
                print(unn.decode_char(rc), end="")
                ry = np.array([[rc]])
            print("end of generation")


        # save a checkpoint (every 500 batches)
        if step // 10 % _50_BATCHES == 0:
            saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
            print("Saved file: " + saved_file)


        inH = outH #re inject the output in the rnn for the next loop
        step += BATCH_SIZE * SEQ_LEN


def main():
    if len(sys.argv) > 1:
        midi_directory = sys.argv[1]
        if os.path.isdir(midi_directory):
            drive_tensorflow(midi_directory)

    else:
        print("give a midi directory")


if __name__ == "__main__":
    main()