import midi_to_data as mtd
import sys
import numpy as np

#c960i48n0,43,384n384,60,128n512,65,128

#one hot encoded vector
#   [0<->127,0,1,2,3,4,5,6,7,8,9,c,i,n,,]
#   0<->127 : for instrument and note
#   128<->137 : for 1,2,3,4,5,6,7,8,9
#   138 : c
#   139 : i
#   140 : n
#   141 : ,
#alphabet size = 141 + 1 = 142
ALPHA_SIZE = 142
END_NOTE_VALUE = 127
END_NUMBERS = 137
# ===== UTILS TO CREATE BATCH AND CUT FILES =====

# map the inputs to the function blocks
options = {"c": 138,
           "i": 139,
           "n": 140,
           ",": 141,

           138: "c",
           139: "i",
           140: "n",
           141: ","
           }


def encode_char(char, full_number=False):
    if char.isdigit():
        if full_number:
            val = int(char)
        else:
            val = int(char) + END_NOTE_VALUE +1 #128 + value
    else:
        val = options[char]

    return val

def list_char_convertion(list_char):
    encoded_list = []
    for char in list_char:
        encoded_list.append(encode_char(char=char))

    return encoded_list


def convert_clock_and_instrument(cl_and_inst):
    encoded_list = []

    # convert clock and instrument
    clock, inst = cl_and_inst.split("i")

    # clock convertion
    encoded_list.extend(list_char_convertion(list(clock)))

    # instrument convertion
    encoded_list.append(encode_char(char="i")) #always put a "i" because it was destroyed during the split
    encoded_list.append(encode_char(char=inst, full_number=True))

    return encoded_list


def convert_note(note):
    encoded_list = []
    encoded_list.append(encode_char(char="n")) #always put a "n" because it was destroyed during the split

    note_comma_splitted = note.split(",")

    encoded_list.extend(list_char_convertion(list(note_comma_splitted[0])))#encode time

    encoded_list.append(encode_char(char=",")) #always put a "," because it was destroyed during the split
    encoded_list.append(encode_char(char=note_comma_splitted[1], full_number=True))#encode all the note in one time
    encoded_list.append(encode_char(char=",")) #always put a "," because it was destroyed during the split

    encoded_list.extend(list_char_convertion(list(note_comma_splitted[2])))#encode duration

    return encoded_list

def encode_text(data_string):
    #data_string example : c960i48n0,43,384n384,60,128n512,65,128

    encoded_list = []
    list_data_n_splitted = data_string.split("n")

    #convert clock and instrument to encoded numbers
    cl_and_inst = list_data_n_splitted.pop(0)
    encoded_list.extend(convert_clock_and_instrument(cl_and_inst))

    #convert note to encoded numbers
    for note in list_data_n_splitted:
        encoded_list.extend(convert_note(note))

    return encoded_list


def decode_char(val):
    val = int(val)
    if val >= 0 and val <= END_NOTE_VALUE:
        ret = val #0 <-> 127
    elif val >= END_NUMBERS +1 and val < ALPHA_SIZE:
        ret = options[val] #c i n ,
    else:
        ret = val - (END_NOTE_VALUE+1) #generated value -128

    return str(ret)

# return training data and validation data
def generate_data_for_nn(data_string_list):
    print("Generate data for nn")
    code_text = []
    filesranges = []

    if len(data_string_list) < 10:
        sys.exit("No enough training data has been found. Aborting.")

    print("Generate training data for nn")
    for data_string in data_string_list:
        if data_string:
            start = len(code_text)
            encoded_data_string = encode_text(data_string)
            code_text.extend(encoded_data_string)
            end = len(code_text)
            filesranges.append({"start": start, "end": end})

    print("Generate validation data for nn")
    # For validation, no more than 10% of the entire text

    # 1% of the text is how many files ?
    total_len = len(code_text)
    validation_len = 0
    nb_files = 0
    for file in reversed(filesranges):
        validation_len += file["end"] - file["start"]
        nb_files += 1
        if validation_len > total_len // 100:
            break

    #take as cutoff the start of the -num file
    cutoff = filesranges[-nb_files]["start"]

    valid_text = code_text[cutoff:] #take as validation the last files
    code_text = code_text[:cutoff] # take as code_text the others books

    print("End generation")

    return code_text, valid_text



def print_data_stats(datalen, valilen, epoch_size):
    datalen_mb = datalen/1024.0/1024.0
    valilen_kb = valilen/1024.0
    print("Training text size is {:.2f}MB with {:.2f}KB set aside for validation.".format(datalen_mb, valilen_kb)
          + " There will be {} batches per epoch".format(epoch_size))



def load_midi_files_to_data(directory="./", batch_size=100, seq_len=40):
    data_string_list = mtd.midi_to_data_all_directory(directory)
    code_text, valid_text = generate_data_for_nn(data_string_list)

    # display some stats on the data
    epoch_size = len(code_text) // (batch_size * seq_len)
    print_data_stats(len(code_text), len(valid_text), epoch_size)

    return code_text, valid_text


def print_validation_stats(loss, accuracy):
    print("VALIDATION STATS:                                  loss: {:.5f},       accuracy: {:.5f}".format(loss,accuracy))



def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: one batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."

    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])#output is input shifted by 1

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch




def sample_from_probabilities(probabilities, topn=ALPHA_SIZE):
    """Roll the dice to produce a random integer in the [0..ALPHASIZE] range,
    according to the provided probabilities. If topn is specified, only the
    topn highest probabilities are taken into account.
    :param probabilities: a list of size ALPHASIZE with individual probabilities
    :param topn: the number of highest probabilities to consider. Defaults to all of them.
    :return: a random integer
    """
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(ALPHA_SIZE, 1, p=p)[0]