import sys, os
import extract_tracks as ext
import midi_to_data as mtd
import uuid

'''
data to keep
in the header :     clock pulse
only the first track for the moment
in the track :
    instrument : Program_c ->ex: i,48
    notes:
        time
        note

c<clock pulse>  
i<instrument>
n<time-start>,<note>,<duration>

complete example:

c960
i48
n0,43,384
n384,60,128
n512,65,128

output from the neural network:
c960i48n0,43,384n384,60,128n512,65,128

become
0, 0, Header, 1, 2, 960
1, 0, Start_track
1, 0, Time_signature, 4, 2, 24, 8
1, 0, End_track
2, 0, Start_track
2, 0, MIDI_port, 0
2, 0, Program_c, 0, 48
2, 0, Note_on_c, 0, 43, 110
2, 384, Note_on_c, 0, 43, 0
2, 384, Note_on_c, 0, 60, 110
2, 512, Note_on_c, 0, 60, 0
2, 512, Note_on_c, 0, 65, 110
2, 640, Note_on_c, 0, 65, 0
2, 640, End_track
0, 0, End_of_file

one hot encoded vector
[0<->127,0,1,2,3,4,5,6,7,8,9,c,i,n,,]
0<->127 : for instrument and note'''


SEQUENCE_NUMBER = 2
CLASSICAL_VELOCITY = "100"

FIXED_FIRST_SEQUENCE = "1, 0, Start_track\n" \
                       "1, 0, Time_signature, 4, 2, 24, 8\n" \
                       "1, 0, End_track"


def create_fixed_start_track(track_num=SEQUENCE_NUMBER):
    return str(track_num) + ", 0, Start_track\n" \
           + str(track_num) + ", 0, MIDI_port, 0"

def create_fixed_end_track(track_num=SEQUENCE_NUMBER, last_time="0"):
    return str(track_num) + ", " + last_time + ", End_track"

def create_instrument_and_header(data_string_n_splitted="", track_num=SEQUENCE_NUMBER):
    #c960i48 =>
    #2, 0, Program_c, 0, 48
    #0, 0, Header, 1, 2, 192

    data = data_string_n_splitted[0]
    clock_pulse_data, instru_data = data.split("i")

    #find instrument
    instrument_num = "1" #piano by default
    if instru_data:
        instrument_num = instru_data

    instrument = str(track_num) + ", 0, Program_c, 0, " + instrument_num

    #find clock pulse and create header
    clock_pulse = "500"
    if clock_pulse_data.startswith("c"):
        clock_pulse = clock_pulse_data.replace("c","")

    header = "0, 0, Header, 1, " + str(track_num) + ", " + clock_pulse


    return instrument, header


def compile_music_string(header, start_track, instrument, note_list, end_track):
    music_string = header + "\n"\
                   + FIXED_FIRST_SEQUENCE + "\n"\
                   + start_track + "\n" + instrument

    for note in note_list:
        music_string += "\n" + note

    music_string += "\n" + end_track + ext.END_OF_FILE_LINE
    return music_string


def create_note_list(data_string_splitted,track_num=SEQUENCE_NUMBER):
    data_string_splitted.pop(0) #delete the instrument

    unordered_note_list = []
    string_track_num = str(track_num)+", "

    for note in data_string_splitted:
        note_data = note.split(",")
        if len(note_data) == 3:
            start_time = note_data[0]
            note = note_data[1]
            duration = note_data[2]

            end_time = str(int(start_time) + int(duration))

            unordered_note_list.append((int(start_time),
                            string_track_num + start_time + ", Note_on_c, 0, " + note + ", " + CLASSICAL_VELOCITY))
            unordered_note_list.append((int(end_time),
                            string_track_num + end_time + ", Note_on_c, 0, " + note + ", 0"))

    sorted_note_list = sorted(unordered_note_list, key=lambda note: note[0])

    note_list = []
    for note in sorted_note_list:
        note_list.append(note[1])

    last_time = sorted_note_list[len(sorted_note_list)-1][0]

    return note_list, str(last_time)



def data_to_midi(data_string=""):
    data_string_n_splitted = data_string.split("n")

    #create the elements to rebuild the csv file
    start_track = create_fixed_start_track(2)
    instrument, header = create_instrument_and_header(data_string_n_splitted=data_string_n_splitted, track_num=2)
    note_list, last_time = create_note_list(data_string_n_splitted,track_num=2)
    end_track = create_fixed_end_track(track_num=2, last_time=last_time)

    music_string = compile_music_string(header, start_track, instrument, note_list, end_track)

    if not os.path.exists("midi_generated"):
        os.mkdir("midi_generated")

    unique_filename = str(uuid.uuid4())
    ext.csv_to_midi(music_string, "midi_generated/" + unique_filename, 1)


def main():
    if len(sys.argv) > 1:
        midi_file = sys.argv[1]
        if os.path.isfile(midi_file):
            data_string = mtd.midi_to_data(in_midi_file=midi_file)
            data_to_midi(data_string)
    else:
        print("give a midi file")


if __name__ == "__main__":
    main()