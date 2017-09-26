import sys, os, glob
import extract_tracks as ext
import instrument_list as il
import instruments_stats as istats

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
0, 0, Header, 1, 2, 960
2, 0, Start_track
2, 0, MIDI_port, 0
2, 0, Title_t, "Golden Sun Overworld"
2, 0, Program_c, 0, 48
2, 0, Control_c, 0, 7, 127
2, 0, Control_c, 0, 10, 64
2, 0, Program_c, 0, 48
2, 0, Control_c, 0, 7, 127
2, 0, Control_c, 0, 10, 64
2, 0, Note_on_c, 0, 43, 110
2, 384, Note_on_c, 0, 43, 0
2, 384, Note_on_c, 0, 60, 110
2, 512, Note_on_c, 0, 60, 0
2, 512, Note_on_c, 0, 65, 110


become
c960
i48
n0,43,384
n384,60,128
n512,65,128

feed in the neural network:
c960i48n0,43,384n384,60,128n512,65,128

# one hot encoded vector
#   [0,1,2,3,4,5,6,7,8,9,c ,i ,n ,,]
'''
'''
in a second time:
    tempo
    on note : velocity
    specificity instrument : Control_c -> s,7,127
'''

SEQUENCE_NUMBER = 2

def extract_midi_track(splitted_csv_data, track_num):
    track = []
    for line in splitted_csv_data:
        if line.startswith(str(track_num) + ", "):
            track.append(line)

    return track


def extract_instrument(track):
    instrument = "i1" #piano
    for line in track:
        if ", Program_c, " in line:
            splitted_line = line.split(", ")
            instrument = "i" + splitted_line[4]
            break

    return instrument

def extract_clock_pulse(splitted_csv_data):
    #0, 0, Header, 1, 2, 960
    clock_pulse = "c500"
    if "0, 0, Header, " in splitted_csv_data[0]:
        splitted_line = splitted_csv_data[0].split(", ")
        clock_pulse = "c" + splitted_line[5]

    return clock_pulse

def extract_notes(track):
    #2, 384, Note_on_c, 0, 60, 110
    #2, 512, Note_on_c, 0, 60, 0
    note_list = []
    for line in track:
            if ", Note_on_c, " in line:
                splitted_line = line.split(", ")
                start_time = splitted_line[1]
                note = splitted_line[4]
                velocity = splitted_line[5]

                note_list.append([start_time, note, velocity])

    return note_list


def find_end_notes(note_list):
    note_list_fusion = []
    i = 0
    note_list_size = len(note_list)
    for note_data in note_list:
        if note_data[2] != "0": #not velocity to 0
            start_time = note_data[0]
            note = note_data[1]
            duration = 1

            #search the end of the note
            for j in range(i+1, note_list_size):
                if note_list[j][1] == note:
                    stop_time = note_list[j][0]
                    duration = int(stop_time) - int(start_time)
                    break

            #now we have, the start, the end and the note
            note_list_fusion.append("n" + start_time + "," + note + "," + str(duration))
        i += 1

    return note_list_fusion


def format_data_to_string(clock_pulse, instrument,note_list):
    data_string = clock_pulse +instrument
    for note in note_list:
        data_string += note

    return data_string


#=== analyse one track (the 2nd) ===


def midi_to_data(in_midi_file=""):
    try:
        csv_data = ext.midi_to_csv(midi_file_path=in_midi_file)
    except UnicodeDecodeError as err:
        print("Error for file ",in_midi_file, " :", err)
    else:

        splitted_csv_data = csv_data.split("\n")
        track = extract_midi_track(splitted_csv_data, SEQUENCE_NUMBER)

        clock_pulse = extract_clock_pulse(splitted_csv_data=splitted_csv_data)
        instrument = extract_instrument(track=track)
        note_list = extract_notes(track=track)
        note_list_fusion = find_end_notes(note_list)

        data_string = format_data_to_string(clock_pulse, instrument, note_list_fusion)
        return data_string

def midi_to_data_all_directory(in_midi_directory="./"):
    data_string_list = []

    for file in glob.glob(in_midi_directory + "**/*.mid", recursive=True):
        file_path = os.path.join(in_midi_directory, file)

        print("midi to data for :", file_path)
        data_string = midi_to_data(in_midi_file=file_path)
        data_string_list.append(data_string)
    return data_string_list


#=== data for specific instruments ==


def inst_to_data(in_midi_file="", instrument_name="piano"):
    data_string_list = []

    try:
        csv_data = ext.midi_to_csv(midi_file_path=in_midi_file)
    except UnicodeDecodeError as err:
        print("Error for file ",in_midi_file, " :", err)
    else:
        splitted_csv_data = csv_data.split("\n")
        number_tracks = istats.get_number_tracks(splitted_csv_data)
        instrument_numbers = il.instrument_classification[instrument_name]

        #search the track playing the asked instrument
        #those tracks are converted to data and added to data_string_list
        for i in range(2, number_tracks + 1):
            track = extract_midi_track(splitted_csv_data, i)
            instrument = extract_instrument(track=track)

            instrument_num = int(instrument.replace("i", ""))
            if instrument_num in instrument_numbers:
                clock_pulse = extract_clock_pulse(splitted_csv_data=splitted_csv_data)
                note_list = extract_notes(track=track)
                note_list_fusion = find_end_notes(note_list)

                data_string = format_data_to_string(clock_pulse, instrument, note_list_fusion)
                data_string_list.append(data_string)

    return data_string_list



def inst_to_data_all_directory(in_midi_directory="./", instrument_name="piano"):
    data_string_list = []

    for file in glob.glob(in_midi_directory + "**/*.mid", recursive=True):
        file_path = os.path.join(in_midi_directory, file)

        print("instrument to data for :", file_path)
        datas_string = inst_to_data(in_midi_file=file_path, instrument_name=instrument_name)
        data_string_list.extend(datas_string)

    return data_string_list



def main():

    if len(sys.argv) == 3:
        instrument_name = sys.argv[2]
        if instrument_name not in il.instrument_stats.keys():
            sys.exit("instrument name not in the list : " + str(il.instrument_stats.keys()))

        midi_file = sys.argv[1]
        if os.path.isdir(midi_file):
            inst_to_data_all_directory(in_midi_directory=midi_file, instrument_name=instrument_name)
        elif os.path.isfile(midi_file):
            inst_to_data(in_midi_file=midi_file, instrument_name=instrument_name)
        else:
            sys.exit("give a midi file or a directory")

    elif len(sys.argv) == 2:
        midi_file = sys.argv[1]
        if os.path.isdir(midi_file):
            midi_to_data_all_directory(in_midi_directory=midi_file)
        elif os.path.isfile(midi_file):
            midi_to_data(in_midi_file=midi_file)
        else:
            sys.exit("give a midi file or a directory")
    else:
        sys.exit("give a midi file or a directory")


if __name__ == "__main__":
    main()