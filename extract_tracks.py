import subprocess
import sys

NUMBER_TRACKS_TO_KEEP = 1
END_OF_FILE_LINE = "\n0, 0, End_of_file"


def midi_to_csv(midi_file_path):
    res = subprocess.run(["midicsv", midi_file_path], universal_newlines = True, stdout=subprocess.PIPE)
    csv_data = res.stdout
    return csv_data

def extract_midi_track(splitted_csv_data, track_num):
    main_sequence = ""
    for line in splitted_csv_data:
        if line.startswith(str(track_num) + ", "):
            main_sequence += "\n" + line

    return main_sequence


def create_correct_header(first_line, n_tracks_to_keep):
    splitted_first_line = first_line.split(", ")
    midi_file_format = splitted_first_line[3]
    tracks_number = str(n_tracks_to_keep + 1)
    division = splitted_first_line[5] #division: the number of clock pulses per quarter note

    correct_header = "0, 0, Header, " + midi_file_format + ", " + tracks_number + ", " + division
    return correct_header

def compile_tracks(header, first_track, others_tracks):
    body = ""
    for track in others_tracks:
        body += track

    return header + first_track + body + END_OF_FILE_LINE


def csv_to_midi(csv_music_string, in_midi_file, n_tracks_to_keep):
    out_file_name = in_midi_file.replace(".mid", "") + "_tracks_0_to_" + str(n_tracks_to_keep+2)
    out_file_name_csv = out_file_name + ".csv"
    out_file_name_midi = out_file_name + ".mid"

    #create the csv file if we want to make some hand modifications
    f = open(out_file_name_csv, "w")
    f.write(csv_music_string)
    f.close()

    #create the new midi file
    res = subprocess.run(["csvmidi", out_file_name_csv, out_file_name_midi])
    return [out_file_name_csv, out_file_name_midi]





def extract_track_from_midi_file(n_tracks_to_keep= NUMBER_TRACKS_TO_KEEP, in_midi_file=""):
    #convert to csv string and select the asked tracks
    csv_data = midi_to_csv(midi_file_path=in_midi_file)
    splitted_csv_data = csv_data.split("\n")

    correct_header = create_correct_header(first_line=splitted_csv_data[0], n_tracks_to_keep=n_tracks_to_keep)
    first_track = extract_midi_track(splitted_csv_data=splitted_csv_data, track_num=1)

    others_tracks = []
    for i in range(2, n_tracks_to_keep+2):
        others_tracks.append(extract_midi_track(splitted_csv_data, track_num=i))

    total_music = compile_tracks(correct_header, first_track, others_tracks)

    [csv_file_name, midi_file_name] = csv_to_midi(total_music, in_midi_file=in_midi_file, n_tracks_to_keep=n_tracks_to_keep)
    print(midi_file_name)



def main():

    if len(sys.argv) > 2:
        n_tracks_to_keep = sys.argv[2]
        midi_file = sys.argv[1]
        extract_track_from_midi_file(int(n_tracks_to_keep), midi_file)
    elif len(sys.argv) > 1:
        midi_file = sys.argv[1]
        extract_track_from_midi_file(in_midi_file=midi_file)
    else:
        print("give a midi file and optionnaly the number of tracks to keep")

if __name__ == "__main__":
    main()