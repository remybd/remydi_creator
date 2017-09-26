import instrument_list as il
import midi_to_data as mtd
import sys, os, glob
import extract_tracks as ext

# === analyse all tracks and find there instruments ===


def get_number_tracks(splitted_csv_data):
    #0, 0, Header, 1, 8, 960
    nb_tracks = "0"
    if "0, 0, Header, " in splitted_csv_data[0]:
        splitted_line = splitted_csv_data[0].split(", ")
        nb_tracks =  int(splitted_line[4])

    return int(nb_tracks)


def add_instrument_to_stats(instrument="0"):
    #search instrument in instrument list
    for i_name, i_list in il.instrument_classification.items():
        if int(instrument) in i_list:
            il.instrument_stats[i_name] += 1
            break


def stats_instrument_one_file(in_midi_file=""):
    try:
        csv_data = ext.midi_to_csv(midi_file_path=in_midi_file)
    except UnicodeDecodeError as err:
        print("Error for file ",in_midi_file, " :", err)
    else:

        splitted_csv_data = csv_data.split("\n")
        number_tracks = get_number_tracks(splitted_csv_data)

        for i in range(2, number_tracks+1):
            track = mtd.extract_midi_track(splitted_csv_data, i)
            instrument = mtd.extract_instrument(track=track)
            instrument = instrument.replace("i","")
            add_instrument_to_stats(instrument)


def stats_instrument_all_directory(in_midi_directory="./"):
    print(glob.glob(in_midi_directory + "**/*.mid", recursive=True))
    for file in glob.glob(in_midi_directory + "**/*.mid", recursive=True):
        file_path = os.path.join(in_midi_directory, file)

        print("stats intruments for :", file_path)
        stats_instrument_one_file(in_midi_file=file_path)




def main():
    if len(sys.argv) > 1:
        midi_file = sys.argv[1]
        if os.path.isdir(midi_file):
            stats_instrument_all_directory(midi_file)
        elif os.path.isfile(midi_file):
            stats_instrument_one_file(in_midi_file=midi_file)

        print(il.instrument_stats)

    else:
        print("give a midi file or a directory")

if __name__ == "__main__":
    main()