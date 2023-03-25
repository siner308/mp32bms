import librosa
from librosa.feature.rhythm import tempo as get_tempo
import json

import pandas

from file_utils import get_mp3_file_list, get_bme_file_list, get_file_list


def get_input(mp3_file):
    y, sr = librosa.load(mp3_file)
    y = librosa.util.normalize(y)
    onset = librosa.onset.onset_detect(y=y, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo = get_tempo(y=y, sr=sr)
    return onset, duration, tempo


def get_output(bme_file) -> (bool, list[float], list[int], float, float):
    """
    bpm이 변하는 곡이라면, 학습에 포함시키지 말자.
    :param bme_file:
    :return:
    """
    # load json
    jsonfile = open(bme_file, mode='r')
    jsondata = json.load(jsonfile)
    has_multi_bpm = len(jsondata["_timing"]["_speedcore"]["_segments"]) > 1
    if has_multi_bpm:
        return True, None, None, None, None

    notecharts = jsondata['_notes']
    beats = [int(100 * notechart['beat']) for notechart in notecharts]
    columns = [notechart['column'] for notechart in notecharts]
    duration = jsondata['_duration']
    bpm = jsondata["_timing"]["_speedcore"]["_segments"][0]["bpm"]
    return False, beats, columns, duration, bpm


def extract_trainingset():
    # get training set
    mp3_files = get_file_list('./mp3_files/M')
    bme_files = get_file_list('./bme_files/M')

    dataframe = pandas.DataFrame()
    for mp3_file in sorted(mp3_files):
        target_bme_file = mp3_file.replace('.mp3', '.json')

        # if bme file does not exist, skip
        if target_bme_file not in bme_files:
            print("❌ (target not in) " + target_bme_file)
            continue

        # get output
        has_multi_bpm, beats, columns, duration_from_bme, bpm_from_bme = get_output(f"./bme_files/M/{target_bme_file}")

        # if bpm is changed, skip
        if has_multi_bpm:
            print("❌ (has multi bpm) " + target_bme_file)
            continue

        # get input
        onset, duration_from_mp3, bpm_from_mp3 = get_input(f"./mp3_files/M/{mp3_file}")

        new_df = pandas.DataFrame({
            'input_onset': [onset],
            'input_duration': duration_from_mp3,
            'input_bpm': bpm_from_mp3,
            'output_beats': [beats],
            'output_columns': [columns],
            'output_duration': duration_from_bme,
            'output_bpm': bpm_from_bme,
        })
        # append to dataframe
        dataframe = pandas.concat([dataframe, new_df]) if not dataframe.empty else new_df
        print(f"✅ [bpm: {bpm_from_mp3} or {bpm_from_bme}] [duration: {duration_from_mp3} or {duration_from_bme}] [notes: {len(columns)}] [onset: {len(onset)}] {target_bme_file}")
    return dataframe


def trainingset_to_csv():
    dataframe = extract_trainingset()
    dataframe.to_csv("training_set.csv", index=False)


if __name__ == "__main__":
    trainingset_to_csv()
