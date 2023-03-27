import math

import numpy as np
import librosa
from librosa.feature.rhythm import tempo as get_tempo
import json

import pandas

from file_utils import get_mp3_file_list, get_bme_file_list, get_file_list


def get_input(mp3_file, duration_from_bme):
    """

    :param mp3_file:
    :return: onset_per_beats, duration, tempo
    """
    y, sr = librosa.load(mp3_file)
    y = librosa.util.normalize(y)
    duration = librosa.get_duration(y=y, sr=sr, center=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, center=True, aggregate=np.median)
    times = librosa.times_like(np.abs(librosa.stft(y)))

    onsets = []
    onset_per_seconds = []
    start = 0
    times_index = 0
    onset_index = 0
    while True:
        if times_index >= len(times):
            break

        unit_time = int(100 * times[times_index])
        if unit_time > 100 * duration_from_bme:
            break

        if unit_time == start:
            onset_per_seconds.append(int(10 * onset_env[onset_index]))
            onset_index += 1
            times_index += 1
        else:
            onset_per_seconds.append(0)

        start = start + 1
        if start % 100 == 0:
            onsets.append(onset_per_seconds)
            onset_per_seconds = []

    onsets.append(onset_per_seconds + [0] * (100 - len(onset_per_seconds)))

    return onsets, duration


def get_output(bme_file) -> (list[list[str]], float, int, int):
    """
    bpm이 변하는 곡이라면, 학습에 포함시키지 말자.
    :param bme_file:
    :return:
    """
    # load json
    jsonfile = open(bme_file, mode='r')
    jsondata = json.load(jsonfile)
    notecharts = jsondata['_notes']
    duration = jsondata['_duration']
    difficulty = jsondata["_songInfo"]["difficulty"]
    level = jsondata["_songInfo"]["level"]

    # times_per_second_list = [[0] * 100] * int(duration + 1)
    columns_per_seconds_list = [["0" for _ in range(100)] for _ in range(int(duration) + 1)]

    for notechart in notecharts:
        time = notechart['time']
        column = notechart['column'] if notechart['column'] != 'SC' else 'S'
        second = int(time)
        float_index = int(100 * (time - second))
        # times_per_second_list[second][float_index] = time
        target_column = columns_per_seconds_list[second][float_index]
        if target_column == "0":
            columns_per_seconds_list[second][float_index] = column
        else:
            columns_per_seconds_list[second][float_index] = "".join(sorted(
                columns_per_seconds_list[second][float_index] + column))

    return columns_per_seconds_list, duration, difficulty, level


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
        columns_per_seconds_list, duration_from_bme, difficulty, level = get_output(
            f"./bme_files/M/{target_bme_file}")

        # get input
        onsets, duration_from_mp3 = get_input(f"./mp3_files/M/{mp3_file}", duration_from_bme)

        for i in range(len(onsets)):
            onset = onsets[i]
            columns = columns_per_seconds_list[i]
            input_duration = int(100 * duration_from_mp3)
            input_difficulty = difficulty
            input_level = level
            output_duration = int(100 * duration_from_bme)
            name = mp3_file.replace('.mp3', '')
            new_df = pandas.DataFrame(
                [[input_duration, input_difficulty, input_level, output_duration, name]],
                columns=['input_duration', 'input_difficulty', 'input_level', 'output_duration', 'name'],
            )

            new_df = pandas.concat([new_df, pandas.DataFrame(onset).transpose().add_prefix("input_onset_")], axis=1)
            new_df = pandas.concat([new_df, pandas.DataFrame(columns).transpose().add_prefix("output_columns_")],
                                   axis=1)

            # append to dataframe
            dataframe = pandas.concat([dataframe, new_df]) if not dataframe.empty else new_df
        print(f"✅ {name}")
    return dataframe


def trainingset_to_csv():
    dataframe = extract_trainingset()
    dataframe.to_csv("training_set.csv", index=False)


if __name__ == "__main__":
    trainingset_to_csv()