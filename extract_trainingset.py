import sys
from multiprocessing import Pool

import numpy as np
import librosa
import json

import pandas

from file_utils import get_file_list

batch_size = 50


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

        unit_time = int(batch_size * times[times_index])
        if unit_time > batch_size * duration_from_bme:
            break

        if unit_time == start:
            onset_per_seconds.append(int(10 * onset_env[onset_index]))
            onset_index += 1
            times_index += 1
        else:
            onset_per_seconds.append(0)

        start = start + 1
        if start % batch_size == 0:
            onsets.append(onset_per_seconds)
            onset_per_seconds = []

    onsets.append(onset_per_seconds + [0] * (batch_size - len(onset_per_seconds)))

    return onsets, duration


def get_output(bme_file) -> (list[list[str]], float, int, int):
    """
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

    # times_per_second_list = [[0] * batch_size] * int(duration + 1)
    columns_per_seconds_list = [[0 for _ in range(batch_size * 8)] for _ in range(int(duration) + 1)]

    for notechart in notecharts:
        time = notechart['time']
        column = notechart['column']
        second = int(time)
        float_index = int(batch_size * (time - second))
        # times_per_second_list[second][float_index] = time
        if column == "1":
            columns_per_seconds_list[second][(float_index * 8)] = 1
        elif column == "2":
            columns_per_seconds_list[second][(float_index * 8) + 1] = 1
        elif column == "3":
            columns_per_seconds_list[second][(float_index * 8) + 2] = 1
        elif column == "4":
            columns_per_seconds_list[second][(float_index * 8) + 3] = 1
        elif column == "5":
            columns_per_seconds_list[second][(float_index * 8) + 4] = 1
        elif column == "6":
            columns_per_seconds_list[second][(float_index * 8) + 5] = 1
        elif column == "7":
            columns_per_seconds_list[second][(float_index * 8) + 6] = 1
        elif column == "SC":
            columns_per_seconds_list[second][(float_index * 8) + 7] = 1
        else:
            raise ValueError(f"column({column}) is not valid")

    return columns_per_seconds_list, duration, difficulty, level


def get_new_df(data):
    onset = data.get('onset')
    columns = data.get('columns')
    # input_duration = int(batch_size * duration_from_mp3)
    # input_difficulty = difficulty
    input_level = data.get('level')
    input_name = data.get('name')
    # output_duration = int(batch_size * duration_from_bme)
    new_df = pandas.DataFrame(
        [[input_level, input_name]],
        columns=['input_level', 'name'],
    )

    new_df = pandas.concat([new_df, pandas.DataFrame(onset).transpose().add_prefix("input_onset_")], axis=1)
    new_df = pandas.concat([new_df, pandas.DataFrame(columns).transpose().add_prefix("output_columns_")],
                           axis=1)

    # append to dataframe
    return new_df


def extract_trainingset(dirname):
    # get training set
    mp3_files = get_file_list('./mp3_files/' + dirname)
    bme_files = get_file_list('./bme_files/' + dirname)

    dataframe = pandas.DataFrame()
    for mp3_file in sorted(mp3_files):
        target_bme_file = mp3_file.replace('.mp3', '.json')

        # if bme file does not exist, skip
        if target_bme_file not in bme_files:
            print("❌ (target not in) " + target_bme_file)
            continue

        # get output
        columns_per_seconds_list, duration_from_bme, difficulty, level = get_output(
            f"./bme_files/{dirname}/{target_bme_file}")

        # get input
        onsets, duration_from_mp3 = get_input(f"./mp3_files/{dirname}/{mp3_file}", duration_from_bme)

        name = mp3_file.replace('.mp3', '')

        with Pool(20) as p:
            data = [dict(onset=onsets[i][::], columns=columns_per_seconds_list[i][::], level=level, name=name) for i in
                    range(min(len(onsets), len(columns_per_seconds_list)))]
            result = p.map(get_new_df, data)

        for df in result:
            dataframe = pandas.concat([dataframe, df], ignore_index=True)
        print(f"✅ {name}")

    dataframe.to_csv(f"training_set_ten_seconds_{dirname}.csv", index=False)


# def trainingset_to_csv():
#     dirs = [
#         "A",
#         "B",
#         "C",
#         "D",
#         "E",
#         "M",
#         "S",
#     ]
#
#     for dir in dirs:
#         extract_trainingset(dir)


if __name__ == "__main__":
    dir = sys.argv[1]
    extract_trainingset(dir)
