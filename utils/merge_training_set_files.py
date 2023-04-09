import pandas


def merge_training_set():
    training_sets = [
        "./training_set_ten_seconds_A.csv",
        "./training_set_ten_seconds_B.csv",
        "./training_set_ten_seconds_C.csv",
        "./training_set_ten_seconds_D.csv",
        "./training_set_ten_seconds_E.csv",
        "./training_set_ten_seconds_F.csv",
        "./training_set_ten_seconds_G.csv",
        "./training_set_ten_seconds_H.csv",
        "./training_set_ten_seconds_I.csv",
        "./training_set_ten_seconds_J.csv",
        "./training_set_ten_seconds_K.csv",
        "./training_set_ten_seconds_L.csv",
        "./training_set_ten_seconds_M.csv",
        "./training_set_ten_seconds_N.csv",
        "./training_set_ten_seconds_O.csv",
        "./training_set_ten_seconds_P.csv",
        "./training_set_ten_seconds_Q.csv",
        "./training_set_ten_seconds_R.csv",
        "./training_set_ten_seconds_S.csv",
        "./training_set_ten_seconds_T.csv",
        "./training_set_ten_seconds_U.csv",
        "./training_set_ten_seconds_V.csv",
        "./training_set_ten_seconds_W.csv",
        "./training_set_ten_seconds_X.csv",
        "./training_set_ten_seconds_Y.csv",
        "./training_set_ten_seconds_Z.csv",
    ]

    training_set = pandas.DataFrame()
    for training_set_path in training_sets:
        training_set = pandas.concat([training_set, pandas.read_csv(training_set_path)])

    training_set.to_csv("./training_set_ten_seconds.csv", index=False)


if __name__ == "__main__":
    merge_training_set()
