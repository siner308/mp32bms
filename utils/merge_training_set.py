import pandas


def merge_training_set():
    training_sets = [
        "./training_set_E.csv",
        "./training_set_m.csv",
        "./training_set_S.csv",
    ]

    training_set = pandas.DataFrame()
    for training_set_path in training_sets:
        training_set = pandas.concat([training_set, pandas.read_csv(training_set_path)])

    training_set.to_csv("../training_set.csv", index=False)


if __name__ == "__main__":
    merge_training_set()
