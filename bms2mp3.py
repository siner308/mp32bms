import os.path
from os import system
from multiprocessing import Pool

from file_utils import get_bms_file_list


def command(info: dict):
    dirname = info.get('dir')
    filename = info.get('file')
    source = f"./bms_files/{dirname}/{filename}"
    output = f"./mp3_files/{dirname}/{filename[:-4:]}.mp3"
    if not os.path.exists(output):
        print(output)
        system(f"./bin/bms2mp3 \"{source}\" \"{output}\"")


if __name__ == "__main__":
    targets = [
        # "0-9",
        "A",
        "B",
        "C",
        "D",
        # "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        #"M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        # "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        # "漢字",
    ]
    if not os.path.isdir("./mp3_files"):
        os.mkdir("./mp3_files")

    for target in targets:
        print("Processing", target)

        files = sorted(get_bms_file_list(target))
        if len(files) == 0:
            continue

        if not os.path.isdir(f"./mp3_files/{target}"):
            os.mkdir(f"./mp3_files/{target}")

        with Pool(20) as p:
            p.map(command, list(map(lambda file: dict(file=file, dir=target), files)))
