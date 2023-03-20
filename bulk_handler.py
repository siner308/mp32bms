from os import system, path, listdir
from multiprocessing import Pool


def file_list(dirname):
    base_dir = './bms_files'
    dirpath = f"{base_dir}/{dirname}"
    if not path.exists(dirpath) or not path.isdir(dirpath):
        return []
    files = listdir(f"{base_dir}/{dirname}")
    return files


def command(filename: str):
    dirname = filename[0].upper()
    source = f"./bms_files/{dirname}/{filename}"
    output = f"./mp3/{dirname}/{filename[:-4:]}.mp3"
    system(f"./bms2mp3 \"{source}\" \"{output}\"")


if __name__ == "__main__":
    targets = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]
    for target in targets:
        print("Processing", target)

        files = file_list(target)
        with Pool(4) as p:
            p.map(command, files)
