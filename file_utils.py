from os import listdir, path


def get_file_list(dirpath):
    if not path.exists(dirpath) or not path.isdir(dirpath):
        return []
    files = listdir(dirpath)
    return files

def get_bms_file_list(dirname):
    base_dir = './bms_files'
    return get_file_list(f"{base_dir}/{dirname}")

def get_bme_file_list(dirname):
    base_dir = './bme_files'
    return get_file_list(f"{base_dir}/{dirname}")


def get_mp3_file_list(dirname):
    base_dir = './mp3_files'
    return get_file_list(f"{base_dir}/{dirname}")
