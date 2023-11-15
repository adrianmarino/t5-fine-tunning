import os
import glob


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path



def remove_dir(path):
    for file in os.listdir(path):
        file = f'{path}/{file}'
        if os.path.isfile(file):
            os.remove(file)


def recursive_remove_dir(path):
    files = glob.glob(f'{path}/*')
    if len(files) > 0:
        [os.remove(path) for path in files]
    remove_dir(path)

def write(path, content):
    with open(path, "w", newline="") as file:
        file.write(content)
    return path