config = {}


def init(**kwargs):
    if "config" in kwargs:
        for key, value in kwargs["config"].items():
            config[key] = value


def watch(*args):
    return


def log(*args):
    return


def Object3D(*args):
    return


def finish():
    return
