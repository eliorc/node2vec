from importlib.metadata import version

def is_dated_gensim_version():
    return version("gensim") < '4.0.0'