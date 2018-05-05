from distutils.core import setup

setup(
    name='node2vec',
    packages=['node2vec'],
    version='0.1.2',
    description='Implementation of the node2vec algorithm.',
    author='Elior Cohen',
    author_email='',
    license='MIT',
    url='https://github.com/eliorc/node2vec',
    install_requires=[
        'networkx',
        'gensim',
        'numpy',
        'tqdm',
        'joblib'
    ],
    keywords=['machine learning', 'embeddings'],
)