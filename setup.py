from setuptools import setup, find_packages

setup(
    name='academic-paper-search-engine',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'streamlit',
        'sentence-transformers',
        'torch',
        'nltk',
        'scikit-learn',
        'spacy[transformers]',
    ],
    dependency_links=[
        'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
    ],
    entry_points={
        'console_scripts': [
            'install-spacy-model = spacy.cli:download --model en_core_web_sm',
        ],
    },
)
