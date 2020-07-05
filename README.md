[![Build Status](https://travis-ci.com/Marko50/FEUP-DISS.svg?token=zxXmDKs9f8jTTEztxFW7&branch=master)](https://travis-ci.com/Marko50/FEUP-DISS.svg?token=zxXmDKs9f8jTTEztxFW7&branch=master)
[![codecov](https://codecov.io/gh/Marko50/FEUP-DISS/branch/master/graph/badge.svg?token=V9LK9NWX9W)](https://codecov.io/gh/Marko50/FEUP-DISS)

# FEUP-DISS

This repository hosts the **Software library on stream based recommender systems**. This is part of my thesis dissertation done at
the **Faculty of Engineering of the University of Porto**.

## General Overview

![general overview](https://i.ibb.co/1qdmqRV/architecture.png)

## Contents

This software library contains code regarding the implementation of recommendation systems on online platforms. It implements some algorithms, test metrics and data structures, so that it can process incoming information, denoted as a stream of ratings. Each rating can be **implicit** or **explicit** and is given from an user to an item. Explicit ratings are direct evaluations given by users, for example on a 1-10 scale. Implicit ratings can be interpreted as actions users took on items(for example a page click) which can be interpreted as the user *liking* that specific item. Each rating is a **tuple** : *(user_id, item_id, rating)*. For implicit feedback the tuple becomes: *(user_id, item_id)*

Current implemented algorithms:

1. **Explicit User-Based Collaborative Filtering**
2. **Explicit User-Based Clustering Collaborative Filtering**
3. **Implicit User-Based Collaborative Filtering**
4. **Implicit User-Based Clustering Collaborative Filtering**
5. **Implicit Locality-Sensitive-Hashing Item-Based Collaborative Filtering**
6. **Implicit Locality-Sensitive-Hashing User-Based Collaborative Filtering**
7. **Implicit Matrix Factorization**
8. **Explicit Matrix Factorization with matrix preprocessing**
9. **Explicit Matrix Factorization without matrix preprocessing**
10. **Implicit Item-Based Collaborative Filtering**
11. **Implicit Item-Based Clustering Collaborative Filtering**

## Install

Currently the library is deployed on **Pypi**.

`pip install increc`

## Dependencies

For development purposes:

1. **Python3.8**
2. **Pipenv** is required to run this on a virtual environment. `pip install --user pipenv`
3. `python -m pipenv lock -r > requirements.txt`
4. `pipenv install -r requirements.txt`

## Run Tests

This library uses **unittest** for unit testing and **flake8** as its linter.

1. `pipenv run python -m unittest discover -v -p "*_test.py"`
2. `pipenv run flake8`

## Run Examples

Usage examples are located in the `examples` folder.

`pipenv run python -m examples.collaborative_filtering.neighborhood.explicit_feedback.user_based.user_based_clustering "data_set"`

## Documentation

Documentation is created using **Sphinx**.

1. `pipenv run sphinx-apidoc -f -o docs/source algorithms`
2. `pipenv run sphinx-apidoc -f -o docs/source stream`
3. `pipenv run sphinx-apidoc -f -o docs/source data_structures`
4. `pipenv run sphinx-apidoc -f -o docs/source utils`
5. `pipenv run sphinx-apidoc -f -o docs/source graphic`
6. `pipenv run sphinx-build -b html docs/source/ docs/build`

https://marko50.github.io/increc-documentation
