[![Build Status](https://travis-ci.com/Marko50/FEUP-DISS.svg?token=zxXmDKs9f8jTTEztxFW7&branch=master)](https://travis-ci.com/Marko50/FEUP-DISS.svg?token=zxXmDKs9f8jTTEztxFW7&branch=master)
[![codecov](https://codecov.io/gh/Marko50/FEUP-DISS/branch/master/graph/badge.svg?token=V9LK9NWX9W)](https://codecov.io/gh/Marko50/FEUP-DISS)

# FEUP-DISS

This repository hosts the **Software library on stream based recommender systems**. This is part of my thesis dissertation done at
the **Faculty of Engineering of the University of Porto**.

## General Overview

![general overview](https://i.ibb.co/59SbBS5/diagram.png)

## Contents

This software library contains code regarding the implementation of recommendation systems on online platforms. It implements some algorithms, test metrics and data structures, so that it can process incoming information, denoted as a stream of ratings. Eeach rating can be **implicit** or **explicit** and is given from an user to an item. Explicit ratings are direct evaluations given by users, for example on a 1-10 scale. Implicit ratings can be interpreted as actions users took on items(for example a page click) which can be interpreted as the user *liking* that specific item. Each rating is a **tuple** : *(user_id, item_id, rating)*.

Current implemented algorithms:

1. **Explicit User-Based Collaborative Filtering**
2. **Implicit User-Based Collaborative Filtering**
3. **Implicit Locality-Sensitive-Hashing Item-Based Collaborative Filtering**

## Install

`pip install -i https://test.pypi.org/simple/ IncREC-Marko50`

## Dependencies

To run this library locally:

1. **Python3.7**
2. **Pipenv** is required to run this on a virtual environment. `pip install --user pipenv`

## Run Tests

This library uses **unittest** for unit testing and **flake8** as its linter.

1. `pipenv run python -m unittest discover -v -p "*_test.py"`
2. `pipenv run flake8`

## Run Examples

Usage examples are located in the `examples` folder.

`pipenv run python -m examples.collaborative_filtering.neighborhood.item_based.implicit.implicit_preq_eval_anim "data_set"`

