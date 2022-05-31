#!/bin/bash

rm -r build/ crescendo.egg-info/ dist/
python setup.py sdist
python setup.py install