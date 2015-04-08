#!/bin/bash

args=$@

function job {
    python analysis/03-fill-dropouts-theano.py $args --pattern "$1/*block0[$2]/"
}

job f9 0123
job 68 0123
job ca 01234
job d1 01234
job 85 01234
job af 012
job af 345
job 16 012
job 16 345
job 65 012
job 65 345
job 14 01234
job c9 01234
job 3b 01234
