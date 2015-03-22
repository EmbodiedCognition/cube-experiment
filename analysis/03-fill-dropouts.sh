#!/bin/bash

args=$@

function job {
    python analysis/03-fill-dropouts.py $args -pattern "$1/*block0[$2]/"
}

job f9 01
job f9 23

job 68 01
job 68 23

job ca 01
job ca 23
job ca 04

job d1 02
job d1 03
job d1 14

job 85 02
job 85 13
job 85 04

job af 01
job af 23
job af 45

job 16 01
job 16 23
job 16 45

job 65 02
job 65 13
job 65 45

job 14 01
job 14 23
job 14 04

job c9 01
job c9 23
job c9 04

job 3b 01
job 3b 23
job 3b 04
