#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"

solute=$1
solvent=$2

if [ $# -gt 2 ] 
then
    echo "You can only enter two arguments at max. " 
    echo "Don't worry, I've handled it for you by passing only the first two :)"
fi

python3 cigin_app/run.py --solute "$solute" --solvent "$solvent"

