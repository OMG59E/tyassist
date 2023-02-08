#!/bin/bash

log_highlight() {
  echo -e "\e[30;31m"$1"$(tput sgr0)"
}

if [ ! $NNP ]; then
  log_highlight "[warn] you didn't export NNP in environment variable, that will use the target specified in config file."
fi

mkdir -p logs
LOG_FILE="logs/tyassist-build-$NNP-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $DENGINE_ROOT/tyassist/tyassist.py build --target $NNP -c config.yml 2>&1 | tee $LOG_FILE"
python3 $DENGINE_ROOT/tyassist/tyassist.py build --target $NNP -c config.yml 2>&1 | tee $LOG_FILE