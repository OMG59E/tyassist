#!/bin/bash

log_highlight() {
  echo -e "\e[30;31m"$1"$(tput sgr0)"
}

if [ ! $NNP ]; then
  log_highlight "[warn] you didn't export NNP in environment variable, that will use the target specified in config file."
fi

if [ ! $BACKEND ]; then
  BACKEND=iss
  log_highlight "[warn] you didn't export BACKEND in environment variable, that will use default backend iss."
fi

mkdir -p logs
LOG_FILE="logs/tyassist-test-$NNP-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $DENGINE_ROOT/tyassist/tyassist.py test --target $NNP --backend $BACKEND -c config.yml 2>&1 | tee $LOG_FILE"
python3 $DENGINE_ROOT/tyassist/tyassist.py test --target $NNP --backend $BACKEND -c config.yml 2>&1 | tee $LOG_FILE