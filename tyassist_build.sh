#!/bin/bash

if [ "$1" ]; then
  TARGET="--target $1"
else
  echo "[warn] you didn't specify target in command, that will use the target specified in config file."
fi

mkdir -p logs
LOG_FILE="logs/tyassist-build-$1-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $DENGINE_ROOT/tyassist/tyassist.py build $TARGET -c config.yml 2>&1 | tee $LOG_FILE"
python3 $DENGINE_ROOT/tyassist/tyassist.py build $TARGET -c config.yml 2>&1 | tee $LOG_FILE