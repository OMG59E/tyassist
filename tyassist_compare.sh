#!/bin/bash

if [ "$1" ]; then
  TARGET="--target $1"
else
  echo "[warn] you didn't specify target in command, that will use the target specified in config file."
fi

python3 $DENGINE_ROOT/tyassist/tyassist.py compare $TARGET -c config.yml --log_dir ./logs