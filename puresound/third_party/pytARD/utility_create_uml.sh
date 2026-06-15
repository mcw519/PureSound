#/bin/bash

CURRENT_PATH=$(pwd)

if ! [ -x "$(command -v pylint)" ]; then
  echo 'Error: pylint is not installed.' >&2
  exit 1
fi

pyreverse -o pdf $CURRENT_PATH

echo "Done! Gonna drink my O'boy now"