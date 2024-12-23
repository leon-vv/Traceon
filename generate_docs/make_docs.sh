#!/bin/bash

VERSION=$(python3 -c "from importlib.metadata import version; print(version('traceon'))")
DIR=../docs/docs/v$VERSION/

if [ -d $DIR ]; then
    echo "Directory $DIR exists."
	exit
fi

python3 ./custom_pdoc.py traceon -o $DIR  --force --html --config latex_math=True 

cp -r ./images $DIR/images/

echo "Done creating documentation for $VERSION"
