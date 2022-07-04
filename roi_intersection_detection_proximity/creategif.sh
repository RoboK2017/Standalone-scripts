#!/bin/bash

# Check args
if [ "$#" -ne 1 ]; then
  echo "usage: ./creategif ROI_OUTPUT_FOLDER"
fi

for f1 in $1/*; do

    if [ ! -d $f1/gif ] 
        then
            mkdir $f1/gif
        fi

    if [ ! -d $f1/mp4 ] 
        then
            mkdir $f1/mp4 
        fi
    
    for f in $f1/*.mp4; do
        fn="$f1/gif/${f##*/}"
        ffmpeg -i "$f" -vf scale=720:-1 "${fn%.*}.gif"
        mv $f  $f1/mp4/"${f##*/}"
    done

done
