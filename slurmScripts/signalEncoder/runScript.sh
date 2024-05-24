#!/bin/bash

numLiftedChannelsStart=64   # Absolute minimum is 16 or 32.
numLiftedChannelsStep=-16
numLiftedChannelsEnd=16     # Memory limited from 48-96.
# Total: 4

numExpandedSignalsStart=6   # Minimum 4; Maximum 6
numExpandedSignalsStep=-1
numExpandedSignalsEnd=2     # Minimum: 2
# Total: 5

numEncodingLayersStart=8    # Absolute minimum is 0.
numEncodingLayersStep=-1
numEncodingLayersEnd=0      # Memory limited from 10-12.
# Total: 9

for numLiftedChannels in $(seq $numLiftedChannelsStart $numLiftedChannelsStep $numLiftedChannelsEnd)
do
  for numExpandedSignals in $(seq $numExpandedSignalsStart $numExpandedSignalsStep $numExpandedSignalsEnd)
  do
      for numEncodingLayers in $(seq $numEncodingLayersStart $numEncodingLayersStep $numEncodingLayersEnd)
      do
          echo "Submitting job with $numLiftedChannels numLiftedChannels and $numExpandedSignals numExpandedSignals and $numEncodingLayers numEncodingLayers on $1

          if [ "$1" == "CPU" ]; then
              sbatch -J "signalEncoder_numLift_${numLiftedChannels}_numExp${numExpandedSignals}_numEnc${numEncodingLayers}_$1" submitSignalEncoder_CPU.sh "$numLiftedChannels" "$numEncodingLayers" "$numExpandedSignals" "$1"
          elif [ "$1" == "GPU" ]; then
              sbatch -J "signalEncoder_numLift_${numLiftedChannels}_numExp${numExpandedSignals}_numEnc${numEncodingLayers}_$1" submitSignalEncoder_GPU.sh "$numLiftedChannels" "$numEncodingLayers" "$numExpandedSignals" "$1"
          else
              echo "No known device listed: $1"
          fi
      done
  done
done
