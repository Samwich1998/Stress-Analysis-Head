#!/bin/sh
start=$(date +%s)

# Pass the compressionFactor to the Python script
srun accelerate launch ./../../metaTrainingControl.py --numLiftedChannels "$1" --numExpandedSignals "$2" --numEncodingLayers "$3" --deviceListed "HPC-$4" --submodel "signalEncoder" --optimizerType "$5"

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"