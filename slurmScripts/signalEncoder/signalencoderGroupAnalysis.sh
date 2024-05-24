#!/bin/sh
start=$(date +%s)

# Pass the parameters to the Python script
srun accelerate launch ./../../metaTrainingControl.py \
    --numSigLiftedChannels "$1" \
    --numSigEncodingLayers "$2" \
    --numExpandedSignals "$3" \
    --deviceListed "HPC-$4" \
    --submodel "signalEncoder" \
    --optimizerType "$5"

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"
