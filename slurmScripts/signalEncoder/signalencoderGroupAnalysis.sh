#!/bin/sh
start=$(date +%s)

# Pass the parameters to the Python script
srun accelerate launch ./../../metaTrainingControl.py \
    --numPosLiftedChannels "$1" \
    --numSigLiftedChannels "$2" \
    --numPosEncodingLayers "$3" \
    --numSigEncodingLayers "$4" \
    --numExpandedSignals "$5" \
    --deviceListed "HPC-$6" \
    --submodel "signalEncoder" \
    --optimizerType "$7"

end=$(date +%s)
runtime=$((end-start))
echo "Runtime: $runtime seconds"
