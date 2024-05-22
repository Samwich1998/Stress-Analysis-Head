#!/bin/bash

optimizers=('Adadelta' 'Adam' 'AdamW' 'NAdam' 'RAdam' 'Adamax' 'ASGD' 'RMSprop' 'Rprop' 'SGD')
numSigLiftedChannels=32
numPosLiftedChannels=2
numPosEncodingLayers=2
numSigEncodingLayers=2
numExpandedSignals=2

for optimizer in "${optimizers[@]}"
do
    echo "Submitting job with $numSigLiftedChannels numSigLiftedChannels, $numPosLiftedChannels numPosLiftedChannels, $numPosEncodingLayers numPosEncodingLayers, $numSigEncodingLayers numSigEncodingLayers, $numExpandedSignals numExpandedSignals on $1 using $optimizer optimizer"

    if [ "$1" == "CPU" ]; then
        sbatch -J "signalEncoder_numPosLift_${numPosLiftedChannels}_numSigLift_${numSigLiftedChannels}_numPosEnc_${numPosEncodingLayers}_numSigEnc_${numSigEncodingLayers}_numExp_${numExpandedSignals}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numPosLiftedChannels" "$numSigLiftedChannels" "$numPosEncodingLayers" "$numSigEncodingLayers" "$numExpandedSignals" "$1" "$optimizer"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numPosLift_${numPosLiftedChannels}_numSigLift_${numSigLiftedChannels}_numPosEnc_${numPosEncodingLayers}_numSigEnc_${numSigEncodingLayers}_numExp_${numExpandedSignals}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numPosLiftedChannels" "$numSigLiftedChannels" "$numPosEncodingLayers" "$numSigEncodingLayers" "$numExpandedSignals" "$1" "$optimizer"
    else
        echo "No known device listed: $1"
    fi
done
