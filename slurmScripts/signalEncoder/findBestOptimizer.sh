#!/bin/bash

optimizers=('Adadelta' 'Adam' 'AdamW' 'NAdam' 'RAdam' 'Adamax' 'ASGD' 'RMSprop' 'Rprop' 'SGD')
numSigLiftedChannels=16
numSigEncodingLayers=4
numExpandedSignals=2
waveletType='bior3.7'

# Clean waveletType by removing dots
waveletTypeCleaned=$(echo "$waveletType" | tr -d '.')

for optimizer in "${optimizers[@]}"
do
    echo "Submitting job with $numSigLiftedChannels numSigLiftedChannels, $numSigEncodingLayers numSigEncodingLayers, $numExpandedSignals numExpandedSignals on $1 using $optimizer optimizer"

    if [ "$1" == "CPU" ]; then
        sbatch -J "signalEncoder_numSigLift_${numSigLiftedChannels}_numSigEnc_${numSigEncodingLayers}_numExp_${numExpandedSignals}_${waveletTypeCleaned}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSigLiftedChannels" "$numSigEncodingLayers" "$numExpandedSignals" "$1" "$waveletType" "$optimizer"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numSigLift_${numSigLiftedChannels}_numSigEnc_${numSigEncodingLayers}_numExp_${numExpandedSignals}_${waveletTypeCleaned}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSigLiftedChannels" "$numSigEncodingLayers" "$numExpandedSignals" "$1" "$waveletType" "$optimizer"
    else
        echo "No known device listed: $1"
    fi
done
