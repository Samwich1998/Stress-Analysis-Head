#!/bin/bash

optimizers=('Adadelta' 'Adam' 'AdamW' 'NAdam' 'RAdam' 'Adamax' 'ASGD' 'RMSprop' 'Rprop' 'SGD')
numLiftedChannels=32
numExpandedSignals=2
numEncodingLayers=2


for optimizer in "${optimizers[@]}"
do
    echo "Submitting job with $numLiftedChannels numLiftedChannels, $numExpandedSignals numExpandedSignals, $numEncodingLayers numEncodingLayers on $1 using $optimizer optimizer"

    if [ "$1" == "CPU" ]; then
        sbatch -J "signalEncoder_numLift_${numLiftedChannels}_numExp${numExpandedSignals}_numEnc${numEncodingLayers}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numLiftedChannels" "$numExpandedSignals" "$numEncodingLayers" "$1" "$optimizer"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numLift_${numLiftedChannels}_numExp${numExpandedSignals}_numEnc${numEncodingLayers}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numLiftedChannels" "$numExpandedSignals" "$numEncodingLayers" "$1" "$optimizer"
    else
        echo "No known device listed: $1"
    fi
done
