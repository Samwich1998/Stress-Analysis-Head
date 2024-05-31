#!/bin/bash

waveletTypes=('cgau1' 'cgau2' 'cgau3' 'cgau4' 'cgau5' 'cgau6' 'cgau7' 'cgau8' 'smor' 'shan' 'gaus1' 'gaus2' 'gaus3' 'gaus4' 'gaus5' 'gaus6' 'gaus7' 'gaus8' 'morl' 'mexh' 'fbsp')
waveletTypes=('dmey' 'db31' 'db32' 'db33' 'db34' 'db35' 'db36' 'db37' 'db38' 'db24' 'db25' 'db26' 'db27' 'db28' 'db29' 'db30' 'coif15' 'coif16' 'coif17' 'coif8' 'coif9' 'coif10' 'coif11' 'coif12' 'coif13' 'coif14')

waveletTypes=( \
    'bior1.1' 'bior1.3' 'bior1.5' 'bior2.2' 'bior2.4' 'bior2.6' 'bior2.8' \
    'bior3.1' 'bior3.3' 'bior3.5' 'bior3.7' 'bior3.9' 'bior4.4' 'bior5.5' 'bior6.8' \
    'coif1' 'coif2' 'coif3' 'coif4' 'coif5' 'coif6' 'coif7' \
    'db1' 'db2' 'db3' 'db4' 'db5' 'db6' 'db7' 'db8' 'db9' 'db10' 'db11' 'db12' \
    'db13' 'db14' 'db15' 'db16' 'db17' 'db18' 'db19' 'db20' 'db21' 'db22' 'db23'  \
    'haar' 'rbio1.1' 'rbio1.3' 'rbio1.5' 'rbio2.2' 'rbio2.4' 'rbio2.6' 'rbio2.8' \
    'rbio3.1' 'rbio3.3' 'rbio3.5' 'rbio3.7' 'rbio3.9' 'rbio4.4' 'rbio5.5' 'rbio6.8' \
    'sym2' 'sym3' 'sym4' 'sym5' 'sym6' 'sym7' 'sym8' 'sym9' 'sym10' \
    'sym11' 'sym12' 'sym13' 'sym14' 'sym15' 'sym16' 'sym17' 'sym18' 'sym19' 'sym20' \
)
numSigLiftedChannels=32
numSigEncodingLayers=2
numExpandedSignals=4
optimizer='AdamW'

for waveletType in "${waveletTypes[@]}"
do
    echo "Submitting job with $numSigLiftedChannels numSigLiftedChannels $numSigEncodingLayers numSigEncodingLayers $numExpandedSignals numExpandedSignals on $1 using $waveletType waveletType and $optimizer optimizer."

    if [ "$1" == "CPU" ]; then
        sbatch -J "signalEncoder_numSigLift_${numSigLiftedChannels}_numSigEnc_${numSigEncodingLayers}_numExp_${numExpandedSignals}_${waveletType}_${optimizer}_$1" submitSignalEncoder_CPU.sh "$numSigLiftedChannels" "$numSigEncodingLayers" "$numExpandedSignals" "$1" "$waveletType" "$optimizer"
    elif [ "$1" == "GPU" ]; then
        sbatch -J "signalEncoder_numSigLift_${numSigLiftedChannels}_numSigEnc_${numSigEncodingLayers}_numExp_${numExpandedSignals}_${waveletType}_${optimizer}_$1" submitSignalEncoder_GPU.sh "$numSigLiftedChannels" "$numSigEncodingLayers" "$numExpandedSignals" "$1" "$waveletType" "$optimizer"
    else
        echo "No known device listed: $1"
    fi
done
