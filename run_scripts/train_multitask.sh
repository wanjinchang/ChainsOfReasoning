#!/bin/sh
time_stamp=`date +"%Y-%m-%d-%H-%M-%S"`
host=`hostname`
if [[ $host == ip* ]]; then
	MAIN_DIR="/home/ubuntu/LSTM-KBC-New"
	script_dir="$MAIN_DIR/run_scripts"
	output_dir="/home/ubuntu/expts/"	
	data_dir="/home/ubuntu/"
else
	MAIN_DIR="/home/rajarshi/LSTM-KBC-New"
	script_dir="$MAIN_DIR/run_scripts"
	output_dir="/iesl/canvas/rajarshi/expts/"	
	data_dir="/home/rajarshi/canvas/"
fi

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 config_file"
	exit 1
fi

config_file=$1
source $config_file
echo "experiment_file "$experiment_file
echo "output_dir "$output_dir
echo "data_dir "$data_dir
echo "gpu_id "$gpu_id
echo "numEpoch" $numEpoch
echo "numEntityTypes "$numEntityTypes
echo "includeEntityTypes "$includeEntityTypes
echo "includeEntity "$includeEntity
echo "numFeatureTemplates "$numFeatureTemplates
echo " relationEmbeddingDim "$relationEmbeddingDim
echo "entityTypeEmbeddingDim " $entityTypeEmbeddingDim
echo "entityEmbeddingDim " $entityEmbeddingDim
echo "rnnHidSize" $rnnHidSize
echo "topK" $topK
echo "Learning Rate" $learningRate
echo "Learning Rate Decay" $learningRateDecay
echo "rnnType " $rnnType
echo "epsilon" $epsilon
echo "gradClipNorm" $gradClipNorm
echo "gradientStepCounter" $gradientStepCounter
echo "saveFrequency" $saveFrequency
echo "batchSize "$batchSize
echo "useGradClip" $useGradClip
echo "package_path" $package_path 
echo "useAdam" $useAdam
echo "paramInit" $paramInit
echo "evaluationFrequency" $evaluationFrequency
echo "createExptDir" $createExptDir
echo "useReLU" $useReLU
echo "l2" $l2
echo "rnnInitialization" $rnnInitialization
echo "regularize "$regularize
echo "numLayers "$numLayers
echo "useDropout" $useDropout
echo "relationVocabSize" $relationVocabSize
echo "entityVocabSize" $entityVocabSize
echo "entityTypeVocabSize" $entityTypeVocabSize
echo "dropout" $dropout
echo "typeBatchSize" $typeBatchSize
echo "type_l2" $type_l2
echo "data_dir_types" $data_dir_types
echo "typeNumEpoch" $typeNumEpoch
echo "typeRegularize" $typeRegularize
echo "typeGradClip" $typeGradClip
echo "typeGradClipNorm" $typeGradClipNorm
echo "typeLearningRate" $typeLearningRate

machine_name=`hostname`
predicate_name=`basename ${data_dir}`

tokFeats=0

output_dir_t=${output_dir}/${time_stamp}
exptDir=${output_dir_t}/${predicate_name}
log=$exptDir/log.txt #where everything will be logged
modelBase=$exptDir/model

if [ $createExptDir -eq 1 ]; then
	mkdir -p ${output_dir_t}
	mkdir -p $exptDir
	#create symlink, combine with machine name
	rm ${output_dir}/LATEST_${machine_name}
	ln -s ${output_dir_t} ${output_dir}/LATEST_${machine_name}
fi


dataOptions="-trainList $data_dir/train.list -testList $data_dir/dev.list -tokenFeatures $tokFeats -minibatch $batchSize -gpuid 0 -learningRate $learningRate -l2 $l2 -numEpochs $numEpoch -useAdam $useAdam"
dataOptions=$dataOptions" -saveFrequency $saveFrequency -evaluationFrequency $evaluationFrequency -model $modelBase -rnnType $rnnType -exptDir $exptDir -relationVocabSize $relationVocabSize -entityTypeVocabSize $entityTypeVocabSize -relationEmbeddingDim $relationEmbeddingDim -entityTypeEmbeddingDim $entityTypeEmbeddingDim -numFeatureTemplates $numFeatureTemplates"
dataOptions=$dataOptions" -numEntityTypes $numEntityTypes -includeEntityTypes $includeEntityTypes -includeEntity $includeEntity -entityVocabSize $entityVocabSize -entityEmbeddingDim $entityEmbeddingDim -rnnHidSize $rnnHidSize -topK $topK -epsilon $epsilon -gradClipNorm $gradClipNorm -gradientStepCounter $gradientStepCounter -useGradClip $useGradClip -package_path $package_path -paramInit $paramInit -createExptDir $createExptDir"
dataOptions=$dataOptions" -useReLU $useReLU -rnnInitialization $rnnInitialization -learningRateDecay $learningRateDecay -regularize $regularize -numLayers $numLayers -useDropout $useDropout -dropout $dropout"
dataOptions=$dataOptions" -typeBatchSize $typeBatchSize -type_l2 $type_l2 -data_dir_types $data_dir_types -typeNumEpoch $typeNumEpoch -typeRegularize $typeRegularize -typeGradClip $typeGradClip -typeGradClipNorm $typeGradClipNorm -typeLearningRate $typeLearningRate"

cmd="th ${MAIN_DIR}/model/MultiTaskTraining.lua $dataOptions"
echo Executing:
echo $cmd
echo "Log file is $log"
CUDA_VISIBLE_DEVICES=$gpu_id  $cmd | tee $log