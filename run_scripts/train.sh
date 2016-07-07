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
echo "K" $K
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
dataOptions=$dataOptions" -useReLU $useReLU -rnnInitialization $rnnInitialization -learningRateDecay $learningRateDecay -regularize $regularize -numLayers $numLayers -useDropout $useDropout -dropout $dropout -K $K"

cmd="th ${MAIN_DIR}/model/OneModel.lua $dataOptions"
echo Executing:
echo $cmd
echo "Log file is $log"
CUDA_VISIBLE_DEVICES=$gpu_id  $cmd | tee $log



# while read predicate
# do
# 	echo "Running for predicate "${predicate}
# 	data_dir_p=${data_dir}/${predicate}
# 	echo ${data_dir_p}
# 	cmd="/bin/bash ${script_dir}/train_entity_type_models.sh ${data_dir_p} ${output_dir_t} ${gpu_id} $numEntityTypes $includeEntityTypes $includeEntity $numFeatureTemplates $relationEmbeddingDim $entityTypeEmbeddingDim $rnnHidSize $topK $numEpoch $learningRate $entityEmbeddingDim $rnnType $epsilon $gradClipNorm $gradientStepCounter $saveFrequency"
# 	echo $cmd
# 	$cmd
# done < $experiment_file

# wait

