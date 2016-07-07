#!/bin/bash
host=`hostname`
if [[ $host == ip* ]]; then
	experiment_dir="/home/ubuntu/LSTM-KBC-New/"
	output_dir="/home/ubuntu/expts/"	
	data_dir="/home/ubuntu/"
else
	experiment_dir="/home/rajarshi/LSTM-KBC-New/"		
	output_dir="/iesl/local/rajarshi/expts/"
	data_dir="/iesl/local/rajarshi"
fi

time_stamp=`date +"%Y-%m-%d-%H-%M-%S"`
gpu_id=0
output_dir=$output_dir/entity_type_baseline/$time_stamp
mkdir -p $output_dir
log=$output_dir/log.txt
config_file=$output_dir/config.txt
input_file=$data_dir'/emnlp_entity_types_data/train/train.torch'
batch_size=32
type_dim=50
entity_dim=50
typeRegularize=1
typeGradClip=1
typeGradClipNorm=5
typeL2=1e-8
numEpochs=15
learningRate=1e-4
preTraining=1
preTrainedRelationModelPath='/home/ubuntu/expts/pretrained_paths_model/model-15_float'

echo "batch_size" $batch_size | tee -a $config_file
echo "type_dim" $type_dim | tee -a $config_file
echo "entity_dim" $entity_dim | tee -a $config_file
echo "typeRegularize" $typeRegularize | tee -a $config_file
echo "typeGradClip" $typeGradClip | tee -a $config_file
echo "typeGradClipNorm" $typeGradClipNorm | tee -a $config_file
echo "typeL2" $typeL2 | tee -a $config_file
echo "learningRate" $learningRate | tee -a $config_file
echo "numEpochs" $numEpochs | tee -a $config_file
echo "gpu_id" $gpu_id | tee -a $config_file
echo "preTraining" $preTraining | tee -a $config_file
echo "preTrainedRelationModelPath" $preTrainedRelationModelPath | tee -a $config_file


cmd="th $experiment_dir/model/TypeModel.lua -input_file $input_file -batch_size $batch_size -type_dim $type_dim -entity_dim $entity_dim -typeRegularize $typeRegularize -typeGradClip $typeGradClip -typeGradClipNorm $typeGradClipNorm -typeL2 $typeL2 -numEpochs $numEpochs -expt_dir $output_dir -learningRate $learningRate -preTraining $preTraining -preTrainedRelationModelPath $preTrainedRelationModelPath"
echo $cmd

CUDA_VISIBLE_DEVICES=$gpu_id  $cmd | tee  $log