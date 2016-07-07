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
	data_dir_types="/iesl/canvas/rajarshi"
fi
experiment_file=$experiment_dir/0.txt
output_dir=$output_dir/multi_task
data_dir=$data_dir/data_full_max_length_8/combined_train_list
data_dir_types=$data_dir_types/emnlp_entity_types_data/train/
gpu_id=2
numEntityTypes=7
includeEntityTypes=1
includeEntity=0
numEpoch=20
typeNumEpoch=50
numFeatureTemplates=10
rnnHidSize=250
relationEmbeddingDim=250
entityTypeEmbeddingDim=50
entityEmbeddingDim=50
relationVocabSize=51390
entityVocabSize=1542690
entityTypeVocabSize=2218
topK=0
regularize=0
typeRegularize=1
learningRate=1e-4
typeLearningRate=1e-4
learningRateDecay=0.0167 #(1/60)
l2=1e-3
type_l2=1e-8
rnnType='rnn' #rnn or lstm as of now
epsilon=1e-8 #epsilon for adam
gradClipNorm=5
typeGradClip=0
typeGradClipNorm=5
gradientStepCounter=100000 #to print loss after gradient updates
saveFrequency=1
batchSize=32
typeBatchSize=1024
useGradClip=1 # 0 == L2 regularization
package_path='/home/rajarshi/EMNLP/LSTM-KBC/model/?.lua'
useAdam=1
paramInit=0.1
evaluationFrequency=5
createExptDir=1 #make it 0 if you dont want to create a directory and only print stuff
useReLU=1
rnnInitialization=1
numLayers=1
useDropout=0
dropout=0.3