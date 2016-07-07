package.path = package.path ..';../model/batcher/?.lua'
package.path = package.path ..';../model/net/?.lua'
package.path = package.path ..';../model/model/?.lua'
package.path = package.path ..';../model/module/?.lua'
package.path = package.path ..';../model/optimizer/?.lua'
package.path = package.path ..';../model/criterion/?.lua'
require 'torch'
require 'nn'
require 'rnn'
require 'optim'
require 'TypeBatcher'
require 'TypeNetwork'
require 'TypeOptimizer'
require 'BPRLoss'
require "ConcatTableNoGrad"

local gpuid = 0
print('USING GPU')
require 'cutorch'
require('cunn')
cutorch.setDevice(gpuid + 1) 


local useCuda = true
local genNeg = true -- generate gen negative samples
local typeVocab = 2218
local entityVocab = 1542690
local dim = 50
local batchSize = 1024
local shuffle = true
local input_file = '/iesl/canvas/rajarshi/emnlp_entity_types_data/train/train.torch'
local typeRegularize = 1
local typeGradClip= false
local typeGradClipNorm = 0
local typeL2 = 1e-8
local numEpochs = 50


local preTrainedRelationModelPath = '/iesl/local/rajarshi/expts/one_model/with_types/model-15'
local preTrainedModel = torch.load(preTrainedRelationModelPath)
preTrainedModel = preTrainedModel.predictor_net
local preTrainedParams, preTrainedGradParams = preTrainedModel:parameters()
local preTrainedEntityTypeLookupTable = preTrainedParams[1]
assert(preTrainedEntityTypeLookupTable:size(1) == typeVocab)

local col_encoder = nn.LookupTable(typeVocab, dim)
local row_encoder = nn.LookupTable(entityVocab, dim)
local typeBatcher = TypeBatcher(input_file, batchSize, shuffle, genNeg, entityVocab)
local t_net = TypeNetwork(useCuda)

--get the network
local type_net = t_net:build_network(row_encoder, col_encoder)
print(type_net)

--initialize
local params, gradParams = type_net:parameters()
for k, param in ipairs(params) do
    param:uniform(-0.1, 0.1)
end
--pretrain the type embeddings
local typeLookupTable = params[2]
typeLookupTable:copy(preTrainedEntityTypeLookupTable)
--criterion
local criterion = nn.BPRLoss()

--cuda them
if useCuda then
    type_net:cuda()
    criterion:cuda()
end

--optimizer
local optimMethod = optim.adam
local beta1 = 0.9
local beta2 = 0.999
local epsilon = 1e-8
local learningRate = 1e-3
local optConfig = {learningRate = learningRate,beta1 = beta1,beta2 = beta2,epsilon = epsilon}

typeOptInfo = {
    optimMethod = optimMethod,
    optConfig = optConfig,
    optState = {},  
    regularization = typeRegularize,
    useCuda = useCuda,
    typeRegularize = typeRegularize,
    typeGradClip = typeGradClip,
    typeGradClipNorm = typeGradClipNorm,
    typeL2 = typeL2,
    numEpochs = numEpochs,
    saveFileName = '/iesl/local/rajarshi/expts/pre_trained_type_prediction/model'
}

--call optimization class
local optimizer = TypeOptimizer(type_net, criterion, typeOptInfo)
optimizer:train(typeBatcher)