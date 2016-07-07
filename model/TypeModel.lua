package.path = package.path ..';../model/batcher/?.lua'
package.path = package.path ..';../model/net/?.lua'
package.path = package.path ..';../model/model/?.lua'
package.path = package.path ..';../model/module/?.lua'
package.path = package.path ..';../model/optimizer/?.lua'
package.path = package.path ..';../model/criterion/?.lua'
require 'torch'
require 'nn'
require 'optim'
require 'TypeBatcher'
require 'TypeNetwork'
require 'TypeOptimizer'
require 'BPRLoss'
require "ConcatTableNoGrad"
require "rnn"

local gpuid = 0
print('USING GPU')
require 'cutorch'
require('cunn')
cutorch.setDevice(gpuid + 1) 


cmd = torch.CmdLine()
cmd:option('-input_file','','input_file')
cmd:option('-batch_size',0,'batch_size')
cmd:option('-type_dim',0,'dim')
cmd:option('-entity_dim',0,'dim')
cmd:option('-typeRegularize',0,'typeRegularize')
cmd:option('-typeGradClip',0,'typeGradClip')
cmd:option('-typeGradClipNorm',0,'typeGradClipNorm')
cmd:option('-typeL2',0,'typeL2')
cmd:option('-numEpochs',0,'numEpochs')
cmd:option('-learningRate',0,'learningRate')
cmd:option('-expt_dir','','expt_dir')

--pretraining options
cmd:option('-preTraining',0,'preTraining')
cmd:option('-preTrainedRelationModelPath','','preTrainedRelationModelPath')

local params = cmd:parse(arg)

local useCuda = true
local genNeg = true -- generate gen negative samples
local typeVocab = 2218
local entityVocab = 1542690
local type_dim = params.type_dim
local entity_dim = params.entity_dim
local batchSize = params.batch_size
local shuffle = true
local input_file = params.input_file
local typeRegularize = params.typeRegularize
local typeGradClip = params.typeGradClip
local typeGradClipNorm = params.typeGradClipNorm
local typeL2 = params.typeL2
local numEpochs = params.numEpochs
local expt_dir=params.expt_dir
local learningRate = params.learningRate
local preTraining = params.preTraining
local preTrainedRelationModelPath = params.preTrainedRelationModelPath

local saveFileName = expt_dir..'/type_model'

local col_encoder = nn.LookupTable(typeVocab, type_dim)
local row_encoder = nn.LookupTable(entityVocab, entity_dim)
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

--pretraining

if preTraining == 1 then
    print('Pretraining with paths model!!!')
    local preTrainedModel = torch.load(preTrainedRelationModelPath)
    -- preTrainedModel = preTrainedModel.predictor_net
    local preTrainedParams, preTrainedGradParams = preTrainedModel:parameters()
    local preTrainedEntityTypeLookupTable = preTrainedParams[1]
    assert(preTrainedEntityTypeLookupTable:size(1) == typeVocab)
    --pretrain the type embeddings
    local typeLookupTable = params[2]
    typeLookupTable:copy(preTrainedEntityTypeLookupTable)
end


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
    saveFileName = saveFileName
}

--call optimization class
local optimizer = TypeOptimizer(type_net, criterion, typeOptInfo)
optimizer:train(typeBatcher)