package.path = package.path ..';../model/batcher/?.lua'
package.path = package.path ..';../model/net/?.lua'
package.path = package.path ..';../model/model/?.lua'
package.path = package.path ..';../model/module/?.lua'
package.path = package.path ..';../model/optimizer/?.lua'
package.path = package.path ..';../model/criterion/?.lua'
require 'torch'
require 'nn'
require 'optim'
require 'rnn'
--Dependencies from this package
require 'MyOptimizer'
require 'OptimizerCallback'
require 'BatcherFileList'
require 'FeatureEmbedding'
require 'MapReduce'
require 'TopK'
require 'MyBCECriterion'
require 'Print'
require 'LogSumExp'
require "ConcatTableNoGrad"
require 'cunn'

local gpuid = 0
cutorch.setDevice(gpuid + 1) 

r_model_path = torch.load('/iesl/local/rajarshi/expts/multi_task/2016-05-16-10-57-19/combined_train_list/model-2')

local r_p, _ = r_model_path.embeddingLayer:parameters()

print(r_p[1][5])

t_model_path =  torch.load('/iesl/local/rajarshi/expts/multi_task/2016-05-16-10-57-19/combined_train_list/type_model-2')

local t_p, _ = t_model_path:parameters()
print(t_p[2][5])