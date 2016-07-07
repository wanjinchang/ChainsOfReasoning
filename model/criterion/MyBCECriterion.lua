local MyBCECriterion, parent = torch.class('nn.MyBCECriterion', 'nn.BCECriterion')

local eps = 1e-12

function MyBCECriterion:__init(classId, weights, sizeAverage)
    parent.__init(self)
    self.classId = classId -- one of the L labels
end


function MyBCECriterion:updateOutput(input, target)
    assert(target:dim() == 1 or (target:dim() == 2 and target:size(2) == 1)) -- target should be batch_size X 1 or batch_size
    assert(input:dim() == 2 and input:size(2) >= self.classId) --input should be a matrix of batch_sz=ize X num_labels and hence num_labels should be >= classId
    --now narrow the tensor
    input = input:narrow(2, self.classId, 1)
    self.output = parent:updateOutput(input, target)

    return self.output
end

function MyBCECriterion:updateGradInput(input, target)

    assert(target:dim() == 1 or (target:dim() == 2 and target:size(2) == 1)) -- target should be batch_size X 1 or batch_size
    assert(input:dim() == 2 and input:size(2) >= self.classId) --input should be a matrix of batch_sz=ize X num_labels and hence num_labels should be >= classId
    local origInput = input
    --now narrow the tensor
    input = input:narrow(2, self.classId, 1)
    local gradInput = parent:updateGradInput(input, target)
    self.gradInput = torch.Tensor():typeAs(origInput):resizeAs(origInput):fill(0)
    self.gradInput:narrow(2, self.classId, 1):copy(gradInput)
    return self.gradInput
    
end