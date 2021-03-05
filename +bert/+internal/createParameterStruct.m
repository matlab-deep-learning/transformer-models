function weightsStruct = createParameterStruct(oldWeightsStruct)
% createParameterStruct   Given the flat struct of BERT model weights, this
% function parses that into a tree-like struct of weights.

% Copyright 2021 The MathWorks, Inc.

f = fieldnames(oldWeightsStruct);
for i = 1:numel(f)
    name = f{i};
    encoderLayerPrefix = "bert_encoder_layer";
    embeddingLayerPrefix = "bert_embeddings";
    poolingLayerPrefix = "bert_pooler_dense";
    langModPrefix = "cls_predictions";
    nspPrefix = "cls_seq_relationship_output";
    genericClassifierPrefix = "classifier_";
    
    weight = dlarray(oldWeightsStruct.(name));

    if startsWith(name,encoderLayerPrefix)
        % BERT transformer layer weights.
        layerIndex = extractBetween(name,encoderLayerPrefix+"_","_");
        newLayerIndex = str2double(layerIndex)+1;
        layerName = encoderLayerPrefix+"_"+layerIndex;
        shortLayerName = "layer_"+newLayerIndex;
        paramName = extractAfter(name,layerName+"_");
        attentionOrFeedforward = iParseAttentionOrFeedforward(paramName);
        [subParamName,subsubParamName] = iParseAttentionAndFeedforwardParamName(paramName,attentionOrFeedforward);
        weightsStruct.("encoder_layers").(shortLayerName).(attentionOrFeedforward).(subParamName).(subsubParamName) = weight;
        
    elseif startsWith(name,embeddingLayerPrefix)
        % Emebdding parameters
        paramName = extractAfter(name,embeddingLayerPrefix+"_");
        if contains(paramName,"LayerNorm")
            [subname,subsubname] = iParseLayerNorm(paramName);
            weightsStruct.("embeddings").(subname).(subsubname) = weight;
        else
            weightsStruct.("embeddings").(paramName) = weight;
        end
        
    elseif startsWith(name,poolingLayerPrefix)
        paramName = extractAfter(name,poolingLayerPrefix+"_");
        weightsStruct.("pooler").(paramName) = weight;
        
    elseif startsWith(name,langModPrefix)
        paramName = extractAfter(name,langModPrefix+"_");
        [subname,subsubname] = iParseLM(paramName);
        weightsStruct.("masked_LM").(subname).(subsubname) = weight;
        
    elseif startsWith(name,nspPrefix)
        paramName = extractAfter(name,nspPrefix+"_");
        if strcmp(paramName,"weights")
            % This parameter wasn't renamed and transposed before
            % uploading. We can fix it here.
            paramName = "kernel";
            weight = weight.';
        end
        weightsStruct.("sequence_relation").(paramName) = weight;
        
    elseif startsWith(name,genericClassifierPrefix)
        paramName = extractAfter(name,genericClassifierPrefix);
        weightsStruct.("classifier").(paramName) = weight;
    end
end
end

function name = iParseAttentionOrFeedforward(name)
if contains(name, "attention")
    name = "attention";
else
    name = "feedforward";
end
end

function [name,subname] = iParseAttentionAndFeedforwardParamName(name,attnOrFeedforward)
switch attnOrFeedforward
    case "attention"
        [name,subname] = iParseAttentionParamName(name);
    case "feedforward"
        [name,subname] = iParseFeedforwardParamName(name);
end
end

function [subname,subsubname] = iParseAttentionParamName(name)
if contains(name,"LayerNorm")
    [subname,subsubname] = iParseLayerNorm(name);
else
    name = strrep(name,"self_","");
    name = strrep(name,"_dense","");
    subname = extractBetween(name,"attention_","_");
    subsubname = extractAfter(name,subname+"_");
end
end

function [subname,subsubname] = iParseFeedforwardParamName(name)
if contains(name,"LayerNorm")
    [subname,subsubname] = iParseLayerNorm(name);
else
    subname = extractBefore(name,"_");
    subsubname = extractAfter(name,"dense_");
end
end

function [subname,subsubname] = iParseLayerNorm(name)
subname = "LayerNorm";
subsubname = extractAfter(name,"LayerNorm_");
end

function [subname,subsubname] = iParseLM(name)
if contains(name,"LayerNorm")
    [subname,subsubname] = iParseLayerNorm(name);
else
    name = strrep(name,"dense_","");
    subname = extractBefore(name,"_");
    subsubname = extractAfter(name,"_");
end    
end