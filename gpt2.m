function mdl = gpt2()
%GPT2 Pretrained GPT-2 transformer model
%   mdl = gpt2 loads a pretrained GPT-2 transformer model and if necessary,
%   downloads the model weights.

mdl = struct;
mdl.Tokenizer = gpt2.tokenizer.GPT2Tokenizer('gpt2-355M', '.');
paramsStructFile = gpt2.internal.getSupportFilePath("gpt2_355M_params.mat");
mdl.Parameters = gpt2.load(paramsStructFile);
end
