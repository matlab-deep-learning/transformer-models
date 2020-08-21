function mdl = gpt2()
%GPT2 Pretrained GPT-2 transformer model
%   mdl = gpt2 loads a pretrained GPT-2 transformer model and if necessary,
%   downloads the model weights.

gpt2.download();
mdl = struct;
mdl.Tokenizer = gpt2.tokenizer.GPT2Tokenizer('gpt2-355M', '.');
mdl.Parameters = gpt2.load('gpt2-355M/parameters.mat');

end
