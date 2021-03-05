# Transformer Models for MATLAB
This repository implements deep learning transformer models in MATLAB.

## Requirements
* MATLAB R2020a or later for GPT-2
* MATLAB R2021a or later for BERT and FinBERT
* Deep Learning Toolbox
* Text Analytics Toolbox for BERT and FinBERT

## Getting Started
Download or [clone](https://www.mathworks.com/help/matlab/matlab_prog/use-source-control-with-projects.html#mw_4cc18625-9e78-4586-9cc4-66e191ae1c2c) this repository to your machine and open it in MATLAB.

## Functions
### [bert](./bert.m)
`mdl = bert` loads a pretrained BERT transformer model and if necessary, downloads the model weights. The `mdl` struct has fields `Tokenizer` containing the BERT tokenizer and `Parameters` to be passed to `bert.model(x,mdl.Parameters)` where `x` can be `seq{1}` where `seq = mdl.Tokenizer.encode("hello world!");`

`mdl = bert('Model',modelName)` specifies an optional model. All models besides `"multilingual-cased"` are case-insensitve. The choices for `modelName` are:
* `"base"` (default) - A 12 layer model with hidden size 768.
* `"multilingual-cased"` - A 12 layer model with hidden size 768. The tokenizer is case-sensitive. This model was trained on multi-lingual data.
* `"medium"` - An 8 layer model with hidden size 512. 
* `"small"` - A 4 layer model with hidden size 512.
* `"mini"` - A 4 layer model with hidden size 256.
* `"tiny"` - A 2 layer model with hidden size 128.

The model parameters match those found on the [original BERT repo](https://github.com/google-research/bert/). The BERT-Base parameters are from the original release, not the update from where the smaller models are sourced.

### [gpt2](./gpt2.m)
`mdl = gpt2` loads a pretrained GPT-2 transformer model and if necessary, downloads the model weights.

### [generateSummary](./generateSummary.m)
`summary = generateSummary(mdl,text)` generates a summary of the string or `char` array `text` using the transformer model `mdl`. The output summary is a char array.

`summary = generateSummary(mdl,text,Name,Value)` specifies additional options using one or more name-value pairs.

* `'MaxSummaryLength'` - The maximum number of tokens in the generated summary. The default is 50.
* `'TopK'` - The number of tokens to sample from when generating the summary. The default is 2.
* `'Temperature'` - Temperature applied to the GPT-2 output probability distribution. The default is 1.
* `'StopCharacter'` - Character to indicate that the summary is complete. The default is `'.'`.

### [finbert](./finbert.m)
`mdl = finbert` loads a pretrained and fine-tuned BERT transformer model for classifying sentiment of financial text. The `mdl` struct is similar to the BERT model struct with additional weights for the sentiment classifier head. The sentiment analysis functionaliy is accessed through `[sentimentClass,sentimentScore] = finbert.sentimentModel(x,mdl.Parameters)` where `seq = mdl.Tokenizer.encode("The FTSE100 suffers dramatic losses on the back of the pandemic."); x = dlarray(seq{1});`.

`mdl = finbert('Model',modelName)` specifies an optional model from the choices:
* `"sentiment-model"` - The fine-tuned sentiment classifier model.
* `"language-model"` - The FinBERT pre-trained language model, which uses a BERT-Base architecture.

The parameters match those found on the [original FinBERT repo](https://github.com/ProsusAI/finBERT).

## Example: Classify Text Data Using BERT
The example [`ClassifyTextDataUsingBERT.m`](./ClassifyTextDataUsingBERT.m) is [this existing example](https://www.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html) reinterpreted to use BERT as an embedding.

## Example: Summarize Text Using GPT-2
The example [`SummarizeTextUsingTransformersExample.m`](./SummarizeTextUsingTransformersExample.m) shows how to summarize a piece of text using GPT-2.

Transformer networks such as GPT-2 can be used to summarize a piece of text. The trained GPT-2 transformer can generate text given an initial sequence of words as input. The model was trained on comments left on various web pages and internet forums.

Because lots of these comments themselves contain a summary indicated by the statement "TL;DR" (Too long, didn't read), you can use the transformer model to generate a summary by appending "TL;DR" to the input text. The `generateSummary` function takes the input text, automatically appends the string `"TL;DR"` and generates the summary.

### Load Transformer Model
Load the GPT-2 transformer model using the [`gpt2`](./gpt2.m) function.

```matlab:Code
mdl = gpt2;
```

### Load Data
Extract the help text for the `eigs` function.

```matlab:Code
inputText = help('eigs');
```

### Generate Summary
Summarize the text using the [`generateSummary`](./generateSummary.m) function.

```matlab:Code
rng('default')
summary = generateSummary(mdl,inputText)
```

```text:Output
summary =

    '    EIGS(AFUN,N,FLAG) returns a vector of AFUN's n smallest magnitude eigenvalues'
```

## Example: Classify Sentiment with FinBERT
The example [`SentimentAnalysisWithFinBERT.m`](./SentimentAnalysisWithFinBERT.m) uses the FinBERT sentiment analysis model to classify sentiment for a handful of example financial sentences.

## Example: Masked Token Prediction with BERT and FinBERT
The examples [`LanguageModelingWithBERT.m`](./LanguageModelingWithBERT.m) and [`LanguageModelingWithFinBERT.m`](./LanguageModelingWithFinBERT.m) demonstrate the language models predicting masked tokens.