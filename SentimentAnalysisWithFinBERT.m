%% SentimentAnalysisWithFinBERT
% FinBERT is a sentiment analysis model based on the BERT-Base architecture.
% It has been pre-trained on financial text data, and fine-tuned for
% sentiment analysis.

mdl = finbert();
%% 
% Create some dummy inputs and encode them for FinBERT.

txt = [
    "In an unprecendented move the NASDAQ has hit new records today following news of a new vaccine."
    "The FTSE100 suffers dramatic losses on the back of the pandemic."
    "The ship unloader is totally enclosed along the entire conveying line to the storage facilities"
    "Marathon estimates the value of its remaining stake in Protalix at $ 27 million"
    "The company said that sales in the three months to the end of March slid to EUR86 .4 m US$ 113.4 m from EUR91 .2 m last year"
    "Ruukki Group calculates that it has lost EUR 4mn in the failed project"
    """Basware Corporation stock exchange release August 31 , 2010 at 16:25 Basware signed a large deal with an international industrial group Basware will deliver Invoice Automation solution and Connectivity Services to an international industrial group """
    "Operating profit rose to EUR 5mn from EUR 2.8 mn in the fourth quarter of 2008"
    ]
seq = mdl.Tokenizer.encode(txt);
%% 
% Pad sequence data.

x = padsequences(seq,2,'PaddingValue',mdl.Tokenizer.PaddingCode);
%% 
% Call the FinBERT model

[sentimentClass,sentimentScore] = finbert.sentimentModel(x,mdl.Parameters)