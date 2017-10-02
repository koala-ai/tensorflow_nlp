# Auto text summarization

In this implement we will be based on the Tensorflow Seq2Seq + Attention model, describes how to train a Chinese automatic generation of news headlines model.
The model of Automatic Summarization has been a hotspot. Direct extraction of important sentences of the extraction method is relatively simple, such as textrank,
and generate (re-generate a new sentence) is more complex, the effect is not satisfactory. At present, the more popular Seq2Seq model, proposed by Sutskever et al.,
Based on the structure of an Encoder-Decoder, the source sentence first Encode into a vector of fixed dimension d, and then generate a Target sentence by a character of the Decoder part.
After adding the Attention attention distribution mechanism, Decoder can generate the new Target Sequence to get the hidden information vector Hidden State of each character before the Encoder coding phase,
so that the accuracy of generating the new sequence is improved.

you can use the following to run:

```
cd nlp/textsum

训练：python run.py --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        ...
        --process=train \
        
预测：python run.py --train_dir=path/to/train_dir \ 
        --data_dir=path/to/data_dir \
        ...
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=infer \
```

## Reference

- [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://arxiv.org/pdf/1602.06023.pdf)