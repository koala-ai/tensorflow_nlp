# Auto text summarization

In this implement we will be based on the Tensorflow `Seq2Seq + Attention` model, describes how to train a Chinese automatic generation of news headlines model.
The model of Automatic Summarization has been a hotspot. Direct extraction of important sentences of the extraction method is relatively simple, such as textrank,
and generate (re-generate a new sentence) is more complex, the effect is not satisfactory. At present, the more popular Seq2Seq model, proposed by Sutskever et al.,
Based on the structure of an `Encoder-Decoder`, the source sentence first encode into a vector of fixed dimension d, and then generate a Target sentence by a character of the Decoder part.
After adding the Attention mechanism, `Decoder` can generate the new target sequence to get the hidden information vector hidden state of each character before the `Encoder` coding phase,
so that the accuracy of generating the new sequence is improved.

you can use the following to run:

```
bazel build nlp/textsum:run

train：bazel-bin/nlp/textsum/run \
    --train_dir=/Users/endy/nlp/tensorflow_nlp/data/textsum/ckpt \
    --data_dir=/Users/endy/nlp/tensorflow_nlp/data/textsum/data \
    --utils_dir=/Users/endy/nlp/tensorflow_nlp/data/textsum/utils \
    --num_steps=2 \
    --process=train
        
predict：python run.py --train_dir=path/to/train_dir \ 
        --data_dir=path/to/data_dir \
        ...
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=infer \
```

## Reference

- [Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond](https://arxiv.org/pdf/1602.06023.pdf)
