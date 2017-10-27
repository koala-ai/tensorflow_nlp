# Pos tagging

Pos tagging is a classification problem of identifying the names of people,organisations,etc (different classes) in a text corpus.
Previous approaches to the problems have involved the usage of hand crafted language specific features, CRF and HMM based models, gazetteers, etc.
Growing interest in deep learning has led to application of deep neural networks to the existing problems like that of NER.
We have implemented a 2 network using tensorflow to classify the named entities.

## Use bilstm recognition

you can use the following to run:

```
bazel build nlp/pos/lstm:run

train：bazel-bin/nlp/pos/lstm/run \
    --train_dir=/Users/endy/nlp/tensorflow_nlp/data/pos/ckpt \
    --data_dir=/Users/endy/nlp/tensorflow_nlp/data/pos/data \
    --utils_dir=/Users/endy/nlp/tensorflow_nlp/data/pos/utils \
    --model=lstm \
    --max_epoch=10 \
    --process=train
        
predict：bazel-bin/nlp/pos/lstm/run \
    --train_dir=/Users/endy/nlp/tensorflow_nlp/data/pos/ckpt \
    --data_dir=/Users/endy/nlp/tensorflow_nlp/data/pos/data \
    --utils_dir=/Users/endy/nlp/tensorflow_nlp/data/pos/utils \
    --predict_file=/Users/endy/nlp/tensorflow_nlp/data/pos/data/predict.txt \
    --result_file=/Users/endy/nlp/tensorflow_nlp/data/pos/data/output.txt \
    --model=lstm \
    --process=infer
```

## Reference

- [Towards Deep Learning in Hindi NER: An approach to tackle the Labelled Data Scarcity](https://arxiv.org/abs/1610.09756)
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) 
- [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)
