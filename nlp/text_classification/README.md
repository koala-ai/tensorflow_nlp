# Text classification

## RNN

```
bazel build nlp/text_classification/rnn_multiclass:run

train：bazel-bin/nlp/text_classification/rnn_multiclass/run \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --model=lstm or gru \
        ...
        --process=train
        
predict：bazel-bin/nlp/text_classification/rnn_multiclass/run \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        ...
        --process=infer
```
## CNN

```
bazel build nlp/text_classification/cnn_multiclass:run

train：bazel-bin/nlp/text_classification/cnn_multiclass/run \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --process=train
        
predict：bazel-bin/nlp/text_classification/cnn_multiclass/run \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=predict
```

## Reference

- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)