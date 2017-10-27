# Text classification

## CNN

```
bazel build nlp/text_classification/cnn_multiclass:run

train：bazel-bin/nlp/text_classification/cnn_multiclass/run \
    --train_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/cnn/ckpt \
    --data_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/data \
    --utils_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/cnn/utils \
    --num_epochs=10 \
    --process=train

        
predict：bazel-bin/nlp/text_classification/cnn_multiclass/run \
    --train_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/cnn/ckpt \
    --data_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/data \
    --utils_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/cnn/utils \
    --predict_file=/Users/endy/nlp/tensorflow_nlp/data/text-classification/cnn/raw.txt \
    --result_file=/Users/endy/nlp/tensorflow_nlp/data/text-classification/cnn/result.txt \
    --process=infer

```
## RNN

```
bazel build nlp/text_classification/rnn_multiclass:run

train：bazel-bin/nlp/text_classification/rnn_muticlass/run \
    --train_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/rnn/ckpt \
    --data_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/data \
    --utils_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/rnn/utils \
    --num_epochs=10 \
    --process=train
        
predict：bazel-bin/nlp/text_classification/rnn_muticlass/run \
    --train_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/rnn/ckpt \
    --data_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/data \
    --utils_dir=/Users/endy/nlp/tensorflow_nlp/data/text-classification/rnn/utils \
    --predict_file=/Users/endy/nlp/tensorflow_nlp/data/text-classification/rnn/raw.txt \
    --result_file=/Users/endy/nlp/tensorflow_nlp/data/text-classification/rnn/result.txt \
    --process=infer

```

## Reference

- [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
- [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)