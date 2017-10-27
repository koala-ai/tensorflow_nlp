# Poems

## Introduce

Chinese classical poetry is undoubtedly the largest and most bright pearl, if Chinese classical literature is compared to a crown.
As a kind of literary form starting from the Pre-Qin Period, classical poetry stretches more than two thousand years, having a farreaching
influence on the development of Chinese history. Poets write poems to record important events, express feelings and make comments. 
This task use rnn to generate poems.

we can use this implement to generate poem:

```
bazel build nlp/poems:run

train：bazel-bin/nlp/poems/run \
    --train_dir=/Users/endy/nlp/tensorflow_nlp/data/poems/ckpt \
    --data_dir=/Users/endy/nlp/tensorflow_nlp/data/poems/data \
    --method=lstm \
    --epochs=10 \
    --process=train
        
generate：bazel-bin/nlp/poems/run \
		--data_dir=/Users/endy/nlp/tensorflow_nlp/data/poems/data \
        --train_dir=/Users/endy/nlp/tensorflow_nlp/data/poems/ckpt \
        --method= lstm \
        --begin_word= begin word of poem \
        --process=generate
```

you can check run.py file to know special information.

## Reference

[Generating Chinese Classical Poems with RNN Encoder-Decoder](https://arxiv.org/pdf/1604.01537.pdf)