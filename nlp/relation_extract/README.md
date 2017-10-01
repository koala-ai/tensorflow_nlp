# Relation extraction

Relationship extraction can be as simple as a classification problem: given two entities and the sentence of text include two entities, identifying the relationship between the two entities.
Using CNN or BI-RNN with Attention of deep learning approach is considered now the state of art of the solution.

## BI-GRU with attention

BI-GRU with character attention come from reference 1

![extract1](../../images/relation_extract1.jpg)

sentence attention come from reference 2

![extract2](../../images/relation_extract2.jpg)

you can use the following to run:

```
cd deepnlp/relation_extract

训练：python run.py \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --process=train
        
预测：python run.py \
        --train_dir=path/to/train_dir \
        --data_dir=path/to/data_dir \
        --predict_file=path/to/predict_file \
        --output_file=path/to/output_file \
        --process=infer
```

# Reference

- [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://anthology.aclweb.org/P16-2034)
- [Neural Relation Extraction with Selective Attention over Instances](http://aclweb.org/anthology/P16-1200)