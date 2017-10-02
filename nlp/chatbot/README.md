# Chatbot

*step1:*

Enter: first download a copy of dgk_shooter_min.conv.zip from [dgk_lost_conv](https://github.com/majoressense/dgk_lost_conv)

Output: and then extract the dgk_shooter_min.conv file

*step2:*

Execute the decode_conv.py script

Enter: python decode_conv.py

Output: will generate a sqlite3 format database file in db / conversation.db

*step3:*

Execute the data_utils.py script

Enter: python data_utils.py

Output: will generate a bucket_dbs directory, which contains a number of sqlite3 format database, which is the size of the data divided into different buckets inside

For example, the length of the question ask is less than or equal to 5, and the output answer answer length is less than 15, it will be put into bucket_5_15_db inside

*step4:train*

```
python run.py \
--size 1024 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 64 \
--num_per_epoch 500000 \
--model_dir ./model/model1
```

*step5:test*

```
python run.py \
--size 1024 \
--num_layers 2 \
--num_epoch 5 \
--batch_size 64 \
--num_per_epoch 500000 \
--model_dir ./model/model1 \
--test true
```

## Reference

- [使用TensorFlow实现的Sequence to Sequence的聊天机器人模型](https://github.com/qhduan/Seq2Seq_Chatbot_QA)