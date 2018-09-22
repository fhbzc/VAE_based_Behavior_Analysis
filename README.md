# VAE_based_Behavior_Analysis
Declara: The code comes from a open-source project of Google(original link here https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)
I modify it to apply it in agents behaviors analysis

How to generate trainable data:

  run GroupClassificationColor.py, it should generate 3 files: 
  train_dataset.npz
  valid_dataset.npz
  test_dataset.npz

How to train the model:

  run rnn_train.py, the trained variable should be stored in log folder in the same directory as rnn_train.py

How to get the classification image

  run rnn_fetch.py, it should generate a file called out_put_array.npy in ./result foler
  copy out_put_array.npy to ./T_SNE, in the same directory with the tsne.py
  run tsne.py, you should be able to see the classification result
