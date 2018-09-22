# VAE_based_Behavior_Analysis


Declaration: The code comes from a open-source project of Google(original link here https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)
I modify and apply it in agents behaviors analysis


How to generate trainable data?
-------

run GroupClassificationColor.py, it should generate 3 files: 
train_dataset.npz
valid_dataset.npz
test_dataset.npz

How to train the model?
---------
run rnn_train.py, the trained variable should be stored in log folder in the same directory as rnn_train.py

How to get the classification image?
-----
run rnn_fetch.py, it should generate a file called out_put_array.npy in ./result foler
copy out_put_array.npy to ./T_SNE, in the same directory with the tsne.py
run tsne.py, you should be able to see the classification result like this

![image](https://github.com/fhbzc/VAE_based_Behavior_Analysis/blob/master/Images/READMEIMAGE.png)

Some other questions:
 =======

 What does GroupClassificationColor.py do?
 -------
 GroupClassificationColor.py process the original data into valid input data, it's worth pointing out that GroupClassificationColor.py will also add a tag for each player, the tag stands for the player type index, and this type index is retrieved by observing agents' player, not by neural network, and we will test our observation-based classification against neural network retrieved classification.

      
 How do I read the classification result?
 ------
The location of points in the classification image stands for the "strategy" of different players, if two points are removed, it means the strategy they are using is very different, and vice verse. The color of the points in the image stand for the player type index determined by observation-based classification.
All points in the same color should be using the same strategy according to the observation-based classification, and close points are suggested to be using similar strategy by the neural network. If we see points in the same color are more likely to "gather together", we consider the classification result from neural network matches well with the classification result from observation result.      
