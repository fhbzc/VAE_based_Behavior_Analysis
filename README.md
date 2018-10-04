# VAE_based_Behavior_Analysis


Declaration: The code comes from a open-source project of Google(original link here https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)
I modify and apply it in agents behaviors analysis


How to generate trainable data?
-------

run <b>GroupClassificationColor.py</b>, it should generate 3 files: 
train_dataset.npz
valid_dataset.npz
test_dataset.npz

How to train the model?
---------
run <b>rnn_train.py</b>, the trained variable should be stored in log folder in the same directory as rnn_train.py

How to get the classification image?
-----
run <b>rnn_fetch.py</b>, it should generate a file called out_put_array.npy in ./result foler
copy out_put_array.npy to ./T_SNE, in the same directory with the tsne.py
run tsne.py, you should be able to see the classification result like this

![image](https://github.com/fhbzc/VAE_based_Behavior_Analysis/blob/master/Images/READMEIMAGE.png)

The code is tested under Python 2.7.12 and tensorflow 1.4.1

Some other questions
 =======

 What does GroupClassificationColor.py do?
 -------
 GroupClassificationColor.py processes the original data into valid input data, it's worth pointing out that GroupClassificationColor.py will also add a tag for each player, the tag stands for the player type index, and this type index is retrieved by observing agents' player, not by neural network, and we will test our observation-based classification against neural network retrieved classification.

      
 How do I read the classification result?
 ------
The location of points in the classification image stands for the "strategy" of different players, if two points are removed, it means the strategy they are using is very different, and vice verse. The color of the points in the image stand for the player type index determined by observation-based classification.
All points in the same color should be using the same strategy according to the observation-based classification, and close points are suggested to be using similar strategy by the neural network. If we see points in the same color are more likely to "gather together", we consider the classification result from neural network matches well with the classification result from observation result.    

 How to stop training?
 ----
 The training process is a wired-in infinite loop, so it won't automatically stop training and you need to stop it manually(like Ctrl-C)
 All parameters will be stored in ./result/ folder.
 
 Is overfitting a problem?
 -----
 Not exactly, I split the data into three parts, the train set, valid set and test set. Train set is used to train the network, and before storing the parameters, we need to test the parameter against the valid set(not the test), if the network behaves poorly on the valid set, these parameters won't be stored. And we only evaluate the ultimate performance with test set. This strategy can prevent the network from serious overfitting, although tiny overfitting is still possible.
 
 
