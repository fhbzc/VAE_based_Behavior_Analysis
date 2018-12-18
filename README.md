# VAE_based_Behavior_Analysis


Declaration: The code comes from a open-source project of Google(original link here   https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)  
I modify and apply it in agents behaviors analysis

Updates on October 12, 2018
-----
I change the directories of this project, move all datas into ./data/, and all log information will be shown in ./log, and all result(csv file, npy file and images) will be in ./result/

How to generate trainable data?
-------

run 

     python GroupClassificationColor.py
   
it should generate 3 files in ./data:  
train_dataset.npz  
valid_dataset.npz  
test_dataset.npz  

How to train the model?
---------
run 

     python rnn_train.py

 the trained variables should be stored in ./log folder  
 
How to get the classification image and inference?
-----
run 

     python rnn_train.py --test True

which will generate 3 files in the ./result folder:
    
*save.csv*   
A csv result which shows the newtork performance for each agents.
    An example of this csv is this:


  | id          | group_type    | prediction accuracy  |  total accuracy|tsne_x|  tsne_y  | action_0_prediction| action_0_actual| ... |
  |:----------:  |:--------:     |:-----------:      |:----------:    |:----:      |:-----:   |:-----:   |:-----:   | :-----:   | 
  |0            |  1            | 0.8                  |  0.9             |  0.013   |   0.001  | 0.908 | 1| ... |


           
The meaning of each column:  
<b>id</b>         agents id  
<b>group_type</b> the type of agents strategy, derived from my observation-based way.  
<b>prediction accuracy</b> the accuracy of network's prediction on agents furture actions.  
<b>total accuracy</b> the total accuracy of network's "prediction" over agents actions, note under the default setting, the first 10 steps are used to encode the strategy, which means the first 10 action "prediction" output is actually reconstruction.  
 <b>tsne_x, tsne_y</b> the cordinates of agents strategy in a 2-D space, derived with t_sne.  
<b> action_x_prediction </b> predicted possibility for this player to hunt stag(action 1) at time step x  
<b> action_x_actual </b> actual action of this player at time step x
   

*out_put_array.npy*  
A npy file contains all inference result and original input.  
*latent_cluster.png*  
A latent vector clustering image of all agents(just like the image below)

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
 
 
