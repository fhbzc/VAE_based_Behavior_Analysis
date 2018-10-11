


import numpy as np
from tsne import *
import os
import csv

class Eval:

    def __init__(self,file_to_load = "./result/out_put_array.npy"):
        self.file_to_load = file_to_load
        self.load = False
        self.data = None
        self.tsne_Y = []

    def Load(self):
        print("load file",self.file_to_load)


        self.data = np.load(self.file_to_load,encoding="latin1")
        self.latent = []
        self.prediction = []
        self.input = []
        self.label = []

        for i in range(len(self.data)):
            self.latent.append(self.data[:,0][i][0])
            self.prediction.append(self.data[:,1][i][0])
            self.input.append(self.data[:,2][i])
            self.label.append(self.data[:,3][i][:,0])

        self.latent = np.array(self.latent) # 200 * 32
        self.prediction = np.array(self.prediction) # 200 * 20, predicted action
        self.input = np.array(self.input) # 200 * 20 * 7
        self.label = np.array(self.label) # 200 * 20 actual action
        self.load = True

        #check the dimension of the data
        assert np.shape(self.data)[1] == 4

    def output_content_and_meaning(self):
        if self.load == False:
            print("Error, load the data by calling self.load before call 'output_content_and_meaning'")
            return False

        print("shape of predicted action",np.shape(self.prediction))
        print("shape of actual action",np.shape(self.label))
        print("shape of input(external state, rival actions, ect)",np.shape(self.input))
        print("shape of latent vector",np.shape(self.latent))
        return True

    def calculate_prediction_accuracy(self, threshold = 0.5, exclude_construction = True, reconstruction_step = 10):
        if self.load == False:
            print("Error, load the data by calling self.load before call 'calculate_prediction_accuracy'")
            return -1
        print("Calculate the prediction accuracy with threshold",threshold,"exclude_construction",exclude_construction,"reconstruction_step",reconstruction_step)
        # return -1 if an error occurs, real number if no error
        '''We calcualte the accuracy in that way,
             the network will output the probablity for one player to hunt stag at one time step, 
             the threshold parameter states if predicted probablity >= threshold, we think the predicted action is "Hunt Stag", else
             the action is "Hunt Rabbit"
             And we compare the prediction action with the actual action to output the accuracy
             exclude_construction is to decide whether we don't take reconstruction into consideration(under current setting, the first 10 steps are reconstruction
             intuitively, reconstruction will cause the calculated accuracy to be higher than the prediction accuracy
             '''
        total_n = 0
        accurate_n = 0
        for batchs in range(len(self.prediction)):
            for timestep in range(len(self.prediction[0])):
                if exclude_construction == True and timestep < reconstruction_step:
                    continue # ignore the first serveral steps
                if self.prediction[batchs][timestep] >= threshold:
                    predict_actoin = 1
                else:
                    predict_actoin = 0
                total_n += 1
                if predict_actoin == self.label[batchs][timestep]:
                    accurate_n += 1
        if accurate_n == 0:
            print("Total action is 0, probably an error due to incorrect parameter setting")
            return  -1
        accuracy = accurate_n * 1.0 / total_n
        return  accuracy

    def _simple_accuracy_cal(self,batchs, reconstruction_step = 10, threshold = 0.5):
        assert self.load == True, "Error, load the data by calling self.load before call '_simple_accuracy_cal'"
        total_n = 0
        accurate_n = 0
        total_excl_n = 0
        accurate_excl_n = 0

        for timestep in range(len(self.prediction[0])):

            if self.prediction[batchs][timestep] >= threshold:
                predict_actoin = 1
            else:
                predict_actoin = 0
            total_n += 1
            if predict_actoin == self.label[batchs][timestep]:
                accurate_n += 1

            if timestep >= reconstruction_step:
                if self.prediction[batchs][timestep] >= threshold:
                    predict_actoin = 1
                else:
                    predict_actoin = 0
                total_excl_n += 1
                if predict_actoin == self.label[batchs][timestep]:
                    accurate_excl_n += 1
        assert accurate_excl_n > 0, "Unkonwn error in '_simple_accuracy_cal', accurate_excl_n == 0"
        assert accurate_n > 0, "Unkonwn error in '_simple_accuracy_cal', accurate_n == 0"

        return   1.0 * accurate_excl_n / total_excl_n, 1.0 * accurate_n / total_n

    def _tsne_call(self):
        assert self.load == True, "Error, load the data by calling self.load before call '_tsne_call'"
        self.tsne_Y = tsne(self.latent, 2, 10, 199)

    def get_tsne(self):
        self._tsne_call()

    def save_as_csv(self):
        if self.load == False:
            print("Error, load the data by calling self.load before call 'save_as_csv'")
            return False

        store_direct = "./result/"
        if os.path.exists(store_direct) == False:
            os.mkdir(store_direct)

        self._tsne_call() # get tsne_distribution and store it in self.tsne_Y
        with open(store_direct+"save.csv",'w') as f:
            f_csv = csv.writer(f)
            # write the title line
            header = ["id","group_type","prediction accuracy","total_accuracy","tsne_x","tsne_y"]
            # add the headers of 20 rounds actual action and prediction action
            for i in range(len(self.label[0])):
                header.append("action_"+str(i)+"_prediction")
                header.append("action_"+str(i)+"_actual")

            f_csv.writerow(header)
            for batches in range(len(self.prediction)):
                row = []
                row.append(self.input[batches][0][6]) # player id
                row.append(self.input[batches][0][5]) # player type
                pre_a,tot_a = self._simple_accuracy_cal(batches)
                row.append(pre_a) # prediction accuracy
                row.append(tot_a) # total accuracy
                row.append(self.tsne_Y[batches,0]) # tsne_x
                row.append(self.tsne_Y[batches,1]) # tsne_y
                for time_step in range(len(self.label[0])):
                    row.append(self.prediction[batches,time_step])
                    row.append(self.label[batches,time_step])
                f_csv.writerow(row)
            f.close()
        return True


    def drawing_latent_image(self,address):
        if os.path.exists(address) == False:
            os.mkdir(address)
        if self.load == False:
            print("Error, load the data by calling self.load before call 'drawing_latent_image'")
            return False
        if len(self.tsne_Y) == 0:
            print("Error in drawing_latent_image, call 'self.get_tsne' to get a tsne_map before call 'drawing_latent_image'")
            return False
        if address[-1] != '/':
            address += "/" # add to make it a directory
        Plot(self.tsne_Y,self.input[:,0,5],address)

        return True
