'''
 This code is used for DARPA Ground Truth Project, Stag Hunt game
 Author: Hongbo Fang
 Date:  August 14 2018


 Group description:
    Group 1(belongs to Type 1):
        Always choose to hunt rabbit
    Group 2(belongs to Type 1):
        Initially choose to hunt rabbit, if f2 = 1, eternal flip(see the definition here:
                            https://docs.google.com/document/d/1R6Bw6HbzooBT-Ztvka5Oc-rF76JFFYiNG6C4m-xkuqk/edit
                            )
    Group 3(belongs to Type 1):
        Choose to hunt rabbit if f0 = 0
        if f0 = 1, repeat the rivalry action in previous time step
    Group 4(belongs to Type 2):
        Always choose to hunt stag
    Group 5(belongs to Type 2):
        Initially choose to hunt stag, if f2 = 1, eternal flip
    Group 6(belongs to Type 3):
        Firstly hunt rabbit, then hunt stag, then rabbit then stage ....
    Group 7(belongs to Type 4):
        Firstly hunt stag, then hunt rabbit, then stag then rabbit...
    Group 8(belongs to Type 5):
        Action in time step 1 cannot be predicted(I assume player chooses the first action randomly)
        for all other actions, player repeat the rivalry action in previous time step
    Group 9(belongs to Type 5):
        Action in time step 1 cannot be predicted(I assume player chooses the first action randomly)
        for all other actions, if f0 = 0: player repeat the rivalry action in previous time step
                               if f0 = 1: player chooses to hunt stag
    Group 10(belongs to Type 5)
        Action in time step 1 cannot be predicted(I assume player chooses the first action randomly)
        for all other actions, if f2 = 0: player repeat the rivalry action in previous time step
                               if f2 = 1: player choose the different action from the rivalry action in last time step
    Group 11(belongs to Type 5)
        Action in time step 1 cannot be predicted(I assume player chooses the first action randomly)
        action_to_be_done
        to decide the action_to_be_done for current time step, player will do 3 things(in the following order):
            set action_to_be_done to be the rivalry action in previous timestep
            if f0 = 1, set action_to_be_done to hunt stag
            if f2 = 1, set action_to_be_done to (not action_to_be_done)
        execute action_to_be_done

 Implementation instruction:
    Put original csv data in the same directory with the current python program
    rename csv data to "original_data.csv"
    run the program, the result will be output in the command line, Unsolved stands for the player ID that can not be classified


'''
import csv
import numpy as np
import random
csv_reader = csv.reader(open("original_data.csv"))
content = []
for row in csv_reader:
    content.append(row)
content.pop(0)

time_step = 20 # time step for each game
Games = []     # contains all game state, each of 20 time steps

assert len(content)%time_step == 0

for i in range(len(content)//time_step):
    one_game = [] # store information of one game
    for t in range(time_step):
        one_game.append(content[i*time_step + t])

    Games.append(one_game)

'''
 Type 1, Group 1:
    all zero
'''

Unsolved = []
Type = []
Group = []

for i in range(5): # 5 types
    Type.append([])

for i in range(3):# 3 groups in type 1
    Type[0].append([])

for i in range(2): # 2 groups in type 2
    Type[1].append([])

Type[2].append([]) # 1 group in type 3


Type[3].append([]) # 1 groups in type 4


for i in range(4): # 4 groups in type 5
    Type[4].append([])

index = 0
for game in Games:

    for player_id in range(2): # 2 players

        # check whether T1_G1
        Select = True
        for t in range(time_step):
            if int(game[t][6 + player_id]) != 0:
                Select = False
                break
        if Select == True:
            Type[0][0].append(int(game[0][10 + player_id]))
            continue

        # check whether T1_G2
        Select = True
        current_state = False
        for t in range(time_step):
            if int(game[t][5]) == 1:
                current_state = not current_state

            if int(game[t][6 + player_id]) != current_state:
                Select = False
                break

        if Select == True:
            Type[0][1].append(int(game[0][10 + player_id]))
            continue

        # check whether T1_G3
        Select = True
        for t in range(time_step):
            if int(game[t][3]) == 1:
                if int(game[t][6 + player_id]) != int(game[t-1][7 - player_id]):
                    Select = False
                    break

            else:
                if int(game[t][6 + player_id]) != 0:
                    Select = False
                    break

        if Select == True:
            Type[0][2].append(int(game[0][10 + player_id]))
            continue



        # check whether T2_G4
        Select = True
        for t in range(time_step):
            if int(game[t][6 + player_id]) != 1:
                Select = False
                break

        if Select == True:
            Type[1][0].append(int(game[0][10 + player_id]))
            continue


        # check whether T2_G5
        Select = True
        current_state = True
        for t in range(time_step):
            if int(game[t][5]) == 1:
                current_state = not current_state

            if int(game[t][6 + player_id]) != current_state:
                Select = False
                break

        if Select == True:
            Type[1][1].append(int(game[0][10 + player_id]))
            continue



        # check whether T3_G6
        Select = True
        current_state = False
        for t in range(time_step):
            if int(game[t][6 + player_id]) != current_state:
                Select = False
                break
            current_state = not current_state

        if Select == True:
            Type[2][0].append(int(game[0][10 + player_id]))
            continue

        # check whether T4_G7
        Select = True
        current_state = True
        for t in range(time_step):
            if int(game[t][6 + player_id]) != current_state:
                Select = False
                break
            current_state = not current_state

        if Select == True:
            Type[3][0].append(int(game[0][10 + player_id]))
            continue


        # check whether T5_G8
        Select = True
        for t in range(1,time_step):
            if int(game[t][6 + player_id]) != int(game[t - 1][7 - player_id]):
                Select = False
                break

        if Select == True:
            Type[4][0].append(int(game[0][10 + player_id]))
            continue


        # check whether T5_G9
        Select = True
        for t in range(1,time_step):
            if int(game[t][3]) == 1:
                if int(game[t][6 + player_id]) != 1:
                    Select = False
                    break
            else:
                if int(game[t][6 + player_id]) != int(game[t - 1][7 - player_id]):
                    Select = False
                    break

        if Select == True:
            Type[4][1].append(int(game[0][10 + player_id]))
            continue


        # check whether T5_G10
        Select = True
        for t in range(1,time_step):
            if int(game[t][5]) == 1:
                if int(game[t][6 + player_id]) == int(game[t - 1][7 - player_id]):
                    Select = False
                    break
            else:
                if int(game[t][6 + player_id]) != int(game[t - 1][7 - player_id]):
                    Select = False
                    break

        if Select == True:
            Type[4][2].append(int(game[0][10 + player_id]))
            continue



        # check whether T5_G11
        Select = True
        for t in range(1,time_step):

            should_step = int(game[t - 1][7 - player_id])
            if int(game[t][3]) == 1:
                should_step = 1
            if int(game[t][5]) == 1:
                should_step = 1 - should_step
            if int(game[t][6 + player_id]) != should_step:
                Select = False
                break

        if Select == True:
            Type[4][3].append(int(game[0][10 + player_id]))
            continue


        Unsolved.append(int(game[0][10 + player_id]))


Group_ID = 1
for i in range(len(Type)):
    for j in range(len(Type[i])):
        print("Group ID",Group_ID)
        print(Type[i][j])
        Group_ID += 1
print("Unsolved")
print(Unsolved)








sequence_length = 20
number = 0
valid_ratio = 0.1
test_ratio = 0.1
player_one_batch = len(content)//sequence_length
total_input = np.zeros([player_one_batch*2,sequence_length,7]) # two players in one game, 3 paramaters plus rivalry
# 7 dimension input at each time for each player:
# f0
# f1
# f2
# rival move
# total reward(not used at all, can simply ignore that)
# Group index(not used in training)
# player id(not used in training)
total_output = np.zeros([player_one_batch*2,sequence_length,1])
print("Debug information")
print("shape of total_input",np.shape(total_input))
print("shape of total_output",np.shape(total_output))

def find_ID_Type_index(Type, id):

    for i in range(len(Type)):
        for j in range(len(Type[i])):
            if id in Type[i][j]:
                return i


    return  5

def find_ID_Group_index(Groups, id):

    Group_ID = 0
    for i in range(len(Type)):
        for j in range(len(Type[i])):
            if id in Type[i][j]:
                return Group_ID
            Group_ID += 1

    return  11

for index in range(len(content)):
    row = content[index]

    # Player 1
    total_input[int(row[1]),int(row[2]),0] = int(row[3])
    total_input[int(row[1]),int(row[2]),1] = int(row[4])
    total_input[int(row[1]),int(row[2]),2] = int(row[5])
    if int(row[2]) == 0: # first time
        total_input[int(row[1]),int(row[2]),3] = 0.5
        total_input[int(row[1]),int(row[2]),4] = 0.5 # playe 1 ID
    else:
        total_input[int(row[1]),int(row[2]),3] = int(content[index-1][7]) #rivalry move
        total_input[int(row[1]),int(row[2]),4] = int(content[index-1][6]) #my move


    # Player 2
    total_input[int(row[1])+player_one_batch,int(row[2]),0] = int(row[3])
    total_input[int(row[1])+player_one_batch,int(row[2]),1] = int(row[4])
    total_input[int(row[1])+player_one_batch,int(row[2]),2] = int(row[5])
    if int(row[2]) == 0: # first time
        total_input[int(row[1])+player_one_batch,int(row[2]),3] = 0.5
        total_input[int(row[1])+player_one_batch,int(row[2]),4] = 0.5
    else:
        total_input[int(row[1])+player_one_batch,int(row[2]),3] = int(content[index-1][6]) #rivalry move
        total_input[int(row[1])+player_one_batch,int(row[2]),4] = int(content[index-1][7]) #my move



    total_input[int(row[1]),int(row[2]),5] = find_ID_Type_index(Type,int(row[10]))
    total_input[int(row[1]),int(row[2]),6] = int(row[10])
    total_output[int(row[1]),int(row[2]),0] = int(row[6])


    total_input[int(row[1])+player_one_batch,int(row[2]),5] = find_ID_Type_index(Type,int(row[11]))
    total_input[int(row[1])+player_one_batch,int(row[2]),6] = int(row[11])
    total_output[int(row[1])+player_one_batch,int(row[2]),0] = int(row[7])


train_dataset_input = []
valid_dataset_input = []
test_dataset_input = []

train_dataset_output = []
valid_dataset_output = []
test_dataset_output = []
for i in range(player_one_batch*2):
    if i < 150:
        train_dataset_input.append(total_input[i])
        train_dataset_output.append(total_output[i])
    elif i< 175:
        valid_dataset_input.append(total_input[i])
        valid_dataset_output.append(total_output[i])
    else:
        test_dataset_input.append(total_input[i])
        test_dataset_output.append(total_output[i])

train_dataset_input = np.array(train_dataset_input)
train_dataset_output = np.array(train_dataset_output)
valid_dataset_input = np.array(valid_dataset_input)
valid_dataset_output = np.array(valid_dataset_output)
test_dataset_input = np.array(test_dataset_input)
test_dataset_output = np.array(test_dataset_output)





# print("np.shape(Train)",np.shape(Train))
print("np.shape(Train[0])",np.shape(train_dataset_input))
print("np.shape(Train[1])",np.shape(train_dataset_output))

# print("np.shape(Valid)",np.shape(Valid))
print("np.shape(Valid[0])",np.shape(valid_dataset_input))
print("np.shape(Valid[1])",np.shape(valid_dataset_output))

# print("np.shape(Test)",np.shape(Test))
print("np.shape(Test[0])",np.shape(test_dataset_input))
print("np.shape(Test[1])",np.shape(test_dataset_output))

np.savez("train_dataset.npz",input = train_dataset_input,output = train_dataset_output )
np.savez("valid_dataset.npz",input = valid_dataset_input,output = valid_dataset_output)
np.savez("test_dataset.npz",input = test_dataset_input,output = test_dataset_output)







