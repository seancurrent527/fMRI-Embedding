#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#this script implements a fMRI analysis framework, connectome-based predictive modeling,
#and integrates it with various embedding techniques to predict participant age and task performance
"""

@author: Anita Shankar shankar.85@osu.edu
last edited December 7th 2020
"""
import argparse
import numpy as np
import scipy.io as sio
import scipy.stats
import matplotlib.pylab as plt
# try to import factorization library
try:
    from embeddings.factorization import *
except:
    print("ERROR: Could not load 'factorization' library")
# try to import word2vec library
try:
    from embeddings.word2vec import *
except:
    print("ERROR: Could not load 'word2vec' library")
# example of how to run in terminal: python Embedding_CPM.py -embed none -target age
# embed options: none, CBOW, skip, matrix, tensor
# target options: age, task

#%% ## MAIN FUNCTION
def cpm_main(x,y,kfolds,pthresh,embedding_type):
    s=np.size(x,0) # get number of nodes
    mat = np.empty((int((s*s-s)/2),np.size(x,2))) #create empty matrix

    # run CV
    y_predict_high,y_predict_low,res_high_all,res_low_all = cpm_cv(x,y,kfolds,pthresh,embedding_type) #implement cross validation
    
    # assess model performance by correlating predicted and actual scores
    r_high,p_high = scipy.stats.pearsonr(y_predict_high,y)
    r_low,p_low = scipy.stats.pearsonr(y_predict_low,y)

    return r_high,p_high,y_predict_high,r_low,p_low,y_predict_low,res_high_all,res_low_all
###            
        
#%% ## CROSS VALIDATION
def cpm_cv(x,y,kfolds,pthresh,embedding_type):
    #split the data into training and testing
    nsubs=np.size(x,2) #get the numbers of subjects
    nfeats=np.size(x,1) #get the number of features
    randinds=np.random.permutation(nsubs) #get a random order for each subject
    ksample=int(np.floor(nsubs/kfolds)) #get number of people in testing set
    y_test = np.empty((kfolds,ksample))#initialize array for y testing set
    y_predict_high = np.empty((kfolds,ksample))#initialize array for y predictions in high network
    y_predict_low = np.empty((kfolds,ksample))#initialize array for  predictions in low network
    res_high_all = [] #initialize residual matrix for high network
    res_low_all = [] #initialize residual matrix for low network
    
    for leftout in range(kfolds): #for each CV loop
        print("Currently on fold ",leftout)
        si=((leftout)*ksample)
        fi=si+ksample
        testinds=randinds[si:fi] #find the indices of data for the testing set
        traininds=np.setdiff1d(randinds,testinds) #find the indices of the data for the training set
    
        nsubs_in_fold=np.size(testinds,0) #calculate the number of subs in each kfold
        
        # assign data to train and test groups
        x_train= x[:,:,traininds]
        y_train= y[traininds]
        x_test= x[:,:,testinds]
        y_test[leftout,:nsubs_in_fold+1]= np.squeeze(y[testinds]);
        
        
        # training
        p,pmask,mdl_high,mdl_low,res_high,res_low,model = cpm_train(x_train,y_train,pthresh,embedding_type) #call training function
        res_high_all.append(res_high) #save the residuals for high network model
        res_low_all.append(res_low) #save the residuals for low network model
        
        # testing
        y_predict_high[leftout,0:nsubs_in_fold+1],y_predict_low[leftout,0:nsubs_in_fold] = cpm_test(x_test,mdl_high,mdl_low,pmask,model) #call the testing function
    
    # Save the predictions from all the kfolds
    y_predict_high_reshape = np.empty(len(randinds))
    y_predict_high_reshape[randinds]=np.reshape(y_predict_high,-1)

    y_predict_low_reshape = np.empty(len(randinds))
    y_predict_low_reshape[randinds]=np.reshape(y_predict_low,-1)

        
    return y_predict_high_reshape,y_predict_low_reshape,res_high_all,res_low_all
###
    
#%% ## TRAINING FUNCTION
def cpm_train(x,y,pthresh,embedding_type):
    x=np.transpose(x)
    # create embedding from training data
    xm = x.mean(axis=0)
    
    ### Needed for CBOW and Skip-Gram ####
    walk = random_walk(xm, steps = 1000)
    one_hot = np.zeros((len(walk), 268))
    for i, pos in enumerate(walk):
        one_hot[i, :] = xm[pos]
    
    #### Train the appropriate model
    if embedding_type == "none":
        # Classic CPM, no embedding function ######
        x_embedded = x
        nx,ny,nz = np.shape(x_embedded)
        x_embedded = np.reshape(x_embedded,(nx,ny*nz))
        x_embedded=np.transpose(x_embedded)
        model = "none"
        
    elif embedding_type == "CBOW":
        # Continuous Bag of Words (CBOW) ######
        model = CBOW(268, 64, 2, 0.1)
        model.train_from_feature_seq(one_hot, epochs = 200)
        # use embedding to encode training data
        x_embedded = model.encode(x)
        nx,ny,nz = np.shape(x_embedded)
        x_embedded = np.reshape(x_embedded,(nx,ny*nz))
        x_embedded=np.transpose(x_embedded)  
        
    elif embedding_type == "skip":
        #Skip-Gram ######
        model = Skip_Gram(268, 64, 2, 0.1)
        model.train_from_feature_seq(one_hot, epochs = 200)
        # use embedding to encode training data
        x_embedded = model.encode(x)
        nx,ny,nz = np.shape(x_embedded)
        x_embedded = np.reshape(x_embedded,(nx,ny*nz))
        x_embedded=np.transpose(x_embedded)  

    elif embedding_type == "matrix":
        ### Matrix factorization ######
        model = MatrixFactorization(xm, 64)
        model.fit(200, 0.0001)
        #use embedding to encode training data
        x_embedded = model.encode(x)
        x_embedded=np.transpose(x_embedded)  
        
    elif embedding_type == "tensor":   
        ### Tensor Factorization ########
        model = TensorFactorization(x, 268)
        model.fit(50)
        # use embedding to encode training data
        x_embedded = model.encode(x)
        x_embedded=np.transpose(x_embedded)    
    ###########

    
    # select significant features across subjects
    nsubs = np.size(x_embedded,1) #define the number of subs in training set
    nnodes = np.size(x_embedded,0) #define number of nodes in training set
    r = np.empty(nnodes) #initialize array for r correlation values
    p = np.empty(nnodes) #initialize array for p values
    
    #for all the data
    for ii in range(np.size(x_embedded,0)):
        r[ii],p[ii] = scipy.stats.pearsonr(x_embedded[ii,:],y) #correlate each matrix cell with behavior across all subjects

    
    # generate masks based on p threshold
    pmask_pos=(r>0)*(p<pthresh) #gives a mask of edges positively correlated with behavior
    pmask_neg=(r<0)*(p<pthresh) #gives a mask of edges negatively correlated with behavior
    pmask = pmask_pos.astype('int') - pmask_neg.astype('int') #create one mask of both
    high_strength = np.zeros(nsubs) #initialize a high strength array for the subs
    low_strength = np.zeros(nsubs) #initialize a low strength array for the subs
    
    # summarize selected features for each subject
    for ii in range(nsubs): #for each subject
        high_strength[ii]=np.mean(x_embedded[pmask_pos,ii]) #calculate a single number to represent their matrix (average of positively correlated significant edges)
        low_strength[ii]=np.mean(x_embedded[pmask_neg,ii])#calculate a single number to represent their matrix (average of negatively correlated significant edges)
    # check for nan
    if np.isnan(high_strength).any():
        high_strength = np.zeros(nsubs) #if there are any Nans, replace with zeros
    if np.isnan(low_strength).any():
        low_strength = np.zeros(nsubs) #if there are any Nans, replace with zeros
    
    
    # fit model to features
    # check if strengths are all zero, and if so replace with average of y
    if not np.any(high_strength):
        avg_y = np.mean(y)
        mdl_high = 0,avg_y
        res_high = np.sum((avg_y-y)**2)
    else:
        mdl_high,res_high,_,_,_=np.polyfit(high_strength,np.transpose(y),1,full=True)
    if not np.any(low_strength):
        avg_y = np.mean(y)
        mdl_low = 0,avg_y
        res_low = np.sum((avg_y-y)**2)
    else:
        mdl_low,res_low,_,_,_=np.polyfit(low_strength,np.transpose(y),1,full=True)
    return p,pmask,mdl_high,mdl_low,res_high,res_low,model
###


#%% ## TESTING
def cpm_test(x,mdl_high,mdl_low,pmask,model):
    x=np.transpose(x)
    # check if using embedding techniques or standard CPM
    if model == "none":
        x_embedded = x
    else:
        # use embedding to encode testing data
        x_embedded = model.encode(x)
    
    ## Check to make sure x_test dimension is correct
    if x_embedded.ndim == 3:
        nx,ny,nz = np.shape(x_embedded)
        x_embedded = np.reshape(x_embedded,(nx,ny*nz))
    ##
    x_embedded=np.transpose(x_embedded)    
    
    # select significant features across subjects and intialize variables
    nsubs = np.size(x_embedded,1)
    high_strength = np.zeros(nsubs)
    low_strength = np.zeros(nsubs)
    y_predict_high = np.zeros(nsubs)
    y_predict_low = np.zeros(nsubs)
    
    # iterate through all subjects
    for ii in range(nsubs):
        high_strength[ii]=np.mean(x_embedded[pmask>0,ii]) #now calculate summary statistic in the testing set from edges derived in training set
        low_strength[ii]=np.mean(x_embedded[pmask<0,ii]) #now calculate summary statistic in the testing set from edges derived in training set
        # check for NaNs and get rid of them if found
        if np.isnan(high_strength[ii]):
            high_strength[ii] = 0
        if np.isnan(low_strength[ii]):
            low_strength[ii] = 0
        # make prediction on test subject
        y_predict_high[ii]=mdl_high[0]*high_strength[ii] + mdl_high[1] #apply model derived in training set to make predictions
        y_predict_low[ii]=mdl_low[0]*low_strength[ii] + mdl_low[1] #apply model derived in training set to make predictions
        
    return y_predict_high,y_predict_low
###
                          
#%% Main Function
if __name__ == "__main__":  
    
    
    ############################## Part 1: Implement User Interface
    
    #import argparse library
    import argparse
    #creating parser object to turn command line strings into python objects
    parser=argparse.ArgumentParser()
    #create input argument for number of kfolds
    #parser.add_argument('-kfolds')
    #create input argument for embedding type
    parser.add_argument('-embed')
    #create input argument for target variable
    parser.add_argument('-target')



    #### default parameters
    kfolds = 20
    embedding_type = "tensor"  # available options are "none", "CBOW", "skip", "matrix", "tensor"
    target = "task" # options are "age", "task", "motion"
    ####
    
    #actually take the command line inputs and put them in object 'args'
    args=parser.parse_args()
    #create and save input args as strings,ints,and floats
    #kfolds=int(args.kfolds)
    embedding_type=str(args.embed)
    target=str(args.target)
    
    
    # Load data
    x = sio.loadmat("Data/conmat_240.mat")['full_conmat']
    z = sio.loadmat("Data/meanFD_240.mat")['meanFD']
    # Load target variable
    if target == "age":
        y = sio.loadmat("Data/age_240.mat")['age']
    elif target == "task":
        y = sio.loadmat("Data/task_240.mat")['mean_rxn']
    elif target == "motion":
        y = sio.loadmat("Data/meanFD_240.mat")['meanFD']

    # get rid of extra singular dimension
    y = np.squeeze(y)
    z = np.squeeze(z)
    
    # replace NaNs with zeros
    x = np.nan_to_num(x)
    
    # some useful constants
    pthresh = 0.01
    num_subs=np.size(y)
    num_train = num_subs-(num_subs/kfolds)
    
    # Run CPM
    r_high,p_high,y_predict_high,r_low,p_low,y_predict_low,res_high_all,res_low_all = cpm_main(x,y,kfolds,pthresh,embedding_type)
    
    # print correlation between predicted and observed scores
    print("Low r = %1.4f, p = %1.4f" % (r_low, p_low))
    print("High r = %1.4f, p = %1.4f" % (r_high, p_high))
    # calculate mean average percent error if wanted
    MAPElow=np.mean(np.abs(y-y_predict_low)/np.abs(y))
    MAPEhigh=np.mean(np.abs(y-y_predict_high)/np.abs(y))
    #print("MAPE_low =%1.4f" % (MAPElow))
    #print("MAPE_high =%1.4f" % (MAPEhigh))
    
    #calculate and print sum of squared error
    SSElow=np.mean((y-y_predict_low)**2)
    SSEhigh=np.mean((y-y_predict_high)**2)
    print("SSE_low =%1.4f" % (SSElow))
    print("SSE_high =%1.4f" % (SSEhigh))
    
    #calculate correlation of predicted scores and motion artifact
    motionRlow, motionPlow=scipy.stats.pearsonr(y_predict_low,z)
    motionRhigh, motionPhigh=scipy.stats.pearsonr(y_predict_high,z)
    print("Low motion = %1.4f, p = %1.4f" % (motionRlow, motionPlow))
    print("High motion = %1.4f, p = %1.4f" % (motionRhigh, motionPhigh))  



    ## If you want to plot correlation and Mean Sum of Squares Error
    
    
    # m,res_test,_,_,_ = np.polyfit(y,y_predict_low,1,full=True)
    # plt.figure(1)
    # plt.scatter(y,y_predict_low)
    # plt.xlabel("Actual")
    # plt.ylabel("Predicted")
    # plt.title("Target Variable = " + target)
    # plt.plot(y,m[0]*y+m[1])
                          
    # plt.figure(2)
    # plt.violinplot(np.array(res_low_all)/num_train) #plot training sets' MSE
    # plt.plot(1,np.mean((y-y_predict_low)**2),'ro') #plot testing set's MSE
    # plt.title("Target Variable = " + target)
    # plt.ylabel("Mean Sum of Square Error")
                          
  # train_size = [ ] 2 20
 # train_mse = [ ]
#   test_mse = [ ]                         
                          
                          
                          
                          
                          
                          