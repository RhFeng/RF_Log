import numpy as np

def train_test_split(x_trainwell, y_trainwell, test_loc, test_inc):
    
    n_features = x_trainwell.shape[1]

    
    test_len = len(test_loc) * test_inc

    X_train = []
    Y_train = []
          
    X_test = []
    Y_test = []
    
    for i in range(len(test_loc)):
        
        X_test = np.append(X_test,x_trainwell[test_loc[i]:test_loc[i]+test_inc,:])
        Y_test = np.append(Y_test,y_trainwell[test_loc[i]:test_loc[i]+test_inc,:])
        
    X_test = np.reshape(X_test,[test_len,n_features])
    Y_test = np.reshape(Y_test,[test_len,1]) 
    
    X_train = x_trainwell
    Y_train = y_trainwell
    
    index = np.zeros((test_len,1),dtype=int)
    
    a = 0
    
    for i in range(test_len):
        for j in range(len(x_trainwell)):
            if X_test[i,0] == x_trainwell[j,0] and X_test[i,n_features-1] == x_trainwell[j,n_features-1]: #and X_test[i,n_features-1] == x_trainwell[j,n_features-1]:
                
                index[a] = j
                a = a+1
    
    X_train = np.delete(X_train, index, 0)
    Y_train = np.delete(Y_train, index, 0)
    
    return X_train, Y_train, X_test, Y_test


