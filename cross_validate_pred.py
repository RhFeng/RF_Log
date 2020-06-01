import numpy as np
from pred_ints import pred_ints

def cross_validate_pred(model, Y_test, X_test, percentile, test_loc, test_inc):
    
    cs_score = np.zeros((len(test_loc), len(percentile)))
    
    for index in range(len(test_loc)):
        
        X_input = X_test[(index)*test_inc:(index+1)*test_inc,:]
        
        truth = Y_test[(index)*test_inc:(index+1)*test_inc,:]
        
        for j in range(len(percentile)):
        
            err_down, err_up = pred_ints(model, X_input, percentile[j])

            correct = 0.
            for i, val in enumerate(truth):
                if err_down[i] <= val <= err_up[i]:
                    correct += 1
                    
            cs_score[index,j] = correct/len(truth)        
                    
    return np.mean(cs_score,axis=0)

    
    
    