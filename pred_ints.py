import numpy as np

def pred_ints(model, X, percentile):
    err_down = []
    err_up = []

    for x in range(len(X)):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(np.reshape(X[x],[1,-1]))[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
        
    return err_down, err_up
