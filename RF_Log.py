import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker
plt.rcParams["font.family"] = "Times New Roman"

#%%
# load the data
x_trainwell = np.loadtxt('x_trainwell1.dat')

# CAL PHI GR DR MR PE RHO
features = ['CAL', 'PHI', 'GR', 'DR', 'MR', 'PE', 'RHO']
n_features = len(features)

# DTS
y_trainwell = np.loadtxt('y_trainwell1.dat')

y_trainwell = np.reshape(y_trainwell,[-1,1])


from plot_well import plot_well_feature, plot_well_target

# define the test location, increment, depth interval
test_loc = [400,700]  #400,700 550,900

test_inc = 100

dz = 0.1524

plot_well_feature(x_trainwell, test_loc, test_inc, dz)

plot_well_target(y_trainwell, test_loc, test_inc, dz)

#%%
from train_test_split import train_test_split

X_train, Y_train, X_test, Y_test = train_test_split(x_trainwell, y_trainwell, test_loc, test_inc)

pd_data = pd.DataFrame(data = X_train, columns = features)

g = sns.pairplot(pd_data,corner=True,markers="o",
                  plot_kws=dict(s=5, edgecolor="b",  linewidth=1))

g.fig.set_figwidth(8)
g.fig.set_figheight(8)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

#%%
# standize the matrix for training data
scaler = StandardScaler()
x_trainwell = scaler.fit_transform(x_trainwell)

# split the train and test data

X_train, Y_train, X_test, Y_test = train_test_split(x_trainwell, y_trainwell, test_loc, test_inc)

ymin = np.min(Y_test) - 5
ymax = np.max(Y_test) + 5

#%%
RF = RandomForestRegressor(random_state = 42)

param_grid = { 
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,8,12]}


CV_RF = GridSearchCV(estimator=RF, param_grid=param_grid, cv= 5)
CV_RF.fit(X_train, Y_train)


#%%
# random forest model

RF = RandomForestRegressor(n_estimators = CV_RF.best_params_['n_estimators'], 
                           max_features = CV_RF.best_params_['max_features'],
                           max_depth = CV_RF.best_params_['max_depth'], min_samples_leaf=1,random_state = 42)

RF.fit(X_train, Y_train)

Y_predict = RF.predict(x_trainwell)

from plot_well import plot_well_predict
plot_well_predict(y_trainwell, Y_predict, test_loc, test_inc, dz)

#%%
# scattering the test prediction
Y_test_predict = RF.predict(X_test)

predict = np.vstack([Y_test_predict.ravel(),Y_test.ravel()])

df = pd.DataFrame(np.transpose(predict), columns=["Prediction", "Reference"])

pt = sns.jointplot(x="Prediction", y="Reference", edgecolor="b",data=df,size=4.5,xlim = (ymin, ymax), ylim = (ymin, ymax))

CC = np.corrcoef(Y_test.ravel(), Y_test_predict.ravel())

pt.fig.text(0.18, 0.75,'CC = %.4f' %CC[0,1], fontsize=10)
pt.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(10))
pt.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(10))

#%%
#prediction interval

from plot_well import plot_pred_interval

plot_pred_interval(RF, Y_test, X_test, test_loc, test_inc, dz)

#%%
# percentages
percentile =[90, 60, 30]

from cross_validate_pred import cross_validate_pred

cs_score = cross_validate_pred(RF, Y_test, X_test, percentile, test_loc, test_inc)

print(cs_score)

#%%
# feature importance
importances = RF.feature_importances_

indices = np.argsort(importances)

fig = plt.figure(figsize=(5, 5))
fig.set_facecolor('white')
ax1 = fig.add_subplot()

plt.title('Feature Importances', fontsize = 14)
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize = 12)
plt.plot([0,0],[1,1])
plt.xlabel('Relative Scores', fontsize = 12)
ax1.tick_params(labelsize = 10)  
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['right'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
#%%
index =  [i for i in range(len(importances)) if importances[i] >= 0.05]

RF.fit(X_train[:,index], Y_train)

Y_test_predict = RF.predict(X_test[:,index])

predict = np.vstack([Y_test_predict.ravel(),Y_test.ravel()])

df = pd.DataFrame(np.transpose(predict), columns=["Prediction", "Reference"])

pt = sns.jointplot(x="Prediction", y="Reference", edgecolor="b",data=df,size=4.5,xlim = (ymin, ymax), ylim = (ymin, ymax))

CC = np.corrcoef(Y_test.ravel(), Y_test_predict.ravel())

pt.fig.text(0.18, 0.75,'CC = %.4f' %CC[0,1], fontsize=10)

pt.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(10))
pt.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(10))

#%%
# prediction interval

plot_pred_interval(RF, Y_test, X_test[:,index], test_loc, test_inc, dz)

#%%
#percentages

cs_score = cross_validate_pred(RF, Y_test, X_test[:,index], percentile, test_loc, test_inc)

print(cs_score)

#%%
# now PCA

pca = PCA()

pca.fit(x_trainwell)

scree = pca.explained_variance_ratio_.cumsum() * 100

pcs_xtrainwell = pca.fit_transform(x_trainwell)

fig, ax = plt.subplots(figsize=(4, 4)) 
scree = pca.explained_variance_ratio_.cumsum() * 100
plt.plot(np.arange(1,8), scree, 'bo-', linewidth=2)
plt.plot([4], [scree[3]], marker='*', color='r',markersize=12)
ax.set_xlim([1,7])
ax.set_ylim([40,100])
ax.set_xlabel('Principal Component',fontsize= 14)
ax.set_ylabel('Variance Explained (%)',fontsize = 14)
ax.grid(linestyle='-.',linewidth=1.5)
ax.tick_params(labelsize = 12)  
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)


#%%

X_train_pca, _, X_test_pca, _ = train_test_split(pcs_xtrainwell[:,:4], y_trainwell, test_loc, test_inc)

RF.fit(X_train_pca, Y_train)

Y_test_predict = RF.predict(X_test_pca)

predict = np.vstack([Y_test_predict.ravel(),Y_test.ravel()])

df = pd.DataFrame(np.transpose(predict), columns=["Prediction", "Reference"])

pt = sns.jointplot(x="Prediction", y="Reference", edgecolor="b",data=df,size=4.5,xlim = (ymin, ymax), ylim = (ymin, ymax))

CC = np.corrcoef(Y_test.ravel(), Y_test_predict.ravel())

pt.fig.text(0.18, 0.75,'CC = %.4f' %CC[0,1], fontsize=10)

pt.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(10))
pt.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(10))


#%%
#prediction interval

plot_pred_interval(RF, Y_test, X_test_pca, test_loc, test_inc, dz)

#%%
#percentage

cs_score = cross_validate_pred(RF, Y_test, X_test_pca, percentile, test_loc, test_inc)

print(cs_score)
















