
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error


# Options:
feature_list = ['Jitter(%)', 'Jitter(Abs)',	'Jitter:RAP',	'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer',	'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
want_to_save = False
save_to = 'results3.csv'
want_save_figs = False
fig_folder = 'figs/1D_loop_1'


# Reading the datasets.
training_data = pd.read_csv('datasets/training_data_M.csv')
validation_data = pd.read_csv('datasets/validating_data_M.csv')

# get labels
updrs = training_data['total_UPDRS'].to_numpy()
updrs_validt = validation_data['total_UPDRS'].to_numpy()

# loading save file:
result_df = pd.read_csv(save_to)

# This loop fits a 1D linear regression model to all feature variables in 'feature_list'
# Scatter plots are printed and results (losses) are saved.
figN = 1
for feature_name in feature_list:

  # Selecting the feature for the 1D linear regression:
  feature = training_data[feature_name].to_numpy()


  # Training the linear regression model:

  # Reshape to 2D arrays (LinearRegression() requires 2D input)
  updrs = updrs.reshape(-1,1)
  feature = feature.reshape(-1,1)

  # Computing the fit and the prediction.
  regObj = LinearRegression().fit(feature, updrs)
  pred_updrs = regObj.predict(feature)


  # Computing error:
  loss_training = np.sum(np.square(updrs - pred_updrs))
  diff_train = updrs - pred_updrs
  #print(diff_train[0:5])
  #print(updrs.shape)
  #print(pred_updrs.shape)

  # Plotting the data:
  fig1 = plt.figure(num=figN)
  figN = figN + 1
  ax1 = fig1.add_subplot()
  ax1.scatter(feature, updrs)
  
  
  # Plot the fit result:
  ax1.scatter(feature, pred_updrs)

  ax1.set_xlabel(feature_name)
  ax1.set_ylabel('total_UPDRS')
  ax1.set_title(feature_name + ' vs total_UPDRS (training set)')

  plt.show(block=False)
  print(feature_name)
  
  if want_save_figs:
    plt.savefig(fig_folder + '/1D_loop_fig_' + feature_name + '_train.pdf', format='pdf')
  
  
  # Validating:
  
  # Computing prediction for the validation set:
  
  feature_validt = validation_data[feature_name].to_numpy()
  
  updrs_validt = updrs_validt.reshape(-1,1)
  feature_validt = feature_validt.reshape(-1,1)
  
  pred_updrs_validt = regObj.predict(feature_validt)
  
  # Computing validation error:
  loss_validation = np.sum(np.square(pred_updrs_validt - updrs_validt))
  diff_valid = pred_updrs_validt - updrs_validt
  #print(diff_valid[0:5])
  #print(pred_updrs_validt.shape)
  #print(updrs_validt.shape)
  
  # Plotting the walidation result:
  fig2 = plt.figure(num=figN)
  figN = figN + 1
  ax2 = fig2.add_subplot()
  ax2.scatter(feature_validt, updrs_validt)
  ax2.scatter(feature_validt, pred_updrs_validt)
  
  ax2.set_title(feature_name + ' vs total_UPDRS (validation set)')
  ax2.set_xlabel(feature_name)
  ax2.set_ylabel('total_UPDRS')
  
  if want_save_figs:
    plt.savefig(fig_folder + '/1D_loop_fig_' + feature_name + '_val.pdf', format='pdf')
  
  

  # Collecting the results:
  i = result_df.shape[0]
  result_df.loc[i,'Feature'] = feature_name
  result_df.loc[i,'Training error'] = loss_training
  result_df.loc[i,'Validation error'] = loss_validation
  result_df.sort_values('Validation error')
    
    
# Print results:
result_df = result_df.sort_values('Validation error')
print('results:')
print(result_df)
    
# Saving the results:    
if want_to_save:
  print('Saving')
  result_df.to_csv(save_to, index=False)
  

plt.show(block=True)