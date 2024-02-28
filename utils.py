import numpy as np

def gen_preds(model):
    return lambda x: model.predict_proba(x)[:, 1] # get predictions from MCBoost

# updated model algorithm using witness
def update_proba(yhat_proba, lambda_, wit_value):
  yhat_proba = yhat_proba + lambda_ * wit_value
  # set all <0 values to 0, all >1 values to 1
  yhat_proba[yhat_proba < 0] = 0
  yhat_proba[yhat_proba > 1] = 1
  yhat_pred = (yhat_proba > 0.5).astype('int')
  return yhat_proba, yhat_pred

# compute calibration error
def compute_calibration_error(wit_value, y_pred, yhat_proba):
  return np.abs((wit_value * (y_pred - yhat_proba)).mean())

# conditioned calibration error
def con_cal_err(bin, wit_value, y_pred, yhat_proba):
  '''
  bin is the number of slices
  wit_value is c*(x)
  y_pred is the predicted label (0 or 1)
  yhat_proba is the probability score (between 0 and 1)
  '''
  err = 0
  incre = 1/bin
  for i in range(bin):
    ind = np.where(np.logical_and(yhat_proba>i*incre, yhat_proba<=(i+1)*incre))
    temp_sum = compute_calibration_error(wit_value[ind], y_pred[ind], yhat_proba[ind])
    # print(temp_sum)
    if (not np.isnan(temp_sum)):
      frac = len(ind[0]) / len(yhat_proba)
      err += (temp_sum ** 2) * frac
  return err


