from sat import model_function
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
tf.logging.set_verbosity(tf.logging.INFO)

print("IMPORT FINISHED")
BATCH_SIZE=196
NUM_EPOCHS=1
training_set=pd.read_pickle("./data.pkl")
features=[]
Caption=[]
for i in range(len(training_set['Caption'])):
  features.append(training_set['Feature_vectors'][i])
  if training_set['Caption'][i].shape[0]==18:
    Caption.append(training_set['Caption'][i])
  else :
    Caption.append(training_set['Caption'][i][:-1])
x_train=np.asarray(features)
y_train=np.array(Caption)
x_train=x_train[:-104]
y_train=y_train[:-104]
print(y_train[0])
# print(type(y_train),y_train.shape,y_train.dtype)
get_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':x_train},
    y=y_train,
    batch_size=196,
    num_epochs=1,
    shuffle=False,
)


feature_cols = [tf.feature_column.numeric_column('x')]
run_config = tf.estimator.RunConfig(model_dir='./checkpoints',save_checkpoints_secs=3600)
params = {'learning_rate': 0.001,'feature_columns': feature_cols}
captioner=tf.estimator.Estimator(model_fn=model_function,params=params,config=run_config)
print("TRAINING STARTED")
# for i in range(NUM_EPOCHS):
  # captioner.train(input_fn=get_input_fn,steps=76)
# eval_result = captioner.evaluate(input_fn=get_input_fn)

# print('\nTrain set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
for i in range(196):
  predictions = captioner.predict(input_fn=get_input_fn)
  print(next(predictions))


