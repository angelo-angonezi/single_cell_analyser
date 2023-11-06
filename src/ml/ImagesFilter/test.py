from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import BinaryAccuracy


# loading the model back
print('loading model...')
model = load_model(save_path)

# testing the model
print('testing model performance on test data set...')
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

# getting test batches
test_batches = test.as_numpy_iterator()

# iterating over batches in test data set
for batch in test_batches:
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

# getting results
precision_result = precision.result()
recall_result = recall.result()
accuracy_result = accuracy.result()

# printing results
print('Precision: ', precision_result)
print('Recall: ', recall_result)
print('Accuracy: ', accuracy_result)

# printing execution message
print('done!')

# end of current module
