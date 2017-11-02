from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib.request
import tensorflow as tf
import numpy as np

# Data sets
IRIS_TRAINING = r"D:\tmp\iris\data\iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = r"D:\tmp\iris\data\iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, 'w') as f:
            f.write(raw.decode())

    if not os.path.exists(IRIS_TEST):
        raw = urllib.request.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, 'w') as f:
            f.write(raw.decode())
            # Load datasets.

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    feature_columns = [tf.feature_column.numeric_column('x',shape=[4])]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[10,20,10],n_classes=3,
                                            model_dir=r'D:\tmp\iris')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    classifier.train(train_input_fn)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(x = {'x':np.array(test_set.data)}, y=test_set.data,num_epochs=1,
                                                       shuffle=False)
    accuracy_score = classifier.evaluate(test_input_fn)['accuracy']

    print('\nTest accuracy: {0:f}\n'.format(accuracy_score))
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
        "New Samples, Class Predictions:    {}\n"
            .format(predicted_classes))


if __name__ == "__main__":
    main()