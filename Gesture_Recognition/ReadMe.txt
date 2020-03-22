Loading Data:

This project uses the Hand Gesture Recognition Database (citation below) available on Kaggle. It contains 20000 images with different hands and hand gestures. There is a total of 10 hand gestures of 10 different people presented in the dataset. There are 5 female subjects and 5 male subjects. The images were captured using the Leap Motion hand tracking device.

Overview:
Load images
Some validation
Preparing the images for training
Use of train_test_split


Creating Model:

To simplify the idea of the model being constructed here, we're going to use the concept of Linear Regression. By using linear regression, we can create a simple model and represent it using the equation y = ax + b.
a and b (slope and intercept, respectively) are the parameters that we're trying to find. By finding the best parameters, for any given value of x, we can predict y. This is the same idea here, but much more complex, with the use of Convolutional Neural Networks.

A Convolutional Neural Network (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other. The pre-processing required in a ConvNet is much lower as compared to other classification algorithms. While in primitive methods filters are hand-engineered, with enough training, ConvNets have the ability to learn these filters/characteristics.

Overview:
Import what the need
Creation of CNN
Compiling and training model
Saving model for later use


Testing Model:


Now that we have the model compiled and trained, we need to check if it's good. First, we run model.evaluate to test the accuracy. Then, we make predictions and plot the images as long with the predicted labels and true labels to check everything. With that, we can see how our algorithm is working.
Later, we produce a confusion matrix, which is a specific table layout that allows visualization of the performance of an algorithm.

Overview:
Evaluate model
Predictions
Plot images with predictions
Visualize model

Conclusion:

Based on the results presented in the previous section, we can conclude that our algorithm successfully classifies different hand gestures images with enough confidence (>95%) based on a Deep Learning model.
