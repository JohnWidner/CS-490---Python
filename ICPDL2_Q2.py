"""
JOhn Widner
ICP Deep Learning 2

"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'C:/Users/John/Desktop/Monty/DeepLearning_Lesson2/data/Smoking.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
data = np.delete(data, [3], axis=1)  # Remove the portions of data not used
data = data.astype(float)  # Cast data as a float
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X1, X2, and Y
X1 = tf.placeholder(tf.float32, name='X1')  # Smoking status
X2 = tf.placeholder(tf.float32, name='X2')  # Age classification
Y = tf.placeholder(tf.float32, name='Y')  # Death Status

# Step 3: create weight and bias, initialized to 0
w1 = tf.Variable(0.0, name='weight1')
w2 = tf.Variable(0.0, name='weight2')
b = tf.Variable(0.0, name='bias')


# Step 4: build model to predict Price
Y_predicted = (X1 * w1) + (X2 * w2) + b


# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')


# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)

    # Step 8: train the model
    for i in range(5):  # train the model 50 epochs
        total_loss_1 = 0
        total_loss_2 = 0
        for x1, x2, y in data:
            # Session runs train_op and fetch values of loss
            _, l_1 = sess.run([optimizer, loss], feed_dict={X1: x1, Y: y})
            _, l_2 = sess.run([optimizer, loss], feed_dict={X2: x2, Y: y})
            total_loss_1 += l_1
            total_loss_2 += l_2
        print('Epoch {0}: {1}'.format(i, total_loss_1 / n_samples))
        print('Epoch {0}: {1}'.format(i, total_loss_2 / n_samples))

    # close the writer when you're done using it
    writer.close()

    # Step 9: output the values of w and b
    w1, w2, b = sess.run([w1, w2, b])

# plot the results
X1, X2, Y = data.T[0], data.T[1], data.T[2]

plt.figure()
plt.plot(X1, Y, 'bo', label='Real data')
plt.plot(X1, X1 * w1 + b, 'r', label='Predicted data')
plt.legend()

plt.figure()
plt.plot(X2, Y, 'bo', label='Real data')
plt.plot(X2, X2 * w2 + b, 'r', label='Predicted data')
plt.legend()

plt.show()
