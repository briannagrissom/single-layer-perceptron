import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(30)

# Read in data
df = pd.read_csv('breast-cancer.csv')

# select the predictor and response data
# Uncomment to use full dataset for predictors

X = df[['area_mean', 'texture_worst', 'concave points_worst']]
y = df['diagnosis'].map({'B': -1, 'M': 1})

# standardize the data to reduce the condition number of the Hessian
X = (X - X.mean()) / X.std()
X['ones'] = [1 for _ in range(len(X))]
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)


# test single layer perceptron using a given weight
def test_single_layer_perceptron(data, labels, weight):
    num_records = data.shape[0]
    num_correct = 0
    # determine if the perceptron is able to predict the label correctly
    for idx in range(num_records):
        sample = data[idx, :]
        label = labels[idx]
        result = np.dot(sample, weight)  # Compute <w^T, x>
        if result > 0:
            prediction = 1
        else:
            prediction = -1
        if label == prediction:
            num_correct = num_correct + 1
    accuracy = num_correct / num_records
    return accuracy


# train single layer perceptron
def train_single_layer_perceptron(predictors, labels, test_predictors, test_labels, iterations):
    # initialize weight
    weights = [np.random.normal(loc=0, scale=0.1, size=predictors.shape[1])]
    num_records = predictors.shape[0]
    loss_vals = list()
    accuracies = list()
    alphas = list()
    # iterate K times
    for _ in range(iterations):
        error_terms = list()
        gradients = list()
        current_weight = weights[-1]
        num_correct = 0
        for idx in range(num_records):
            sample = predictors[idx, :]  # x_i
            label = labels[idx]  # y_i
            result = np.dot(sample, current_weight)  # <x_i, w(k)>
            if result > 0:  # if <x_i, w(k)> > 0
                prediction = 1
            else:  # if x^tw < 0
                prediction = -1
            if prediction == label:
                num_correct = num_correct + 1
            error = 0.5 * (label - result) ** 2  # compute f(x)
            gradient = (- (label - result)) * sample  # compute - gradient f(x)
            error_terms.append(error)
            gradients.append(gradient)

        # compute accuracy for iteration k
        accuracy = np.round(num_correct / num_records, 3)
        accuracies.append(accuracy)

        # average loss value
        loss = (1 / num_records) * sum(error_terms)
        loss_vals.append(loss)

        # updating the weight step
        Q_matrix = (1 / num_records) * sum([np.outer(predictors[i, :], predictors[i, :]) for i in range(num_records)])
        gradient_vector = (1 / num_records) * sum(gradients)
        alpha_k = np.dot(gradient_vector, gradient_vector) / (gradient_vector @ Q_matrix @ gradient_vector)
        alphas.append(alpha_k)
        next_weight = current_weight - alpha_k * gradient_vector
        weights.append(next_weight)

    # Compile results
    result_df = pd.DataFrame({'Iteration': list(range(1, iterations + 1)),
                              'alpha_k': alphas,
                              'Accuracy': accuracies,
                              'Loss': loss_vals})
    # Find best weight value based on the testing data
    test_accuracies = list()
    for we in weights:
        acc = test_single_layer_perceptron(test_predictors, test_labels, weight=we)
        test_accuracies.append(acc)
    best_acc = np.max(test_accuracies)
    best_iter = np.argmax(test_accuracies)
    optimal_weight = weights[best_iter]

    return result_df, optimal_weight, best_acc


result_table, best_weight, best_acc = train_single_layer_perceptron(predictors=X_train, labels=y_train,
                                                                    test_predictors=X_test,
                                                                    test_labels=y_test,
                                                                    iterations=50)
result_table.to_csv('training_results.csv')

# Separate benign and malignant data points
b_indices = list(np.where(df['diagnosis'] == 'B')[0])
m_indices = list(np.where(df['diagnosis'] == 'M')[0])
X_b, X_m = X.values[b_indices, :], X.values[m_indices, :]

# w_1x_1 + w_2x_2 + w_3x_3 = -b
w1, w2, w3, b = best_weight

# values for the hyperplane
x_1 = np.linspace(-3, 3, 15)
x_2 = np.linspace(-3, 3, 15)
X_1, X_2 = np.meshgrid(x_1, x_2)
X_3 = (-w1 * X_1 - w2 * X_2 - b) / w3

fig, axs = plt.subplots(1, 1, subplot_kw={'projection': '3d'}, figsize=(12, 10))

# Plot the hyperplane and the data points
axs.scatter(X_b[:, 0], X_b[:, 1], X_b[:, 2])
axs.scatter(X_m[:, 0], X_m[:, 1], X_m[:, 2])
axs.plot_surface(X_1, X_2, X_3, color='cyan', alpha=0.5, edgecolor='k')
plt.legend(['Benign', 'Malignant'])
axs.set_xlabel('area_mean')
axs.set_ylabel('texture_worst')
axs.set_zlabel('concave points_worst')
axs.set_title('Plot of w^Tx = 0 and data points')
axs.view_init(45, 45)

plt.show()

plt.plot(list(range(1, 51)), result_table['Loss'])
plt.plot(list(range(1, 51)), result_table['Accuracy'])
plt.legend(['Training loss', 'Training accuracy'])
plt.title('Training loss and training accuracy over 50 iterations')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.show()

