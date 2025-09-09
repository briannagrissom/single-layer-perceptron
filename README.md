This project predicts tumors as either benign or malignant by using three numeric feature variables (mean area of lobes, worst mean texture, worst mean of concave points) taken from the tumor. The prediction method is a single-layer perceptron, where the optimal weights are found using the gradient descent algorithm on the mean squared error loss function. This is encapsulated in the Python file `single_layer_perceptron.py`.
This file: 
1) Reads in the data (https://www.kaggle.com/datasets/rahmasleam/breast-cancer)
2) Standardizes data to speed up algorithm convergence (by reducing the condition number of the Hessian matrix) and splits data into training and testing sets
3) Performs the iterative gradient descent algorithm to find optimal hyperplane weights that maximize testing accuracy
4) Plots the hyperplane in 3-D space with color-coded data points to visualize separation.

The file `Tumor_Classification.pdf` is the project report.
The file `Final_Presentation.pdf` is the project presentation.
The file `breast_cancer.csv` is the dataset from Kaggle.
