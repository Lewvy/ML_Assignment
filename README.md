Initialize the ImageProcessor with a target size for image resizing.
        
        Parameters:
        target_size (tuple): The desired dimensions (width, height) to resize images.

Load an image, convert it to grayscale, resize it, normalize pixel values,
        and flatten it into a one-dimensional array.
        
        Parameters:
        image_path (str): The file path of the image.
        
        Returns:
        np.array: Flattened, normalized grayscale image.

Load a dataset from a directory structure.
        The function assumes that stress images are in 'base_path' and non-stress
        images are in a sibling directory named 'NoStress'.
        
        Parameters:
        base_path (str): The directory containing stress images.
        
        Returns:
        tuple: (features, labels) where features is a NumPy array of processed images
               and labels is a NumPy array of corresponding class labels (1 for stress, 0 for no-stress).

Initialize the logistic regression model.
        
        Parameters:
        - learning_rate: Step size for gradient descent.
        - num_iterations: Number of iterations for training.

Compute the sigmoid function, which maps input values to a probability range [0,1].
        
        Parameters:
        - z: The input value(s), which is typically a linear combination of weights and input features.
        
        Returns:
        - Sigmoid-transformed values.

Train the logistic regression model using mini-batch gradient descent.
        
        Parameters:
        - X: Training data (features), shape (n_samples, n_features).
        - y: Training labels (binary: 0 or 1), shape (n_samples,).
        - batch_size: Number of samples per training batch (for mini-batch gradient descent).

Predict binary class labels (0 or 1) for input data.
        
        Parameters:
        - X: Input data (features), shape (n_samples, n_features).
        
        Returns:
        - Binary predictions (0 or 1), shape (n_samples,).

Initialize the KNN classifier.

Store the training data.

Predict class labels using KNN.

Initialize the logistic regression model.

Compute the sigmoid function.

Train the logistic regression model using mini-batch gradient descent.

Predict binary class labels (0 or 1) for input data.

Initialize the KNN classifier.

Store the training data.

Predict class labels using KNN.

A Naive Bayes classifier that assumes features are normally distributed (Gaussian Naive Bayes).
    It includes smoothing to handle zero probabilities and a minimum variance to prevent division by zero.

Initializes the NaiveBayesClassifier.

        Args:
            smoothing (float): A small value added to class priors to prevent zero probabilities.
            min_variance (float): A minimum variance value to prevent division by zero in the Gaussian probability calculation.

Calculates the natural logarithm of x, ensuring it's never zero by adding smoothing.

        Args:
            x (float or np.ndarray): The value(s) to take the logarithm of.

        Returns:
            float or np.ndarray: The logarithm of x, with smoothing applied.

Calculates the log probability of x given a Gaussian distribution with the specified mean and variance.

        Args:
            x (float or np.ndarray): The value(s) to calculate the probability for.
            mean (float or np.ndarray): The mean of the Gaussian distribution.
            var (float or np.ndarray): The variance of the Gaussian distribution.

        Returns:
            float or np.ndarray: The log probability of x.

Fits the Naive Bayes classifier to the training data.

        Args:
            X (np.ndarray): The training features.
            y (np.ndarray): The training labels.

Predicts the probability of each sample belonging to each class.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: The probability matrix, where each row represents a sample and each column represents a class.

Predicts the class label for each sample.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: The predicted class labels.

Calculates the accuracy of the classifier on the given data.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The true labels.

        Returns:
            float: The accuracy of the classifier.

Trains and evaluates a RobustNaiveBayesClassifier.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.

    Returns:
        tuple: A tuple containing the trained classifier and the predicted probabilities on the test set.

A Decision Tree Classifier for binary or multi-class classification.

    Args:
        max_depth (int): The maximum depth of the tree.  Limits how many levels the tree can grow to prevent overfitting. Default is 5.
        min_samples_split (int): The minimum number of samples required to split an internal node.  Prevents splitting nodes with too few samples. Default is 2.

Initializes the DecisionTreeClassifier.

Represents a node in the decision tree.

Initializes a Node.

            Args:
                feature (int): The index of the feature used for splitting at this node.  None for leaf nodes.
                threshold (float): The threshold value used for splitting at this node. None for leaf nodes.
                left (Node): The left child node (for values <= threshold). None for leaf nodes.
                right (Node): The right child node (for values > threshold). None for leaf nodes.
                value (int): The predicted class value for this node (for leaf nodes). None for internal nodes.

Calculates the entropy of a set of labels.  Entropy measures the impurity or disorder in the labels.

        Args:
            y (np.ndarray): The array of labels.

        Returns:
            float: The entropy of the labels.

Calculates the information gain from splitting a node. Information gain is the reduction in entropy.

        Args:
            parent (np.ndarray): The labels of the parent node.
            left_child (np.ndarray): The labels of the left child node.
            right_child (np.ndarray): The labels of the right child node.

        Returns:
            float: The information gain.

Finds the best feature and threshold to split the data. This is where the greedy search happens.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The labels.

        Returns:
            tuple: The best feature index and threshold value.  Returns (None, None) if no good split is found.

Recursively builds the decision tree.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The labels.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the (sub)tree.

Fits the decision tree to the training data.

        Args:
            X (np.ndarray): The training features.
            y (np.ndarray): The training labels.

Predicts the class label for a single sample.

        Args:
            x (np.ndarray): A single sample's features.
            node (Node): The current node being evaluated.

        Returns:
            int: The predicted class label.

Predicts the class labels for a set of samples.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: The predicted class labels.

Trains and evaluates Logistic Regression, KNN, and Decision Tree models,
    plots their ROC curves, and returns their performance metrics.

    Args:
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        learning_rate (float): Learning rate for Logistic Regression.
        num_iterations (int): Number of iterations for Logistic Regression.
        k (int): Number of neighbors for KNN.
        max_depth (int): Maximum depth for Decision Tree.

    Returns:
        tuple: A tuple containing the performance metrics for Logistic Regression, KNN, and Decision Tree.

Plots the ROC curves for Logistic Regression, KNN, and Decision Tree models.

    Args:
        logreg (LogisticRegression): Trained Logistic Regression model.
        knn (KNNClassifier): Trained KNN model.
        dtree (DecisionTreeClassifier): Trained Decision Tree model.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.

Plots a single ROC curve.

    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities.
        model_name (str): Name of the model.

Main function to train and evaluate the models and print their performance.

A Decision Tree Classifier with post-pruning to prevent overfitting.

    Args:
        max_depth (int): The maximum depth of the tree.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        pruning_threshold (float): The threshold for pruning. If pruning a node doesn't decrease accuracy by more than this threshold, the node is pruned.

Initializes the DecisionTreeClassifierPruned.

Represents a node in the decision tree.

Initializes a Node.

            Args:
                feature (int): The index of the feature used for splitting at this node.
                threshold (float): The threshold value used for splitting at this node.
                left (Node): The left child node.
                right (Node): The right child node.
                value (int): The predicted class value for this node (for leaf nodes).

Calculates the entropy of a set of labels.

        Args:
            y (np.ndarray): The array of labels.

        Returns:
            float: The entropy of the labels.

Calculates the information gain from splitting a node.

        Args:
            parent (np.ndarray): The labels of the parent node.
            left_child (np.ndarray): The labels of the left child node.
            right_child (np.ndarray): The labels of the right child node.

        Returns:
            float: The information gain.

Finds the best feature and threshold to split the data.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The labels.

        Returns:
            tuple: The best feature index and threshold value.

Recursively builds the decision tree.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The labels.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the (sub)tree.

Fits the decision tree to the training data and then prunes it.

        Args:
            X (np.ndarray): The training features.
            y (np.ndarray): The training labels.

Predicts the class label for a single sample.

        Args:
            x (np.ndarray): A single sample's features.
            node (Node): The current node being evaluated.

        Returns:
            int: The predicted class label.

Predicts the class labels for a set of samples.

        Args:
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: The predicted class labels.

Calculates the accuracy of the tree on the given data.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The true labels.

        Returns:
            float: The accuracy of the tree.

Performs post-pruning on the tree.  This is where the magic happens.  It recursively traverses the tree,
        attempting to prune each node and checking if the pruning improves accuracy.

        Args:
            X (np.ndarray): The validation features.
            y (np.ndarray): The validation labels.

Recursively prunes the tree.

            Args:
                node (Node): The current node being evaluated.

            Returns:
                Node: The pruned node (or the original node if it wasn't pruned).

Replaces a target node in the tree with a leaf node.

        Args:
            current_node (Node): The current node being evaluated.
            target_node (Node): The node to replace with a leaf.

        Returns:
            Node: The modified node (or a new leaf node if the target node was found).

