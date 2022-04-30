import numpy as np
from scipy.stats import mode as smode


def mode(lst, direction="row"):
    # Keep track of frequency
    freq = {}

    for i in lst:
        if direction == "column":
            i = i[0]
        freq.setdefault(i, 0)
        freq[i] += 1

    # Find most frequent
    hf = max(freq.values())

    hflst = []

    for i, j in freq.items():
        if j == hf:
            hflst.append(i)

    return [hf]


class RandomForest:
    def __init__(
        self,
        x,
        Y,
        n_trees,
        n_features,
        sample_size=None,
        depth=10,
        random_state=None,
    ):
        if sample_size is None:
            sample_size = x.shape[0]

        if random_state is not None:
            np.random.seed(random_state)

        self.n_features = n_features
        self.x = x
        self.Y = Y
        self.sample_size = sample_size
        self.depth = depth
        self.trees = []
        for i in range(n_trees):
            print(f"Generating tree {i+1}")
            self.trees.append(self.generate_tree())

    def generate_tree(self):
        # Choose sample_size subset number of rows
        idxs = np.random.permutation(self.x.shape[0])[: self.sample_size]
        # Choose n_features subset number of features
        f_idxs = np.random.permutation(self.x.shape[1] - 1)[: self.n_features]
        f_idxs = list(f_idxs)
        f_idxs.append(self.x.shape[1] - 1)
        f_idxs = np.array(f_idxs)

        # Fit the Decision tree with these subset of rows and features
        clf = DecisionTree(self.depth)
        clf.fit(self.x.iloc[idxs, f_idxs], self.Y)
        return clf

    def predict(self, x):
        trees_pred = [t.predict(x) for t in self.trees]
        return smode(trees_pred, axis=0)[0][0]


class DecisionTree:
    def __init__(self, max_depth=5, depth=1):
        self.max_depth = max_depth
        self.depth = depth
        self.left = None  # left tree
        self.right = None  # right tree

    def fit(self, data, target):
        self.data = data
        self.target = target
        self.columns = self.data.columns.tolist()
        self.columns.remove(target)
        if self.depth <= self.max_depth:
            self.gini_index_score = self._calculate_gini_index(self.data[self.target])

            # Find the best split using Information Gain using Gini Index
            (
                self.criteria,
                self.split_feature,
                self.information_gain,
            ) = self._find_best_split()

            if self.criteria is not None and self.information_gain > 0:
                self._create_branches()

    def _create_branches(self):
        self.left = DecisionTree(max_depth=self.max_depth, depth=self.depth + 1)
        self.right = DecisionTree(max_depth=self.max_depth, depth=self.depth + 1)
        left_rows = self.data[self.data[self.split_feature] <= self.criteria]
        right_rows = self.data[self.data[self.split_feature] > self.criteria]
        # Recursively fit the left and right branches
        self.left.fit(data=left_rows, target=self.target)
        self.right.fit(data=right_rows, target=self.target)

    def _calculate_gini_index(self, data):
        if data is None or data.empty:
            return 0
        classes = np.unique(data)
        n_vals = len(data)
        gini_index = 1.0
        for cls in classes:
            gini_index -= np.square(np.sum(data == cls) / n_vals)
        return gini_index

    def _find_best_split(self):
        best_split = {}
        for col in self.columns:
            information_gain, split = self._find_best_split_for_feature(col)
            if split is None:
                continue
            if not best_split or best_split["information_gain"] < information_gain:
                best_split = {
                    "split": split,
                    "col": col,
                    "information_gain": information_gain,
                }

        return (
            best_split.get("split"),
            best_split.get("col"),
            best_split.get("information_gain"),
        )

    def _find_best_split_for_feature(self, col):
        x = self.data[col]
        unique_values = x.unique()
        if len(unique_values) == 1:
            return None, None
        information_gain = None
        split = None
        for val in unique_values:
            left = x <= val
            right = x > val
            left_data = self.data[left]
            right_data = self.data[right]
            left_gini_index = self._calculate_gini_index(left_data[self.target])
            right_gini_index = self._calculate_gini_index(right_data[self.target])
            score = self._calculate_information_gain(
                left_count=len(left_data),
                left_gini_index=left_gini_index,
                right_count=len(right_data),
                right_gini_index=right_gini_index,
            )
            if information_gain is None or score > information_gain:
                information_gain = score
                split = val
        return information_gain, split

    def _calculate_information_gain(
        self, left_count, left_gini_index, right_count, right_gini_index
    ):
        left_prob = left_count / len(self.data)
        right_prob = right_count / len(self.data)
        return self.gini_index_score - (
            left_prob * left_gini_index + right_prob * right_gini_index
        )

    def predict(self, data):
        return np.array(
            [self._traverse_tree(row) for _, row in data.iterrows()],
            dtype=object,
        )

    def _traverse_tree(self, row):
        if self.is_leaf_node():
            return self.probability()
        tree = self.left if row[self.split_feature] <= self.criteria else self.right
        return tree._traverse_tree(row)

    def is_leaf_node(self):
        return self.left is None and self.right is None

    def probability(self):
        return mode(self.data[self.target])


if __name__ == "__main__":
    import pandas as pd
    import pickle
    import os

    print("Loading dataset...")

    # Check if fer2013.csv file exists
    if not os.path.exists("fer2013.csv"):
        print("fer2013.csv file not found. Ensure the file is in the same directory. Download from: https://www.kaggle.com/datasets/deadskull7/fer2013")
        exit()

    dataset = pd.read_csv("fer2013.csv")
    print(dataset)
    print("Loaded dataset")
    expression_classes = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    ]

    # Image dimensions
    WIDTH = 48
    HEIGHT = 48

    # Filter some expressions
    filters = []
    filters = [expression_classes.index(i) for i in filters]
    if len(filters) > 0:
        dataset = dataset[dataset["emotion"].isin(filters)]

    # Split the dataset into training and test sets based on the Usage column
    training_set = dataset.loc[dataset["Usage"] == "Training"]
    test_set = dataset.loc[dataset["Usage"] == "PublicTest"]

    # Prepare training dataset
    X_Train = []
    Y_Train = []
    train_datapoints = training_set["pixels"].tolist()
    for xseq in train_datapoints:
        xx = [int(i) for i in xseq.split(" ")]
        xx = np.asarray(xx).reshape(WIDTH, HEIGHT)
        X_Train.append(xx.astype("float32"))
    X_Train = np.asarray(X_Train)
    X_Train = np.expand_dims(X_Train, -1)
    Y_Train = np.asarray(training_set["emotion"])

    # Prepare testing dataset
    X_Test = []
    Y_Test = []
    test_datapoints = test_set["pixels"].tolist()
    for xseq in test_datapoints:
        xx = [int(i) for i in xseq.split(" ")]
        xx = np.asarray(xx).reshape(WIDTH, HEIGHT)
        X_Test.append(xx.astype("float32"))
    X_Test = np.asarray(X_Test)
    X_Test = np.expand_dims(X_Test, -1)
    Y_Test = np.asarray(test_set["emotion"])
    X_Train = X_Train / 255.0
    X_Test = X_Test / 255.0
    # Reshape the X_Train from 4D to a 2D array
    nsamples, nx, ny, nrgb = X_Train.shape
    X_Train2 = X_Train.reshape((nsamples, nx * ny * nrgb))

    # Reshape the X_Test from 4D to a 2D array
    nsamples, nx, ny, nrgb = X_Test.shape
    X_Test2 = X_Test.reshape((nsamples, nx * ny * nrgb))

    # Convert X_Train2 to a DataFrame
    train_dataframe = pd.DataFrame(X_Train2)
    # Append the Y_Train to the DataFrame
    train_dataframe["expression"] = Y_Train
    test_dataframe = pd.DataFrame(X_Test2)

    print("Starting to train...")

    n_features = int(np.sqrt(train_dataframe.shape[1]))
    sample_size = int(train_dataframe.shape[0] * 2 / 3)

    clf = RandomForest(
        x=train_dataframe,
        Y="expression",
        n_trees=5,
        n_features=n_features,
        sample_size=sample_size,
        depth=5,
        random_state=42,
    )
    print("Finishing training")

    pred = clf.predict(test_dataframe)

    total = len(pred)
    score = 0.0
    Y_Test = Y_Test.T
    for i in range(total):
        if pred[i] == Y_Test[i]:
            score += 1.0
    score /= total

    print("Score: ", score)

    model_file = open("random_forest_model.pkl", "wb")
    pickle.dump(clf, model_file)
    model_file.close()
    print("Model saved to random_forest_model.pkl")
