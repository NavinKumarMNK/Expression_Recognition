import os
import cv2
import numpy as np
import pandas as pd
import pickle
import zipfile
from scipy.stats import mode as smode


def mode(lst, direction="row"):
    # creating a dictionary
    freq = {}

    for i in lst:
        if direction == "column":
            i = i[0]
        freq.setdefault(i, 0)
        freq[i] += 1

    # Find most frequent
    hf = max(freq.values())

    # creating an empty list
    hflst = []

    # using for loop we are checking for most
    # repeated value
    for i, j in freq.items():
        if j == hf:
            hflst.append(i)

    # returning the result
    return [hf]


class RandomForest:
    def __init__(
        self,
        x,
        target,
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
        self.target = target
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
        clf.fit(self.x.iloc[idxs, f_idxs], self.target)
        return clf

    def predict(self, x):
        trees_pred = [t.predict(x) for t in self.trees]
        return smode(trees_pred, axis=0)[0][0]


class DecisionTree:
    def __init__(self, max_depth=5, depth=1):
        self.max_depth = max_depth
        self.depth = depth
        self.left = None  # left branch
        self.right = None  # right branch

    def fit(self, data, target):
        self.data = data
        self.target = target
        self.columns = self.data.columns.tolist()
        self.columns.remove(target)
        if self.depth <= self.max_depth:
            self.__validate_data()
            self.gini_index_score = self._calculate_gini_index(self.data[self.target])

            # Find the best split using Entropy and Info Gain
            (
                self.criteria,
                self.split_feature,
                self.information_gain,
            ) = self.__find_best_split()

            if self.criteria is not None and self.information_gain > 0:
                self.__create_branches()

    def __create_branches(self):
        self.left = DecisionTree(max_depth=self.max_depth, depth=self.depth + 1)
        self.right = DecisionTree(max_depth=self.max_depth, depth=self.depth + 1)
        left_rows = self.data[self.data[self.split_feature] <= self.criteria]
        right_rows = self.data[self.data[self.split_feature] > self.criteria]
        self.left.fit(data=left_rows, target=self.target)
        self.right.fit(data=right_rows, target=self.target)

    def _calculate_gini_index(self, data):
        if data is None or data.empty:
            return 0
        gini_index = 1.0
        classes = np.unique(data)
        n_vals = len(data)
        for cls in classes:
            gini_index -= np.square(np.sum(data == cls) / n_vals)
        return gini_index

    def __find_best_split(self):
        best_split = {}
        for col in self.columns:
            information_gain, split = self.__find_best_split_for_column(col)
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

    def __find_best_split_for_column(self, col):
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
            score = self.__calculate_information_gain(
                left_count=len(left_data),
                left_gini_index=left_gini_index,
                right_count=len(right_data),
                right_gini_index=right_gini_index,
            )
            if information_gain is None or score > information_gain:
                information_gain = score
                split = val
        return information_gain, split

    def __calculate_information_gain(
        self, left_count, left_gini_index, right_count, right_gini_index
    ):
        return self.gini_index_score - (
            (left_count / len(self.data)) * left_gini_index
            + (right_count / len(self.data)) * right_gini_index
        )

    def predict(self, data):
        return np.array(
            [self._pass_data_through_tree(row) for _, row in data.iterrows()],
            dtype=object,
        )

    def __validate_data(self):
        non_numeric_columns = (
            self.data[self.columns]
            .select_dtypes(include=["category", "object", "bool"])
            .columns.tolist()
        )
        if len(set(self.columns).intersection(set(non_numeric_columns))) != 0:
            raise RuntimeError("Not all columns are numeric")

    def _pass_data_through_tree(self, row):
        if self.is_leaf_node:
            return self.probability
        tree = self.left if row[self.split_feature] <= self.criteria else self.right
        return tree._pass_data_through_tree(row)

    @property
    def is_leaf_node(self):
        return self.left is None

    @property
    def probability(self):
        return mode(self.data[self.target])


raw_folder = "realdata/raw"
processed_folder = "realdata/processed"
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
raw_images = os.listdir(raw_folder)
print(f"Processing {len(raw_images)} images")

X_Test = []
original_images = []
# Read all images in raw_folder
for image in raw_images:
    # Read the image
    img = cv2.imread(raw_folder + "/" + image)
    # Resize the image to given dimentions
    resized = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    original_images.append(resized)
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Reshape image to correct size of model
    X_Test.append(gray.astype("float32"))

    cv2.imwrite(processed_folder + "/" + image, gray)

X_Test = np.asarray(X_Test)
X_Test = np.expand_dims(X_Test, -1)

# Normalize the images and reshape for the model
X_Test = X_Test / 255.0
nsamples, nx, ny, nrgb = X_Test.shape
X_Test = X_Test.reshape((nsamples, nx * ny * nrgb))
X_Test = pd.DataFrame(X_Test)


zip_filename = "random_forest_model.zip"
data = zipfile.ZipFile(zip_filename, "r")
data.extractall()

model_filename = "random_forest_model.pkl"
loaded_model = pickle.load(open(model_filename, "rb"))

predictions = loaded_model.predict(X_Test)
expression_classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

num_images = len(raw_images)
for i in range(num_images):
    image = cv2.cvtColor(original_images[i], cv2.COLOR_BGR2RGB)
    print(
        f"{expression_classes[predictions[i][0]]} (actual = {raw_images[i].split('-')[0]})"
    )
