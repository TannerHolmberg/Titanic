import pandas as pd
import numpy as np
from graphviz import Digraph


# ----------- PREPROCESSING -----------

def simplify_titles(title):
    if title in ['Mr']:
        return 'Mr'
    elif title in ['Mrs']:
        return 'Mrs'
    elif title in ['Miss']:
        return 'Miss'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Dr', 'Rev', 'Col', 'Major']:
        return 'Professional'
    else:
        return 'Rare'


def preProcess(data):
    # Extract + simplify title
    data["Title"] = data["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    data["Title"] = data["Title"].apply(simplify_titles)

    data.drop("Name", axis=1, inplace=True)

    # Fill missing values
    data["Age"].fillna(data["Age"].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data["Fare"].fillna(data["Fare"].median(), inplace=True)

    # Add new feature: FamilySize
    data['FamilySize'] = data['SibSp'] + data['Parch']

    def simplify_family(size):
        if size == 0:
            return 'Alone'
        elif size <= 2:
            return 'Small'
        elif size <= 5:
            return 'Medium'
        else:
            return 'Large'

    data['FamilySize'] = data['FamilySize'].apply(simplify_family)

    # Drop old columns AFTER creating FamilySize
    data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

    # Drop unused features
    data.drop("Cabin", axis=1, inplace=True)
    data.drop("Ticket", axis=1, inplace=True)

    # Bin continuous variables
    data['Fare'] = pd.cut(data['Fare'], bins=3)
    data['Age'] = pd.cut(data['Age'], bins=3)


# ----------- ENTROPY -----------

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])


def feature_entropy(df, feature, target='Survived'):
    total_len = len(df)
    expected_entropy = 0
    values = df[feature].unique()

    for val in values:
        subset = df[df[feature] == val][target].to_numpy()
        subset_entropy = entropy(subset)
        expected_entropy += (len(subset) / total_len) * subset_entropy

    return expected_entropy


def information_gain(df, feature, target='Survived'):
    return entropy(df[target].values) - feature_entropy(df, feature, target)


# ----------- DECISION TREE -----------

def build_tree(df, features, target='Survived', depth=0, max_depth=5, min_samples=5):
    # Stop if pure
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # Stop if no features left
    if len(features) == 0:
        return df[target].mode()[0]

    # Stop if too deep
    if depth == max_depth:
        return df[target].mode()[0]

    # Stop if too few samples
    if len(df) < min_samples:
        return df[target].mode()[0]

    # Choose best feature
    gains = [information_gain(df, f, target) for f in features]
    best_feature = features[np.argmax(gains)]

    tree = {best_feature: {}}

    for val in df[best_feature].unique():
        subset = df[df[best_feature] == val]

        if subset.empty:
            tree[best_feature][val] = df[target].mode()[0]
        else:
            remaining_features = [f for f in features if f != best_feature]
            tree[best_feature][val] = build_tree(
                subset,
                remaining_features,
                target,
                depth + 1,
                max_depth,
                min_samples
            )

    return tree


# ----------- VISUALIZATION -----------

def visualize_tree(tree):
    dot = Digraph()

    def add_nodes(tree, parent=None, edge_label=""):
        node_id = str(id(tree))

        if not isinstance(tree, dict):
            color = "lightblue" if tree == 1 else "lightcoral"
            dot.node(node_id, label=f"Class: {tree}", style="filled", fillcolor=color)
            if parent:
                dot.edge(parent, node_id, label=edge_label)
            return

        feature = list(tree.keys())[0]
        dot.node(node_id, label=feature)

        if parent:
            dot.edge(parent, node_id, label=edge_label)

        for value, subtree in tree[feature].items():
            add_nodes(subtree, node_id, str(value))

    add_nodes(tree)
    return dot


# ----------- PREDICTION -----------


def predict(tree, sample, default=0):
    # If it's a leaf node → return prediction
    if not isinstance(tree, dict):
        return tree

    feature = list(tree.keys())[0]
    value = sample[feature]

    # If value exists in tree branch → follow it
    if value in tree[feature]:
        return predict(tree[feature][value], sample)
    else:
        # fallback (important!)
        return default  # default prediction (can improve later)


# ----------- MAIN -----------

def main():
    data = pd.read_csv("train.csv")
    preProcess(data)

    features = ['Sex', 'Pclass', 'Title', 'Embarked', 'Age', 'Fare', 'FamilySize']

    print("\nInformation Gain:")
    for feature in features:
        print(f"{feature}: {information_gain(data, feature):.4f}")

    tree = build_tree(data, features, max_depth=3, min_samples=50)

    dot = visualize_tree(tree)
    dot.render("titanic_tree", format="png", view=True)

    # Load test data
    test_data = pd.read_csv("test.csv")
    preProcess(test_data)

    # Predict
    default = data["Survived"].mode()[0]
    predictions = []
    for _, row in test_data.iterrows():
        pred = predict(tree, row, default)
        predictions.append(pred)

    # Save
    output = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

    output.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()