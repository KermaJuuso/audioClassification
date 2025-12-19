import csv
import time
import numpy as np
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from CNNclassifier import CNNClassifier
import joblib

train_features = "train_features.csv"
valid_features = "validation_features.csv"
test_features = "test_features.csv"

# Separate the features from the labels
def read_features(feature_file, cls_coding):
    with open(feature_file, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        next(csvreader, None)  # skip header
        labels = []
        feats = []
        for row in csvreader:
            labels.append(cls_coding[row[1]])
            feats.append(row[2:])

    # Convert to numpy
    labels = np.array(labels)
    feats = np.array(feats, dtype=np.float32)

    return labels, feats


def classify(classes=("car", "tram")):
    # Load data
    cls_coding = {c: i for i, c in enumerate(classes)}
    train_lb, train_ft = read_features(train_features, cls_coding)
    valid_lb, valid_ft = read_features(valid_features, cls_coding)
    test_lb, test_ft = read_features(test_features, cls_coding)

    # Define classifiers
    single_classifiers = [("Random Forest", RandomForestClassifier()), ("KNeighbors", KNeighborsClassifier()),
                          ("SVM", SVC()), ("Naive Bayes", GaussianNB())]

    classifiers = single_classifiers + [("Stacking", StackingClassifier(single_classifiers)),
                                        ("Voting", VotingClassifier(single_classifiers))]

    best_classifiers = {name:c for name, c in classifiers}

    # Collect some statistics about the performance of the classifiers
    stats = {name: {'accuracy': 0, 'train_time': 0.} for name, _ in classifiers}

    # Train each classifier a couple of times to find the one best fitting to the validation data
    for i in range(10):
        print("Train-test-cycle", i + 1)
        for name, classifier in classifiers:
            # print("Training", name)
            start = time.time()
            classifier.fit(train_ft, train_lb)
            train = time.time()

            # print("Testing", name)
            acc = classifier.score(valid_ft, valid_lb)
            if acc > stats[name]['accuracy']:  # save best model
                best_classifiers[name] = deepcopy(classifier)
                stats[name]['accuracy'] = acc
                stats[name]['train_time'] = train - start
            # print("  finished...")
            # print()
    print()

    # Save Random Forest for demo
    joblib.dump(best_classifiers["Random Forest"], "random_forest_model.joblib")

    # Evaluate each classifier with the test data
    for name, classifier in best_classifiers.items():
        start = time.time()
        prediction = classifier.predict(test_ft)
        stats[name]['accuracy'] = accuracy_score(prediction, test_lb)
        stats[name]['precision'] = precision_score(prediction, test_lb)
        stats[name]['recall'] = recall_score(prediction, test_lb)
        stats[name]['test_time'] = time.time() - start

    # Print results
    for i, (name, _) in enumerate(classifiers):
        print(f'{name}:')
        print(f'  accuracy:    {np.average(stats[name]["accuracy"]) * 100:3.1f} %')
        print(f'  precision:   {np.average(stats[name]["precision"]) * 100:3.1f} %')
        print(f'  recall:      {np.average(stats[name]["recall"]) * 100:3.1f} %')
        print(f'  train time: {np.average(stats[name]["train_time"]) * 1000:4.1f} ms')
        print(f'  test time:  {np.average(stats[name]["test_time"]) * 1000:3.1f} ms')
        print()

    ## CNN classifier for comparison (requires some environment setup (check readme))
    CNNclassifier = CNNClassifier()
    start = time.time()
    l, a, p, r = CNNclassifier.eval()
    test_time = time.time() - start
    print("CNN for comparison:")
    print(f'  accuracy:    {a * 100:3.1f} %')
    print(f'  precision:   {p * 100:3.1f} %')
    print(f'  recall:      {r * 100:3.1f} %')
    print(f'  train time: 36.58 s')
    print(f'  test time:   {test_time * 1000:3.0f} ms')


if __name__ == "__main__":
    classify()
