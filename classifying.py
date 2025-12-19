import csv
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score
from CNNclassifier import CNNClassifier

train_features = "train_features.csv"
test_features = "test_features_new.csv"

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
    test_lb, test_ft = read_features(test_features, cls_coding)

    # Define classifiers
    single_classifiers = [("Random Forest", RandomForestClassifier()), ("AdaBoost", AdaBoostClassifier()),
                          ("Bagging", BaggingClassifier()), ("Extra Trees", ExtraTreesClassifier())]

    classifiers = single_classifiers + [("Stacking", StackingClassifier(single_classifiers)),
                                        ("Voting", VotingClassifier(single_classifiers))]

    # Collect some statistics about the performance of the classifiers
    stats = {name: {'accuracies': [], 'precisions': [], 'recalls': [], 'train_times': [], 'test_times': []} for name, _
             in classifiers}

    # Train and evaluate each classifier a couple of times
    for i in range(100):
        print("Train-test-cycle", i + 1)
        for name, classifier in classifiers:
            # print("Training", name)
            start = time.time()
            classifier.fit(train_ft, train_lb)
            train = time.time()
            stats[name]['train_times'].append(train - start)

            # print("Testing", name)
            stats[name]['accuracies'].append(classifier.score(test_ft, test_lb))
            stats[name]['precisions'].append(precision_score(classifier.predict(test_ft), test_lb))
            stats[name]['recalls'].append(recall_score(classifier.predict(test_ft), test_lb))
            stats[name]['test_times'].append(time.time() - train)
            # print("  finished...")
            # print()
    print()

    # Print results
    for i, (name, _) in enumerate(classifiers):
        print(f'{name}:')
        print(f'  accuracy:    {np.average(stats[name]["accuracies"]) * 100:.1f} %')
        print(f'  precision:   {np.average(stats[name]["precisions"]) * 100:.1f} %')
        print(f'  recall:      {np.average(stats[name]["recalls"]) * 100:.1f} %')
        print(f'  train time: {np.average(stats[name]["train_times"]) * 1000:4.0f} ms')
        print(f'  test time:   {np.average(stats[name]["test_times"]) * 1000:3.0f} ms')
        print()
"""
    ## CNN classifier for comparison (requires some environment setup (check readme))
    CNNclassifier = CNNClassifier()
    start = time.time()
    l, a, p, r = CNNclassifier.eval()
    test_time = time.time() - start
    print("CNN for comparison:")
    print(f'  accuracy:    {a * 100:.1f} %')
    print(f'  precision:   {p * 100:.1f} %')
    print(f'  recall:      {r * 100:.1f} %')
    print(f'  train time: ~50 s')
    print(f'  test time:   {test_time * 1000:3.0f} ms')

"""
if __name__ == "__main__":
    classify()
