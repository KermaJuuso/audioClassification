import csv
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import precision_score, recall_score
from CNNclassifier import CNNClassifier
from sklearn.model_selection import train_test_split


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

    # Validatio split
    train_ft, val_ft, train_lb, val_lb = train_test_split(
        train_ft, train_lb, test_size=0.2, random_state=42, stratify=train_lb
    )

    # Define classifiers
    single_classifiers = [("Random Forest", RandomForestClassifier()), ("AdaBoost", AdaBoostClassifier()),
                          ("Bagging", BaggingClassifier()), ("Extra Trees", ExtraTreesClassifier())]

    classifiers = single_classifiers + [("Stacking", StackingClassifier(single_classifiers)),
                                        ("Voting", VotingClassifier(single_classifiers))]

    # Collect some statistics about the performance of the classifiers
    stats = {name: {'accuracies': [], 'precisions': [], 'recalls': [], 'train_times': [], 'test_times': []} for name, _
             in classifiers}

    # Train and evaluate each classifier
    for name, classifier in classifiers:
        start = time.time()
        classifier.fit(train_ft, train_lb)
        train_time = time.time() - start
        stats[name]['train_times'].append(train_time)

        start = time.time()
        preds = classifier.predict(val_ft)
        test_time = time.time() - start

        stats[name]['accuracies'].append(classifier.score(val_ft, val_lb))
        stats[name]['precisions'].append(precision_score(val_lb, preds))
        stats[name]['recalls'].append(recall_score(val_lb, preds))
        stats[name]['test_times'].append(test_time)
    print()

    print("\nValidation results:\n")
    for name in stats:
        print(f"{name}:")
        print(f"  accuracy:  {np.average(stats[name]['accuracies']) * 100:.1f} %")
        print(f"  precision: {np.average(stats[name]['precisions']) * 100:.1f} %")
        print(f"  recall:    {np.average(stats[name]['recalls']) * 100:.1f} %")
        print()

    # Print results
    print("Final evaluation on test set:\n")

    for name, classifier in classifiers:
        classifier.fit(train_ft, train_lb)  # train once on full training data
        preds = classifier.predict(test_ft)

        print(f"{name}:")
        print(f"  accuracy:  {classifier.score(test_ft, test_lb) * 100:.1f} %")
        print(f"  precision: {precision_score(test_lb, preds) * 100:.1f} %")
        print(f"  recall:    {recall_score(test_lb, preds) * 100:.1f} %")
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
