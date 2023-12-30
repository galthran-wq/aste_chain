
def compute_metrics(all_examples, all_predictions, n_components=3):
    tp = 0
    fp = 0
    fn = 0
    for example_predictions, example in zip(all_predictions, all_examples):
        predictions = [
            prediction 
            if isinstance(prediction, list)
            else (prediction,)
            for prediction in example_predictions
        ]
        predictions = [
            tuple(triplet[:n_components])
            for triplet in example_predictions
        ]
        labels = [
            tuple(triplet[:n_components])
            for triplet in example
        ]
        current_tp = 0
        for prediction in predictions:
            if prediction in labels:
                current_tp += 1
            else:
                fp += 1
        tp += current_tp
        fn += len(labels) - current_tp
    return {
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
    }


if __name__ == "__main__":
    assert compute_metrics(
        all_examples=[
            [("a",), ("b",), ("c",), ("d",)],
            [("a",), ("b",), ("c",)]
        ],
        all_predictions=[
            [("a",), ("b",), ("c",), ("d",)],
            [("a",), ("b",), ("c",), ("d",)]
        ],
        n_components=1
    ) == {'precision': 0.875, 'recall': 1.0}
