def calculate_accuracy(
    true_labels,
    predictions,
):

    nb_TP = sum(predictions == true_labels)
    accuracy = nb_TP / len(predictions)

    print("(calculate_accuracy) task info:")
    print("(calculate_accuracy) - number of classes:", true_labels.nunique())
    print("\n(calculate_accuracy) Metrics:")
    print("(calculate_accuracy) - accuracy:", accuracy)
    return accuracy
