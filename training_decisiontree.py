import pandas
import sys


def run():
    dataset = pandas.read_csv('./training_set_backfront.csv')
    input = dataset.loc[:, dataset.columns.str.startswith('input')]
    output = dataset.loc[:, dataset.columns.str.startswith('output')]
    # split training set and validation set
    input_training_set = input[:int(len(input) * 0.8)]
    input_test_set = input[int(len(input) * 0.8):]

    output_training_set = output[:int(len(output) * 0.8)]
    output_test_set = output[int(len(output) * 0.8):]

    # train randomforest model
    from sklearn.tree import DecisionTreeClassifier

    decisiontree_classifier = DecisionTreeClassifier(
        # criterion='entropy',
        # max_depth=10,
        # min_samples_leaf=1,
        # min_samples_split=2,
        # random_state=0,
    )
    # train
    decisiontree_classifier.fit(input_training_set, output_training_set)
    # save model
    import pickle

    filename = 'decisiontree_classifier_model.sav'
    pickle.dump(decisiontree_classifier, open(filename, 'wb'))
    # load model
    # loaded_model = pickle.load(open(filename, 'rb'))
    # validation
    validation_set_prediction = decisiontree_classifier.predict(input_test_set)
    # save validation set prediction
    validation_set_prediction = pandas.DataFrame(validation_set_prediction)
    validation_set_prediction.to_csv('validation_set_prediction_with_decisiontree_classifier.csv')


if __name__ == "__main__":
    run()
