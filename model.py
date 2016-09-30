import pickle
import os
import argparse

from sklearn.feature_extraction.text import *
from sklearn.svm import SVC
from sklearn.externals import joblib
from stop_words import get_stop_words

titles = ['Россия', 'Мир', 'Бывший СССР', 'Финансы',
          'Бизнес', 'Силовые структуры', 'Наука и техника', 'Культура',
          'Спорт', 'Интернет и СМИ', 'Ценности', 'Путешествия', 'Из жизни'
          ]


def train(args):
    classifier_path = os.path.join(args.out_folder, 'classifier.clf')
    vectorizer_path = os.path.join(args.out_folder, 'vectorizer.clf')

    with open(args.data, 'rb') as f:
        data, labels = pickle.load(f)

    if data:
        count = len(data)
        train_data, train_labels = data[:int(count * .75)], labels[:int(count * .75)]
        test_data, test_labels = data[int(count * .75):], labels[int(count * .75):]

        print('Prepared train data with length: {0}, and test data with length: {1}'.format(len(train_data),
                                                                                            len(test_data)))

        vectorizer = TfidfVectorizer(max_features=1000, stop_words=get_stop_words('ru'))
        mat = vectorizer.fit_transform(train_data)
        mat = mat.todense()
        clf = SVC(kernel='linear')
        clf.fit(mat, train_labels)

        score = clf.score(vectorizer.transform(test_data).todense(), test_labels)

        print('Training done! Precision: {0}'.format(score))

        if not os.path.exists(args.out_folder):
            os.makedirs(args.out_folder)
        else:
            for file in os.listdir(args.out_folder):
                file_path = os.path.join(args.out_folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        joblib.dump(clf, classifier_path)
        joblib.dump(vectorizer, vectorizer_path)


def test(args):
    classifier_path = os.path.join(args.model, 'classifier.clf')
    vectorizer_path = os.path.join(args.model, 'vectorizer.clf')

    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)

    with open(args.file, 'r') as f:
        data = f.read()

    if data:
        mat = vectorizer.transform([data, ]).todense()
        category = classifier.predict(mat)[0]

        print('Decided category: {0}'.format(titles[category]))
    else:
        print('Error during processing file!')


parser = argparse.ArgumentParser(description='Creates model for text corpus.')
subparsers = parser.add_subparsers(help='Run modes', dest='mode')

# Train command
train_parser = subparsers.add_parser('train', help='Trains model to recognize text theme')
train_parser.add_argument('--data', help='File with training data', default='data.dat', type=str)
train_parser.add_argument('--out_folder', help='Folder, to store trained model', default='model', type=str)
train_parser.set_defaults(func=train)

# Test command
test_parser = subparsers.add_parser('test', help='Tests trained model')
test_parser.add_argument('--model', help='Trained model folder', default='model', type=str)
test_parser.add_argument('--file', help='File with data to produce', required=True)
test_parser.set_defaults(func=test)


arguments = parser.parse_args()
if arguments.mode:
    arguments.func(arguments)
