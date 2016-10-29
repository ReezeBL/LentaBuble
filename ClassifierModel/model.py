import argparse
import logging
import os
import pickle

import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from stop_words import get_stop_words

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

titles = ['Россия', 'Мир', 'Бывший СССР', 'Финансы',
          'Бизнес', 'Силовые структуры', 'Наука и техника', 'Культура',
          'Спорт', 'Интернет и СМИ', 'Ценности', 'Путешествия', 'Из жизни'
          ]


def train(args):
    classifier_path = os.path.join(args.out_folder, 'classifier.clf')

    df = pd.read_csv(args.data, encoding='utf-8')
    df = df.dropna()

    if not df.empty:
        train_data, test_data, train_labels, test_labels = train_test_split(df['Data'], df['Labels'], test_size=.2,
                                                                            random_state=0)
        print('Prepared train data with length: {0}, and test data with length: {1}'.format(len(train_data),
                                                                                            len(test_data)))
        clf = Pipeline([('vect', TfidfVectorizer(stop_words=get_stop_words('ru'), smooth_idf=True, min_df=.003,
                                                 ngram_range=(1, 3))),
                        ('svm', SVC(kernel='linear'))])

        clf.fit(train_data, train_labels)
        predicted = clf.predict(test_data)

        print('Training done! Report:')
        print(metrics.classification_report(test_labels, predicted, target_names=titles))

        if not os.path.exists(args.out_folder):
            os.makedirs(args.out_folder)
        else:
            for file in os.listdir(args.out_folder):
                file_path = os.path.join(args.out_folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        joblib.dump(clf, classifier_path)


def traings(args):
    print('Loading data..')
    df = pd.read_csv(args.data, encoding='utf-8')
    df = df.dropna()

    if not df.empty:
        train_data, test_data, train_labels, test_labels = train_test_split(df['Data'], df['Labels'], test_size=.2,
                                                                            random_state=0)
        clf = Pipeline([('vect', TfidfVectorizer(stop_words=get_stop_words('ru'), smooth_idf=True, max_features=3000)),
                        ('svm', SVC(kernel='linear'))])

        parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
                      }

        gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
        print('Running GridSearch..')
        gs_clf.fit(train_data[:3000], train_labels[:3000])
        print('Training done!\r\nBest score: {0}'.format(gs_clf.best_score_))
        for param_name in sorted(parameters.keys()):
            print('{0}: {1}'.format(param_name, gs_clf.best_params_[param_name]))


def test(args):
    classifier_path = os.path.join(args.model, 'classifier.clf')
    classifier = joblib.load(classifier_path)

    with open(args.file, 'r') as f:
        data = f.read()

    if data:
        category = classifier.predict([data, ])[0]

        print('Decided category: {0}'.format(titles[category]))
    else:
        print('Error during processing file!')


def view(args):
    with open(args.file, 'rb') as f:
        data, _ = pickle.load(f)
    if data:
        c, length = 0, len(data)
        print('Loaded {0} arcticles\r\nPrint article number to view, -1 to leave'.format(length))
        while c != -1:
            c = int(input())
            if c < length:
                print(data[c])
            else:
                print('Invalid argument')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates model for text corpus.')
    subparsers = parser.add_subparsers(help='Run modes', dest='mode')

    # Train command
    train_parser = subparsers.add_parser('train', help='Trains model to recognize text theme')
    train_parser.add_argument('--data', help='File with training data', default='data.csv', type=str)
    train_parser.add_argument('--out_folder', help='Folder, to store trained model', default='model', type=str)
    train_parser.set_defaults(func=train)

    # Train with GridSearch
    train_parser = subparsers.add_parser('traings', help='Trains model to recognize text theme,'
                                                         ' using GridSearch do determine better parameters')
    train_parser.add_argument('--data', help='File with training data', default='data.csv', type=str)
    train_parser.set_defaults(func=traings)

    # Test command
    test_parser = subparsers.add_parser('test', help='Tests trained model')
    test_parser.add_argument('--model', help='Trained model folder', default='model', type=str)
    test_parser.add_argument('--file', help='File with data to produce', required=True)
    test_parser.set_defaults(func=test)

    # View parser
    view_parser = subparsers.add_parser('view', help='View data')
    view_parser.add_argument('--file', help='Data file to view')
    view_parser.set_defaults(func=view)

    arguments = parser.parse_args()
    if arguments.mode:
        arguments.func(arguments)



