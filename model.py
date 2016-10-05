import pickle
import os
import argparse
import pandas as pd
import logging

from sklearn.feature_extraction.text import *
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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

    if not df.empty:
        train_data, test_data, train_labels, test_labels = train_test_split(df['Data'], df['Labels'], test_size=.2, random_state=0)
        print('Prepared train data with length: {0}, and test data with length: {1}'.format(len(train_data),
                                                                                            len(test_data)))
        clf = Pipeline([('vect', TfidfVectorizer(max_features=1000, stop_words=get_stop_words('ru'))),
                       ('svm', SVC(kernel='linear'))])

        clf.fit(train_data, train_labels)
        score = clf.score(test_data, test_labels)

        print('Training done! Precision: {0}'.format(score))

        if not os.path.exists(args.out_folder):
            os.makedirs(args.out_folder)
        else:
            for file in os.listdir(args.out_folder):
                file_path = os.path.join(args.out_folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        joblib.dump(clf, classifier_path)


def test(args):
    classifier_path = os.path.join(args.model, 'classifier.clf')
    classifier = joblib.load(classifier_path)

    with open(args.file, 'r') as f:
        data = f.read()

    if data:
        category = classifier.predict([data, ])

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


parser = argparse.ArgumentParser(description='Creates model for text corpus.')
subparsers = parser.add_subparsers(help='Run modes', dest='mode')

# Train command
train_parser = subparsers.add_parser('train', help='Trains model to recognize text theme')
train_parser.add_argument('--data', help='File with training data', default='data.csv', type=str)
train_parser.add_argument('--out_folder', help='Folder, to store trained model', default='model', type=str)
train_parser.set_defaults(func=train)

# Test command
test_parser = subparsers.add_parser('test', help='Tests trained model')
test_parser.add_argument('--model', help='Trained model folder', default='model', type=str)
test_parser.add_argument('--file', help='File with data to produce', required=True)
test_parser.set_defaults(func=test)

#View parser
view_parser = subparsers.add_parser('view', help='View data')
view_parser.add_argument('--file', help='Data file to view')
view_parser.set_defaults(func=view)

arguments = parser.parse_args()
if arguments.mode:
    arguments.func(arguments)
