from flask import Flask, request, render_template
from keras.models import load_model
from sklearn.externals import joblib

DEBUG = False
SECRET_KEY = '1488 xor 228'
USERNAME = 'admin'
PASSWORD = 'default'

app = Flask(__name__)
app.config.from_object(__name__)


vct = joblib.load('ClassifierModel/model/vct.clf')
model = load_model('ClassifierModel/model/nn.h5')

titles = ['Россия', 'Мир', 'Бывший СССР', 'Финансы',
          'Бизнес', 'Силовые структуры', 'Наука и техника', 'Культура',
          'Спорт', 'Интернет и СМИ', 'Ценности', 'Путешествия', 'Из жизни'
          ]


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/clf', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        text = request.form.get('news_text')
        results = process_text(text)
        return render_template('results.html', text=text, entries=results)
    return render_template('home.html')


def process_text(text):
    transformed = vct.transform([text]).todense()
    classification_result = model.predict_classes(transformed, verbose=False)
    results = titles[classification_result]
    if len(classification_result) > 1:
        return results
    return [results]


if __name__ == '__main__':
    app.run(use_reloader=False)
