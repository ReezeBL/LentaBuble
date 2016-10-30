from flask import Flask, request, render_template

DEBUG = True
SECRET_KEY = '1488 xor 228'
USERNAME = 'admin'
PASSWORD = 'default'

app = Flask(__name__)
app.config.from_object(__name__)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/clf', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        text = request.form.get('news_text')
        return render_template('results.html', text=text)
    return render_template('home.html')

if __name__ == '__main__':
    app.run()
