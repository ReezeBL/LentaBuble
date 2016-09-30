import requests
import lxml.html
import datetime
import pickle

url = 'https://lenta.ru/{0}'
api_url = 'https://lenta.ru/rubrics/{rubric}/{year}/{month:02d}/{day:02d}'
rubrics = ['russia', 'world', 'ussr', 'economics',
           'business', 'forces', 'science', 'culture',
           'sport', 'media', 'style', 'travel', 'life'
           ]
headers = {'user-agent':'Lenta/1.2.1 (iPhone; iOS 9.3.1; Scale/2.00)'}


def get_news_links(rubric_name, date):
    r = requests.get(api_url.format(rubric=rubric_name, year=date.year, month=date.month, day=date.day))
    tree = lxml.html.fromstring(r.text)
    refs = tree.xpath('//h3/a/@href')
    return refs


def get_article_content(article_url):
    r = requests.get(url.format(article_url))
    tree = lxml.html.fromstring(r.text)
    text = tree.xpath('//div[@itemprop="articleBody"]//p/text()')
    text = ' '.join(text)
    return text


def crawl():
    current = datetime.date.today()
    delta = datetime.timedelta(1)
    end_date = datetime.date(2016, 7, 27)

    data, labels = [], []
    try:
        while current > end_date:
            print('Gathering data on {0}'.format(current))
            for i, rubric in enumerate(rubrics):
                refs = get_news_links(rubric, current)
                for ref in refs:
                    article_text = get_article_content(ref)
                    data.append(article_text)
                    labels.append(i)
            current = current - delta
    finally:
        with open('data.dat', 'wb') as f:
            pickle.dump((data, labels), f)
        print('Dumped {0} articles.'.format(len(data)))

if __name__ == '__main__':
    crawl()
