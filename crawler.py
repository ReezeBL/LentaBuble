import asyncio
from aiohttp import ClientSession
import datetime

import lxml.html
import pandas as pd
import requests

url = 'https://lenta.ru/{0}'
api_url = 'https://lenta.ru/rubrics/{rubric}/{year}/{month:02d}/{day:02d}'
rubrics = ['russia', 'world', 'ussr', 'economics',
           'business', 'forces', 'science', 'culture',
           'sport', 'media', 'style', 'travel', 'life'
           ]
headers = {'user-agent':'Lenta/1.2.1 (iPhone; iOS 9.3.1; Scale/2.00)'}


async def fetch(url, session):
    async with session.get(url) as response:
        return await response.text()


async def get_news_links(rubric_name, date, session):
    r = await fetch(api_url.format(rubric=rubric_name, year=date.year, month=date.month, day=date.day), session)
    tree = lxml.html.fromstring(r)
    refs = tree.xpath('//h3/a/@href')
    return refs


async def get_article_content(article_url, session):
    r = await fetch(url.format(article_url), session)
    tree = lxml.html.fromstring(r)
    text = tree.xpath('//div[@itemprop="articleBody"]//p/text()')
    text = ' '.join(text)
    return text


async def gather_rubric_content(rubric, date, session):
    refs = await get_news_links(rubric, date, session)
    tasks = [asyncio.ensure_future(get_article_content(ref, session)) for ref in refs]
    return await asyncio.gather(*tasks)


async def crawl_rubric(rubric, date, label, session):
    articles = await gather_rubric_content(rubric, date, session)
    return [(article, label) for article in articles]


async def crawl_date(date, session):
    data = []
    try:
        print('Gathering data on {0}'.format(date))
        tasks = [asyncio.ensure_future(crawl_rubric(rubric, date, i, session)) for i, rubric in enumerate(rubrics)]
        results = await asyncio.gather(*tasks)
        for result in results:
            data += result
    except BaseException as e:
        print('Error during processing data on {0}, message{1}'.format(date, e))
    finally:
        return data


async def crawl_period(base: datetime.date, days: int, session: ClientSession):
    dates = [base - datetime.timedelta(x) for x in range(days)]
    tasks = [asyncio.ensure_future(crawl_date(date, session)) for date in dates]
    results = await asyncio.gather(*tasks)
    data = []
    for result in results:
        data += result
    return data


async def crawl():
    period = 7
    current = datetime.date.today()
    end_date = datetime.date(2016, 7, 27)

    data = []
    async with ClientSession() as session:
        while current > end_date:
            period = min(period, (current - end_date).days)
            data += await crawl_period(current, period, session)
            current -= datetime.timedelta(period)

    print('Downloading done! Storing data')
    df = pd.DataFrame(data=data, columns=['Data', 'Labels'])
    df.to_csv('Articles.csv')
    print('Stored {0} articles.'.format(len(data)))

async def test():
    async with ClientSession() as session:
        data = await fetch('https://lenta.ru/', session)
        print(data)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.ensure_future(crawl()))
    loop.close()
