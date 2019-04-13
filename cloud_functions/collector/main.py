import csv
import io
import logging
import os
import re
import zipfile
from datetime import datetime, timezone, timedelta

import firebase_admin
import jaconv
import requests
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1beta1.batch import WriteBatch
from google.cloud.firestore_v1beta1.client import Client

PROJECT_NAME = 'stock-price-pridictor'


def collector(event, context):
    """Collect stock price."""
    date = datetime.now(tz=timezone(timedelta(hours=9)))
    short_year = date.year - 2000
    padded_month = str(date.month).zfill(2)
    padded_date = str(date.day).zfill(2)
    short_date = str(short_year) + padded_month + padded_date

    url = f'http://example.com/20190401.zip'
    logging.debug(url)

    r = requests.get(url)
    if r.status_code == 200:
        file = io.BytesIO(r.content)
        print(len(file.getvalue()))
        csv_path = zipfile.ZipFile(file).extract(f'T{short_date}.csv', path='/tmp')
        with open(csv_path, mode='r', encoding='sjis') as f:
            reader = csv.reader(f)
            save_data(list(reader))

        # Clean up csv file
        os.remove(csv_path)


def save_data(stock_prices: list) -> None:
    """Save csv to Firestore."""
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
        'projectId': PROJECT_NAME
    })

    db: Client = firestore.client()
    batch: WriteBatch = db.batch()
    for idx, row in enumerate(stock_prices):
        # Invalid format
        if len(row) != 10:
            return

        code = int(row[1])
        # Zenkaku to Hankaku
        r = re.search('[0-9]+ (.+)', row[3]).group(1)
        company_name = jaconv.z2h(r, kana=False, digit=True, ascii=True)
        market = jaconv.z2h(row[9], kana=False, digit=True, ascii=True)
        date = datetime.strptime(row[0], '%Y/%m/%d')
        fields = {
            'date': date,
            'code': code,
            'company_name': company_name,
            'opening_quotation': float(row[4]),
            'high': float(row[5]),
            'low': float(row[6]),
            'closing_quotation': float(row[7]),
            'turnover': int(row[8]),
            'market': market
        }

        # Commit batch
        if (idx + 1) % 500 == 0:
            batch.commit()
            batch = db.batch()

        doc_ref = db.collection('stock_price').document(f'{code}-{date.strftime("%Y%m%d")}')
        batch.set(doc_ref, fields)

    batch.commit()
