from datetime import datetime, timedelta, timezone

import firebase_admin
import jpholiday
from firebase_admin import credentials
from firebase_admin import firestore
from google.api_core.datetime_helpers import from_rfc3339
from google.cloud.firestore import Query
from google.cloud.firestore_v1beta1 import DocumentSnapshot
from google.cloud.firestore_v1beta1.client import Client
# from oauth2client.client import GoogleCredentials
from googleapiclient import discovery

PROJECT_NAME = 'stock-price-pridictor'


def main(event, context):
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
        'projectId': PROJECT_NAME
    })

    db: Client = firestore.client()

    query_date = _get_query_date()

    price_ref = db.collection('stock_price')
    prices: [DocumentSnapshot] = price_ref.where('date', '>=', query_date).get()

    def build_data(price: DocumentSnapshot) -> dict:
        data = price.to_dict()
        return {
            'date': data['date'],
            'code': data['code'],
            'opening_quotation': data['opening_quotation'],
            'high': data['high'],
            'turnover': data['turnover'],
            'closing_quotation': data['closing_quotation'],
            'low': data['low']
        }
    stock_prices = list(map(build_data, prices))
    predictable_stock_codes = []
    data_sets = []

    for code in range(1001, 9999):
        filtered = list(filter(lambda x: x['code'] == code, stock_prices))
        if len(filtered) == 0:
            continue

        data = sorted(filtered, key=lambda x: x['date'])

        def build_dataset(d: dict) -> list:
            return [
                d['opening_quotation'],
                d['high'],
                d['turnover'],
                d['closing_quotation'],
                d['low']
            ]
        data_sets.append(list(map(build_dataset, data)))
        predictable_stock_codes.append(data[0]['code'])

    input = list(map(lambda d: {'input': d}, data_sets))

    ml = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(PROJECT_NAME, 'stock_predictor')
    name += '/versions/{}'.format('stock_price_predictor')
    response = ml.projects().predict(
        name=name,
        body={'instances': input}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    print(response['predictions'])
    prediction_results = response['predictions']

    # TODO save data at next day or week day
    l = list(filter(lambda x: x['code'] == code, stock_prices)) # FIXME get from now date
    s = sorted(l, key=lambda x: x['date'])
    print(s)
    predict_datetime = from_rfc3339(s[-1]['date'].rfc3339()) # Get latest date in predicted dataset
    original_datetime = predict_datetime

    return

    while True:
        delta = timedelta(days=1)
        predict_datetime = predict_datetime + delta

        # Skip Japan holidays and JPX holidays (Dec 31, Jun 1, 2 and 3)

        if not _is_jpx_holiday(predict_datetime):
            break


def _get_query_date(days_term=10) -> datetime:
    """Gets date between now and specified before days except JPX holidays."""
    delta = -1
    days_count = 1
    now = datetime.now()
    now = datetime(now.year, now.month, now.day, tzinfo=timezone(timedelta(hours=9)))
    while True:
        if days_count > days_term:
            break

        if not _is_jpx_holiday(now + timedelta(days=delta)):
            days_count += 1
        delta -= 1

    delta += 1 # Because delta is subtracted additional 1

    return now + timedelta(days=delta)


def _is_jpx_holiday(d: datetime) -> bool:
    if jpholiday.is_holiday(d.date()):
        return True

    if d.month == 12 and d.day == 31:
        return True

    if d.month == 1 and (d.day == 1 or d.day == 2 or d.day == 3):
        return True

    if d.weekday() >= 5: # Skip Saturday and Sunday
        return True

    return False


if __name__ == '__main__':
    main(None, None)
