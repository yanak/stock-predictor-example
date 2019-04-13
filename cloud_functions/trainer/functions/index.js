const functions = require('firebase-functions');
const admin = require('firebase-admin');
const Papa = require('papaparse');
const {Storage} = require('@google-cloud/storage');
const fs = require('fs');
const os = require('os');
const path = require('path');

admin.initializeApp(functions.config().firebase);
let db = admin.firestore();

exports.train = functions.pubsub.topic('train').onPublish((message) => {
    const startDate = new Date('2015-1-1');
    startDate.setTime(startDate.getTime() + 1000*60*60*9);
    const start = admin.firestore.Timestamp.fromDate(startDate);

    const endDate = new Date('2019-1-10');
    endDate.setTime(endDate.getTime() + 1000*60*60*9);
    const end = admin.firestore.Timestamp.fromDate(endDate);

    return db.collection('stock_price').doc('2802').collection('price')
        .where('date', '>=', start)
        .where('date', '<=', end)
        .orderBy('date', 'asc')
        .get()
        .then((snapshot) => {
            if (snapshot.empty) {
                console.log('No matching documents');
                return;
            }

            const stocks = snapshot.docs.map(doc => {
                let stock = doc.data();
                stock['date'] = stock['date'].toDate();
                return stock
            });
            const csv = Papa.unparse(stocks);
            const tempFilePath = path.join(os.tmpdir() + 'dataset.csv');
            fs.writeFileSync(tempFilePath, csv);

            const projectId = 'stock-price-predictor';
            const storage = new Storage({
                projectId: projectId
            });
            const bucketName = 'training-dataset-1';
            const bucket = storage.bucket(bucketName);
            bucket.upload(tempFilePath, {
                gzip: true,
                metadata: {
                    contentEncoding: 'gzip',
                    contentType: 'text/csv'
                }
            })
        });
});
