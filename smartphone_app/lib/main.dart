import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        title: 'Startup Name Generator',
        home: Scaffold(
          appBar: AppBar(
              title: Text('Stock Price Predictor')
          ),
          body: new StockPrice()
        )
    );
  }
}

class StockPrice extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return StreamBuilder<QuerySnapshot>(
      stream: Firestore.instance.collection('stock_price').where('date', isGreaterThan: new DateTime.now()).snapshots(),
      builder: (BuildContext context, AsyncSnapshot<QuerySnapshot> snapshot) {
        if (snapshot.hasError)
          return new Text('Error: ${snapshot.error}');
        switch (snapshot.connectionState) {
           case ConnectionState.waiting: return new Text('Loading...');
          default:
            return new ListView(
              children: snapshot.data.documents.map((DocumentSnapshot document) {
                return new ListTile(
                  title: new Text("date: " + document.data['date'].toString() + ", code: " + document.data['code'].toString()),
                  subtitle: new Text("high: " + document.data['high'].toString()),
                );
              }).toList(),
            );
        }
      },
    );
  }
}
