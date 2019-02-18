#!/bin/sh
cd /data
mkdir bbc-topic-classification
cd bbc-topic-classification
wget http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
unzip bbc-fulltext.zip -d ./bbc_data
rm bbc-fulltext.zip
cp -r ./bbc_data/bbc/* ./bbc_data
rm -r ./bbc_data/bbc/
rm ./bbc_data/README.TXT

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -j glove.6B.zip glove.6B.300d.txt -d .
rm glove.6B.zip
