#!/bin/sh
TAG=aws-try1

if [ -z "$SPARK_BIN" ]; then
    echo "Need to set SPARK_BIN"
    exit 1
fi

cd datasets
DSET=2c30m_dataset.txt
echo generating $DSET ..
python ./dtgen.py 100
mv dataset.txt $DSET
wc -l $DSET
cd ..

echo Using input: $DSET

cp k_means_coreset/*.py spark/spark_kmeans_python.py .
sed "s/DATASET/file:\/\/datasets\/$DSET/;" spark/config.ini > ./config.ini
vi config.ini
cp demo_spark.py demo_spark1.py
vi demo_spark1.py
$SPARK_BIN/spark-submit demo_spark1.py

cp spark_coreset/go.sh .
cp spark_coreset/k_means_coreset/*.py .
cp spark_coreset/spark/{config.ini,tree.py} .
cp spark_coreset/graph.py .
echo input_s3_url = /root/spark/spark_coreset/datasets/mixed-1m_dataset.txt

mkdir res.$TAG
cp results* figure* demo_spark1.py res.$TAG
cat results_mis.csv
ls -ltr res.$TAG

