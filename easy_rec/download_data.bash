#!/bin/bash
cd ..

mkdir data
cd data

mkdir raw && mkdir processed
cd raw

# MovieLens
curl -k -o ml-20m.zip https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
rm ml-20m.zip

curl -o ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
rm ml-1m.zip

curl -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
rm ml-100k.zip

curl -k -o ml-latest-small.zip https://files.grouplens.org/datasets/movielens/ml-latest-small.zip 
unzip ml-latest-small.zip
rm ml-latest-small.zip


# Amazon Beauty
mkdir amazon_beauty
cd amazon_beauty
curl -k -o All_Beauty.json.gz https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz
gzip -d All_Beauty.json.gz
cd ..

# Behance
mkdir behance
cd behance
curl -k -o Behance_appreciate_1M.gz https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/behance/Behance_appreciate_1M.gz
gzip -d Behance_appreciate_1M.gz
cd ..

# Foursquare
mkdir foursquare-tky
#curl -o dataset_tsmc2014.zip http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip
curl -o dataset_tsmc2014.zip http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip
unzip dataset_tsmc2014.zip
mv dataset_tsmc2014 foursquare-nyc
cp foursquare-nyc/dataset_TSMC2014_TKY.txt foursquare-tky
cp foursquare-nyc/dataset_TSMC2014_readme.txt foursquare-tky
rm dataset_tsmc2014.zip && rm foursquare-nyc/dataset_TSMC2014_TKY.txt


# Steam
mkdir steam
cd steam
curl --max-time 2000 -o steam_reviews.json.gz https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz
gzip -d steam_reviews.json.gz steam.json
mv steam_reviews.json steam.json
cd ..


# Book-Crossing
mkdir bookcrossing
cd bookcrossing
wget -O BX-Book-Ratings.csv https://github.com/ashwanidv100/Recommendation-System---Book-Crossing-Dataset/raw/refs/heads/master/BX-CSV-Dump/BX-Book-Ratings.csv
cd ..

#Gowalla
mkdir gowalla
cd gowalla
wget https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
gzip -d loc-gowalla_totalCheckins.txt.gz
cd ..


#Yelp?
#wget -O yelp.json.zip https://business.yelp.com/external-assets/files/Yelp-JSON.zip --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36"
# header needed to avoid 403 Forbidden error
#unzip yelp.json.zip
#cd Yelp\ JSON
#tar -xvzf yelp_dataset.tar