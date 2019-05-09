# Smart-Bullet
A Cloud-Assisted Bullet Screen Filter based on Deep Learning

# Data Preprocessing
## Data crawl
We use the crawler to crawl data from Tensent video.
The crawled bullet dataset can be found [here](https://github.com/hniu1/Smart-Bullet/tree/master/data_preprocessing/dataset).
## Run
```
cd data_preprocessing
python data_preprocessing.py
--dataset
dataset/bulletData.csv
--data_size
1000
--upcount_num
100
--num_negative_select
100
```

