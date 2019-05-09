# Smart-Bullet
A Cloud-Assisted Bullet Screen Filter based on Deep Learning

# Data Preprocessing
## Data crawl
We use the crawler to crawl data from Tensent video.
The crawled bullet dataset can be found [here](https://github.com/hniu1/Smart-Bullet/tree/master/data_preprocessing/dataset).
## Run
In the following we list some important arguments in```data_preprocessing.py```:
* ```--dataset```: path to the input raw dataset.
* ```--upcount_num```: set the positive data lower upcount level.
* ```--data_size```: the number of data select to process.
* ```--num_negative_select```: the num of selected negative data.

```
cd data_preprocessing
python data_preprocessing.py --dataset dataset/bulletData.csv --upcount_num 100 --num_negative_select 100000
```
The output will be two files (positive, negative).

