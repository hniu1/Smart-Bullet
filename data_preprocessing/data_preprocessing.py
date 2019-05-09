# coding: utf-8

import argparse
import csv
import pickle
import collections
import operator
import re
import json
import numpy as np
import pandas as pd
import random
import argparse
import operator
import re
import json
import sklearn.preprocessing
import jieba


_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")

#clean upcount which is not numbers
def clean_upcount(data):
    if not re.match(r'^[0-9]+$', data):
        return False
    else:
        return True

#clean content which are not \w
def clean_content(data):
    if re.match(ur'[\u4e00-\u9fff]+', data) or re.match(r'\w', data):
        return True
    else:
        return False

def positive_tokenizer(split_sentence, counter, stop_words, negative_words):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    final = []
    negative_count = 0


    split_sentence = re.sub("[\s+\.\!\/_,$¥%^*(+\"\']+|[+——！{}、！·《》\[\]'：“”【】，\-_——。？?::、~@#￥%……&*（）]+".decode("utf8"),\
                        "".decode("utf8"),split_sentence)
    split_sentence = list(jieba.cut(split_sentence, cut_all=False))

    for w in split_sentence:
        if w in negative_words:
            negative_count += 1
    if negative_count == 0:
        for seg in split_sentence:
            # clean stopwords
                if seg not in stop_words:
                    # if seg in high_frq_words:
                    final.append(seg)

    for space_separated_fragment in final:
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))

    if counter % 5000 == 0:
        print('positive tokenizing counter: ', counter)

    return [w for w in words if w]

def negative_tokenizer(split_sentence, counter, stop_words, negative_words):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    final = []

    try:
        split_sentence = re.sub("[\s+\.\!\/_,$¥%^*(+\"\']+|[+——！{}、！·《》\[\]'：“”【】，\-_——。？?::、~@#￥%……&*（）]+".decode("utf8"),\
                            "".decode("utf8"),split_sentence)
        split_sentence = list(jieba.cut(split_sentence, cut_all=False))

        for w in split_sentence:
            if w in negative_words:
                for seg in split_sentence:
                    # clean stopwords
                        if seg not in stop_words:
                            # if seg in high_frq_words:
                            final.append(seg)
                continue

    except:
        print('sentence', counter, 'no negative words or in stop words')
        # print(split_sentence)


    for space_separated_fragment in final:
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))

    if counter % 5000 == 0:
        print('negative tokenizing counter: ', counter)

    return [w for w in words if w]

def merge_data(data):

    merged_data = []
    del_list = []
    for i in range(len(data)):
        for j in range(len(data[i+1:])):
            if data[i]['recipe'] == data[i+j+1]['recipe']:
                data[i]['upcount'] += data[i+j+1]['upcount']
                del_list.append(i+j+1)


    del_sort = list(set(del_list))

    del_sort.sort(reverse=True)

    print('The number of deleted data because of same comment:', len(del_sort))

    for index in del_sort:
        del data[index]

    return data

def parse_recipes(input_data):
  # Example entry:
  #"title": "Tweet New RSS Feed Item",
  #"description": "Automatically tweet new RSS feed items.",
  #"action_channel": "TwitterV2API",
  #"event_channel": "RSSAPI",
  #"action": "tweet",
  #"event": "new_feed",
  #"rule": "{u'message': u'{{title}}: {{link}}'}"
    recipes = []
    for item in input_data:
        if item["Upcount"] == None:
            item["Upcount"]="None"

        recipes.append({
            'recipe': item["Content"],
            'upcount': item["Upcount"],
        })
    return recipes


def parse_bullet(input_data_positive, input_data_negative):
    c = 0
    stop_words = [line.strip().decode('utf-8') for line in open('stop_words.txt').readlines()]
    negative_words = [line.strip().decode('utf-8') for line in open('negative_words.txt').readlines()]

    input_recipes_positive = parse_recipes(input_data_positive)
    input_recipes_negative = parse_recipes(input_data_negative)

    data_positive = []
    data_negative = []

    for item in input_recipes_positive:
        temp = positive_tokenizer(item['recipe'], c, stop_words, negative_words)
        if len(temp) == 0:
            continue
        else:
            data_positive.append({'recipe': temp,
                         'upcount': item['upcount'],
                         })
            c += 1
    print('positive tokenizing is finished, num: ', c)

    c = 0

    for item in input_recipes_negative:
        temp = negative_tokenizer(item['recipe'], c, stop_words, negative_words)
        if len(temp) == 0:
            continue
        else:
            data_negative.append({'recipe': temp,
                         'upcount': item['upcount'],
                         })
            c += 1
    print('negative tokenizing is finished, num: ', c)
    print('positive same data merge:')
    positive_merged = merge_data(data_positive)
    print('negative same data merge:')
    negative_merged = merge_data(data_negative)

    return positive_merged, negative_merged

def data_load():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--output_train', required=True, help='path of output.')
    # parser.add_argument('--output_dev', required=True, help='path of output.')
    # parser.add_argument('--output_test', required=True, help='path of output.')
    # parser.add_argument('--level', required=True, help='the number of levels, only 5 or 10.')
    parser.add_argument('--data_size', help='the number of data.')
    parser.add_argument('--dataset', required=True, help='path of data.')
    # parser.add_argument('--output_data', help='path of output.')
    # parser.add_argument('--input_data', help='input data need parse')
    parser.add_argument('--upcount_num', required=True,
                        help='select positive data if upcount >= this num, if will generate two df')
    parser.add_argument('--num_negative_select', help='the num of selected negative data')
    # parser.add_argument('--input_negative_words', help='the path of negative words')
    # parser.add_argument('--output_data_positive', help='the path of positive output data')
    # parser.add_argument('--output_data_negative', help='the path of negative output data')

    args = parser.parse_args()

    # read data from path
    bullet_df = pd.read_csv(args.dataset,
                            encoding='utf-8', header=None, sep='\t')

    print('The loaded data shape is', bullet_df.shape)

    # Rename columns
    bullet_df.columns = ['TargetID', 'CommentID', 'Content', 'Upcount', 'IsFriend', 'IsOp', 'IsSelf', 'TimePoint',
                         'Vip_Degree']
    if args.data_size:
        num = int(args.data_size)
        bullet_df = bullet_df[:num]

    # drop any row if content or upcount is none
    bullet_df = bullet_df[pd.notnull(bullet_df['Content'])]
    bullet_df = bullet_df[pd.notnull(bullet_df['Upcount'])]

    print('The data shape after clean none cells is ', bullet_df.shape)

    # clean Upcount and Content columns
    bullet_df['cleanContent'] = bullet_df.apply(lambda x: clean_content(x.Content), axis=1)
    bullet_df['cleanUpcount'] = bullet_df.apply(lambda x: clean_upcount(x.Upcount), axis=1)

    clean_list = bullet_df[(bullet_df['cleanUpcount'] == False)].index.tolist()
    bullet_df.drop(clean_list, axis=0, inplace=True)

    clean_content_list = bullet_df[(bullet_df['cleanContent'] == False)].index.tolist()
    bullet_df.drop(clean_content_list, axis=0, inplace=True)

    print('The data shape after cleanup: ', bullet_df.shape)

    bullet_df['Upcount'] = bullet_df['Upcount'].astype('int')

    clean_df = bullet_df[['Content', 'Upcount']]

    clean_df.index = range(len(clean_df))

    positive_list = clean_df[(clean_df['Upcount'] >= int(args.upcount_num))].index.tolist()
    bullet_df_positive = clean_df.iloc[positive_list, :]

    negative_list = clean_df[(clean_df['Upcount'] == 0)].index.tolist()
    if args.num_negative_select:
        num_negative_select = int(args.num_negative_select)
        list_of_random_items = random.sample(negative_list, num_negative_select)
        bullet_df_negative = clean_df.iloc[list_of_random_items, :]
    else:
        bullet_df_negative = clean_df.iloc[negative_list, :]

    print('The positive data shape when upcount larger than', int(args.upcount_num),':', bullet_df_positive.shape)
    print('The negative data shape when upcount is 0', bullet_df_negative.shape)
    print('The data shape', bullet_df.shape)

    bullet_df_positive.index = range(len(bullet_df_positive))
    bullet_df_negative.index = range(len(bullet_df_negative))

    data_positive = bullet_df_positive.to_json(orient="records", force_ascii=False)
    data_negative = bullet_df_negative.to_json(orient="records", force_ascii=False)

    input_data_positive = json.loads(data_positive)
    input_data_negative = json.loads(data_negative)

    return input_data_positive, input_data_negative

def join_string(wordlist):
    desiredString = ""
    for item in wordlist:
        desiredString += item + ' '
    return desiredString

def data_label(positive_data, negative_data):
    # with open('positive.json', 'rb') as psf:
    #     positve_data = json.load(psf)

    positiveString = ""
    positive_words_list = [item['recipe'] for item in positive_data][:len(negative_data)]

    x = open('finalTest/danmaku/positive', 'w')
    for item in positive_words_list:
        temp_string = join_string(item)
        positiveString += temp_string.strip() + "\n"

    x.write(positiveString.encode('utf-8'))
    x.close()

    # --------------------- end of positive --------------------------- #

    # with open('test_negative_words_final.json', 'rb') as ngf:
    #     negative_data = json.load(ngf)

    negativeString = ""
    negative_words_list = [item['recipe'] for item in negative_data]

    y = open('finalTest/danmaku/negative', 'w')
    for item in negative_words_list:
        temp_string = join_string(item)
        negativeString += temp_string.strip() + "\n"

    y.write(negativeString.encode('utf-8'))
    y.close()

    print('data pre-processing finished')



def main():
    input_data_positive, input_data_negative = data_load()

    data_p, data_n = parse_bullet(input_data_positive, input_data_negative)

    data_label(data_p, data_n)


    # jsObj_p = json.dumps(data_p, indent=4, separators=(',', ':'))
    # jsObj_n = json.dumps(data_n, indent=4, separators=(',', ':'))
    #
    # fileObject = open('finalTest/positive.json', 'w')
    # fileObject.write(jsObj_p)
    # fileObject.close()
    #
    # fileObject = open('finalTest/negative.json', 'w')
    # fileObject.write(jsObj_n)
    # fileObject.close()

    return


if __name__ == '__main__':
    main()
