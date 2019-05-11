# coding: utf-8
import tensorflow as tf
import numpy as np
import json
import os
import time
import datetime
from tensorflow.contrib import learn
import re
import sklearn.preprocessing
import jieba
from flask import Flask, request, jsonify, make_response

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")


def convert2vector (sentencelist, maximum, worddict):
    regulated_word_list = []
    c = 0
    none_index = []
    for sentence in sentencelist:
        # print sentence
        # temp_words = sentence.strip().split(' ')
        # temp_words = [worddict[item.decode('utf8')] for item in sentence]
        # temp_words = [worddict[item] for item in sentence]

        try:
            temp_words = [worddict[item] for item in sentence]
        except:
            temp_words = []
            for item in sentence:
                try:
                    temp_words.append(worddict[item])
                except:
                    continue
        if len(temp_words) == 0:
            none_index.append(c)

        if len(temp_words) > maximum:
            temp_words = temp_words[:maximum]
        elif len(temp_words) < maximum:
            temp_words.extend([0] * (maximum - len(temp_words)))
        else:
            pass

        regulated_word_list.append(temp_words)
        c += 1
    return regulated_word_list, none_index

def basic_tokenizer(split_sentence, counter, stop_words):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    final = []
    negative_count = 0


    split_sentence = re.sub("[\s+\.\!\/_,$¥%^*(+\"\']+|[+——！{}、！·《》\[\]'：“”【】，\-_——。？?::、~@#￥%……&*（）]+".decode("utf8"),\
                        "".decode("utf8"),split_sentence)
    split_sentence = list(jieba.cut(split_sentence, cut_all=False))


    for seg in split_sentence:
        # clean stopwords
            if seg not in stop_words:
                # if seg in high_frq_words:
                final.append(seg)

    for space_separated_fragment in final:
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))

    if counter % 10 == 0:
        print('tokenizing counter: ', counter)

    return [w for w in words if w]

def preprocess(wholeDict, stop_words, input_data):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    # x_text, y = data_helpers.load_data_and_labels("./data/test")
    # with open("./data/negative.json", 'r') as js:
    #     data_dict = json.load(js)

    data = input_data["content"]
    c = 0

    data_tokenize = []
    for item in data:
        data_tokenize.append(basic_tokenizer(item, c, stop_words))
        c += 1

    y = [[0, 1] for _ in data_tokenize]


    # Build vocabulary
    wordlists, none_index = convert2vector(data_tokenize, 132, wholeDict)
    x = np.array(wordlists)

    print("Vocabulary Size: {:d}".format(len(wholeDict.keys())))
    # print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("Predict num: {:d}".format(len(y)))
    return x, y, none_index

def predict(sess, x_input, y_input, input_x, input_y, predictions, dropout_keep_prob, none_index):
        pred = sess.run(
        predictions,
        feed_dict={
            input_x: x_input,
            input_y: y_input,
            dropout_keep_prob: 1.0
        })
    c = 0

    for item in none_index:
        pred[item] = 0

    for item in pred:
        if item == 1:
            c+=1
    print(pred)
    return pred

app = Flask(__name__)

# @app.route("/")
# def getdata():
# 	danmu = requests.get("https://api.bilibili.com/x/v1/dm/list.so?oid=42679001")
# 	##danmu_xml = ElementTree.fromstring(danmu.content)
# 	return danmu.content
@app.route("/",methods=['POST'])
def operate_json():
    data = request.get_json()
    x_input, y_input, none_index = preprocess(wholeDict, stop_words, data)

    output = predict(sess, x_input, y_input, input_x, input_y, predictions, dropout_keep_prob, none_index)

    return jsonify({"id":list(output)})

if __name__ == '__main__':
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('model/checkpoints/model-3200.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('model/checkpoints'))
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        input_y = graph.get_tensor_by_name("input_y:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        predictions = graph.get_tensor_by_name("output/predictions:0")

        with open("wholeDict.json", 'r') as Dict:
            wholeDict = json.load(Dict)

        stop_words = [line.strip().decode('utf-8') for line in open('stop_words.txt').readlines()]

        # app.run(debug=True)
        app.run(host="127.0.0.1", port="5000")