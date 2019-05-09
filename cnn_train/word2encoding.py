# -*- coding: utf-8 -*-

def buildChineseDict (wordlists):
    worddict = {}
    countNumber = 1
    for sentence in wordlists:
        # print sentence
        words = sentence.strip().split(" ")
        # print words
        for word in words:
            if word in worddict.keys():
                continue
            else:
                worddict[word] = countNumber
                countNumber += 1

    return worddict


def convert2vector (sentencelist, maximum, worddict):
    regulated_word_list = []
    for sentence in sentencelist:
        # print sentence
        temp_words = sentence.strip().split(' ')
        temp_words = [worddict[item] for item in temp_words]
        if len(temp_words) > maximum:
            temp_words = temp_words[:maximum]
        elif len(temp_words) < maximum:
            temp_words.extend([0] * (maximum - len(temp_words)))
        else:
            pass
        regulated_word_list.append(temp_words)
    return regulated_word_list

