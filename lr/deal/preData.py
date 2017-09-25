#!/usr/bin/python
# -*- coding: utf-8 -*-
import struct
from scipy import sparse
import os
import jieba
import numpy as np


class preData:
    def __init__(self):
        self.tag_count = 0
        self.sub_count = 0
        self.word_vector_dict = {}

    def load_word_vectors(self, w2v_file):
        """加载词向量二进制到内存
        :param w2v_file:
        """
        float_size = 4  # 一个浮点数4字节
        max_w = 50  # 最大单词字数
        input_file = open(w2v_file, "rb")
        # 获取词表数目及向量维度
        words_and_size = input_file.readline()
        words_and_size = words_and_size.strip()
        words = long(words_and_size.split(' ')[0])
        word_vec_dim = long(words_and_size.split(' ')[1])
        print("词表总词数：%d" % words)
        print("词向量维度：%d" % word_vec_dim)

        for b in range(0, words):
            a = 0
            word = ''
            # 读取一个词
            while True:
                c = input_file.read(1)
                word = word + c
                if False == c or c == ' ':
                    break
                if a < max_w and c != '\n':
                    a = a + 1
            word = word.strip()
            vector = []

            for index in range(0, word_vec_dim):
                m = input_file.read(float_size)
                (weight,) = struct.unpack('f', m)
                f_weight = float(weight)
                vector.append(f_weight)

            # 将词及其对应的向量存到dict中
            try:
                self.word_vector_dict[word.decode('utf-8')] = vector[0:word_vec_dim]
            except:
                # 异常的词舍弃掉
                # print('bad word:' + word)
                pass

        input_file.close()
        print "finish"

    def count_word_tag(self, data_set):
        for l in data_set.tag:
            c = set()
            words = jieba.cut(l.strip(), cut_all=False)
            for word in words:
                c.add(word)
            if self.tag_count < len(c):
                self.tag_count = len(c)

    def count_word_subject(self, data_set):
        for s in data_set.subject:
            c = set()
            words = jieba.cut(s.strip(), cut_all=False)
            for word in words:
                c.add(word)
            if self.sub_count < len(c):
                self.sub_count = len(c)

    def transform_data(self, data_set):
        print "transforming data in to words 2 vector"
        data = []
        target = []
        for l in data_set.values:
            tag_vector = self.tag_array(l[1])
            sub_vector = self.sub_array(l[2])
            data.append(np.hstack((tag_vector, sub_vector)))
            target.append(l[0])
        print "done"
        return sparse.csr_matrix(np.asarray(data)), np.asarray(target)

    def tag_array(self, tag):
        tag_vector = np.zeros(1)
        words = jieba.cut(tag.strip(), cut_all=False)
        index = 0
        for word in words:
            w_vec = self.word_vector_dict[word]  # 200维
            index += 1
            if len(w_vec) > 0:
                tag_vector = np.hstack((tag_vector, np.array(w_vec)))
            else:
                tag_vector = np.hstack((tag_vector, np.zeros(200)))
        return np.hstack((tag_vector, np.zeros((self.tag_count - index) * 200)))[1:]

    def sub_array(self, sub):
        sub_vector = np.zeros(1)
        words = jieba.cut(sub.strip(), cut_all=False)
        index = 0
        for word in words:
            w_vec = self.word_vector_dict[word]  # 200维
            index += 1
            if len(w_vec) > 0:
                sub_vector = np.hstack((sub_vector, np.array(w_vec)))
            else:
                sub_vector = np.hstack((sub_vector, np.zeros(200)))
        return np.hstack((sub_vector, np.zeros((self.sub_count - index) * 200)))[1:]