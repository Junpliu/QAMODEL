import pandas as pd
import vocab_utils
import matplotlib.pyplot as plt
import numpy as np


def tongji(in_path):
    df = pd.read_csv(in_path, sep="\t", encoding="utf-8", names=["q1", "q2", "label"])
    df["q1_word_len"] = df.q1.map(lambda x: len(vocab_utils.text_to_word_list(x)))
    df["q2_word_len"] = df.q2.map(lambda x: len(vocab_utils.text_to_word_list(x)))
    df["q1_char_len"] = df.q1.map(lambda x: len(vocab_utils.text_to_char_list(x)))
    df["q2_char_len"] = df.q2.map(lambda x: len(vocab_utils.text_to_char_list(x)))
    word1_df = df.groupby(df["q1_word_len"])
    print(word1_df.q1_word_len.describe())
    word2_df = df.groupby(df["q2_word_len"])
    print(word2_df.q2_word_len.describe())
    char1_df = df.groupby(df["q1_char_len"])
    print(char1_df.q1_char_len.describe())
    char2_df = df.groupby(df["q2_char_len"])
    print(char1_df.q2_char_len.describe())

    fig = plt.figure()

    x = np.array(list(word1_df.q1_word_len.min()))
    y = np.array(list(word1_df.q1_word_len.count()))
    plt.subplot(221, xlabel="length", ylabel="count", title="query word level length")
    plt.plot(x, y)

    x = np.array(list(word2_df.q2_word_len.min()))
    y = np.array(list(word2_df.q2_word_len.count()))
    plt.subplot(222, xlabel="length", ylabel="count", title="question word level length")
    plt.plot(x, y)

    x = np.array(list(char1_df.q1_char_len.min()))
    y = np.array(list(char1_df.q1_char_len.count()))
    plt.subplot(223, xlabel="length", ylabel="count", title="query char level length")
    plt.plot(x, y)

    x = np.array(list(char2_df.q2_char_len.min()))
    y = np.array(list(char2_df.q2_char_len.count()))
    plt.subplot(224, xlabel="length", ylabel="count", title="question char level length")
    plt.plot(x, y)

    fig.tight_layout()  # 不加这行代码会导致存储的子图有重叠
    plt.savefig("../data/qq_simscore/seqlen.png")
    plt.show()


tongji("../data/qq_simscore/train.txt")
