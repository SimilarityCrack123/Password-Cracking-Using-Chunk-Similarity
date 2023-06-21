import re
import random
import tqdm
import threading

# define training data
# sentences = [
#             ['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
#             ['this', 'is', 'the', 'second', 'sentence'],
#             ['yet', 'another', 'sentence'],
#             ['one', 'more', 'sentence'],
#             ['and', 'the', 'final', 'sentence']]
# # train model
# model = Word2Vec(sentences, min_count=1)
# # fit a 2d PCA model to the vectors
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()
from compue import read_file, vname

train_test_percent = 0.8


def load_keyboard(wordset):
    ll = read_file("data/keyboard.txt", 0)
    ll = sorted(ll, key=lambda x: len(x), reverse=True)
    wordset.extend([k for k in ll if len(k) <= 5])
    chunks_dict = {}
    chunk_len = 3
    for passwd in ll:
        for i in range(len(passwd) - chunk_len):
            ch = passwd[i:i + chunk_len]
            if ch in chunks_dict:
                chunks_dict[ch] += 1
            else:
                chunks_dict[ch] = 0
    res = {key: val for key, val in sorted(chunks_dict.items(), key=lambda ele: ele[1], reverse=True)}
    keyboard_chunks = [k for k in res.keys() if res[k] >= 0]
    wordset.extend(keyboard_chunks)


def load_pinyin(wordset):
    ll = read_file("data/pinyin.txt", 0)
    ll = sorted(ll, key=lambda x: len(x), reverse=True)

    wordset.extend([k for k in ll if len(k) >= 3])
    return [k for k in ll if len(k) < 3]


def load_words(wordset):
    ll = read_file(f"data/{vname}/words_in_pass.txt", 0, n=3000)
    ll = sorted(ll, key=lambda x: len(x), reverse=True)
    wordset.extend([k for k in ll if len(k) >= 3])
    return [k for k in ll if len(k) < 3]


def load_website(wordset):
    ll = read_file("data/website.txt", 0, n=5000)
    ll = sorted(ll, key=lambda x: len(x), reverse=True)
    wordset.extend([k for k in ll if len(k) >= 3])
    return [k for k in ll if len(k) < 3]


def load_words_split(wordset):
    ll = read_file(f"data/{vname}/words_in_pass.txt", 0, n=3000)
    ll = sorted(ll, key=lambda x: len(x), reverse=True)
    wordset.extend([k for k in ll if 4 >= len(k) >= 3])
    # 把长单词拆分，比如 computer
    long_words = [k for k in ll if len(k) >= 5]
    sen = set()
    for passwd in long_words:
        chunk_len = 3
        for i in range(len(passwd) - chunk_len + 1):
            ch = passwd[i:i + chunk_len]
            sen.add(ch)
    wordset.extend(sen)

    return [k for k in ll if len(k) < 3]


all_words = []


def load_predefined_words():
    global all_words
    if len(all_words) == 0:
        load_keyboard(all_words)
        small_pin = load_pinyin(all_words)
        small_word = load_words(all_words)
        small_website = load_website(all_words)
        all_words.extend(small_pin)
        all_words.extend(small_word)
        all_words.extend(small_website)
    return all_words


year_month_pattern1 = re.compile(r'(19[6789]\d|20[0123]\d).?([01]?\d)')
year_month_pattern2 = re.compile(r'([6789]\d|[0123]\d)([01]?\d)')
month_day_pattern = re.compile(r'([01]?\d).?([0123]\d)')
year_month_day_pattern = re.compile(r'(19[6789]\d|20[0123]\d).?([01]?\d).?([0123]\d)')
year_pattern = re.compile(r'\D*(19[6789]\d|20[0123]\d)(\D*)')


def match_year_month_day(word):
    search_obj = year_month_day_pattern.search(word)
    if search_obj is None:
        return None
    else:
        return [search_obj.group(), search_obj.group(1), search_obj.group(2), search_obj.group(3)]


def match_month_day(word):
    search_obj = month_day_pattern.search(word)
    if search_obj is None:
        return None
    else:
        return [search_obj.group(), search_obj.group(1), search_obj.group(2)]


def match_year_month(word):
    search_obj = year_pattern.search(word)
    if search_obj is not None:
        return [search_obj.group(1), search_obj.group(1)]

    search_obj = year_month_pattern1.search(word)
    if search_obj is None:
        search_obj = year_month_pattern2.search(word)
        if search_obj is None:
            return None
    return [search_obj.group(), search_obj.group(1), search_obj.group(2)]


def __psss2chunk(wordset, passwd):
    if len(passwd) == 0:
        return []

    def __get_res(matched):
        if ind == 0:
            # 在最开始，将剩余的
            other = passwd[len(matched):]
            other_result = __psss2chunk(wordset, other)
            res = whole[1:] + other_result
            return res
        elif ind > 0:
            left = passwd[:ind]
            right = passwd[ind+len(matched):]
            left_res = __psss2chunk(wordset, left)
            right_res = __psss2chunk(wordset, right)
            res = left_res + whole[1:] + right_res
            return res
        else:
            raise Exception

    whole = match_year_month_day(passwd)
    if whole is None:
        whole = match_year_month(passwd)
        if whole is None:
            whole = match_month_day(passwd)
    if whole is not None:
        w = whole[0]
        ind = passwd.find(w)
        if ind >= 0:
            result = __get_res(w)
            return result
        else:
            return []

    for w in wordset:
        ind = passwd.find(w)
        if ind >= 0:
            whole = [w, w]
            result = __get_res(w)
            return result
    # 没有匹配到，按两个字母拆分
    sen = []
    chunk_len = 2
    if len(passwd) > chunk_len:
        for i in range(len(passwd) - chunk_len+1):
            ch = passwd[i:i + chunk_len]
            sen.append(ch)
        return sen
    else:
        return [passwd]


def passwd2chunk(passwd):
    wordset = load_predefined_words()
    return __psss2chunk(wordset, passwd)


def word_in_pass():
    # 看看单词在口令里出现情况
    words = read_file("data/words.txt", 0, n=5000)
    passwd = read_file(f"data/{vname}/train_csdn.txt", 0, n=1000000000)
    count_dict = {}
    for w in words:
        count_dict[w] = 0
    progress = tqdm.tqdm(passwd)
    for p in progress:
        p = p.lower()
        for w in words:
            if w in p:
                count_dict[w] += 1

    res = {key: val for key, val in sorted(count_dict.items(), key=lambda ele: ele[1], reverse=True)}
    txt = [k for k in res.keys() if res[k] > 0]
    f = open(f"data/{vname}/words_in_pass.txt", "w")
    f.write("\n".join(txt))
    f.close()

# 对所有口令进行分块，写入chunk_csdn.txt
def get_chunks(file_name):

    processed = {}
    sentences = []
    with open(file_name) as f:
        lines = f.readlines()
        pbar = tqdm.tqdm(lines)
        for line in pbar:
            line = line.strip()
            # 之前处理过，直接取
            if line in processed:
                sentences.append(processed[line])
            else:
                # 对当前口令进行分块
                sen = passwd2chunk(line)
                res = ' '.join(sen)
                sentences.append(res)
                processed[line] = res

    f1 = open(f"data/{vname}/chunk_csdn.txt", "w")
    f1.write('\n'.join(sentences))
    f1.close()

    # using items() to get all items
    # lambda function is passed in key to perform sort by key
    # passing 2nd element of items()
    # adding "reversed = True" for reversed order
    # string = ''
    # res = {key: val for key, val in sorted(chunks_dict.items(), key=lambda ele: ele[1], reverse=True)}
    # for k in res.keys():
    #     string += str(res[k]) + " " + str(k) + "\n"
    # f1 = open("chunk.txt", "w")
    # f1.write(string)
    # f1.close()


# 对所有chunk进行计数，并按频数降序排列，写入chunk.txt
def count_chunk(dataset, split_method):
    chunks_dict = {}

    with open(f"data/{vname}/{dataset}/chunk_{dataset}_{split_method}.txt") as f:
        lines = f.readlines()
        pbar = tqdm.tqdm(lines)
        for line in pbar:
            line = line.split()
            for ch in line:
                if ch in chunks_dict:
                    chunks_dict[ch] += 1
                else:
                    chunks_dict[ch] = 0

    # using items() to get all items
    # lambda function is passed in key to perform sort by key
    # passing 2nd element of items()
    # adding "reversed = True" for reversed order
    string = ''
    res = {key: val for key, val in sorted(chunks_dict.items(), key=lambda ele: ele[1], reverse=True)}
    for k in res.keys():
        string += str(res[k]) + " " + str(k) + "\n"
    f1 = open(f"data/{vname}/{dataset}/chunk_{split_method}.txt", "w")
    f1.write(string)
    f1.close()


# def top_chunk():
#     r = read_file("chunk.txt", item_ind=1, n=1000)
#     print(r)


def get_chunks111(file_name):

    class myThread(threading.Thread):
        def __init__(self, data, sentences, progress=False):
            threading.Thread.__init__(self)
            self.data = data
            self.sentences = sentences
            self.progress_bar = progress

        def run(self):
            if self.progress_bar:
                pbar = tqdm.tqdm(self.data)
            else:
                pbar = self.data
            for line in pbar:
                line = line.strip()
                sen = passwd2chunk(line)
                self.sentences.append(' '.join(sen))


    chunks_dict = {}

    with open(file_name) as f:
        lines = f.readlines()
    part_num = int(len(lines)*0.25)
    part1 = lines[:part_num]
    part2 = lines[part_num:2*part_num]
    part3 = lines[2*part_num:3*part_num]
    part4 = lines[3*part_num:]
    sentence1 = []
    sentence2 = []
    sentence3 = []
    sentence4 = []

    t1 = myThread(part1, sentence1, True)
    t2 = myThread(part2, sentence2)
    t3 = myThread(part3, sentence3)
    t4 = myThread(part4, sentence4)
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    all_sentences = sentence1 + sentence2 + sentence3 + sentence4

    f1 = open("chunk_csdn.txt", "w")
    f1.write('\n'.join(all_sentences))
    f1.close()

    # using items() to get all items
    # lambda function is passed in key to perform sort by key
    # passing 2nd element of items()
    # adding "reversed = True" for reversed order
    # string = ''
    # res = {key: val for key, val in sorted(chunks_dict.items(), key=lambda ele: ele[1], reverse=True)}
    # for k in res.keys():
    #     string += str(res[k]) + " " + str(k) + "\n"
    # f1 = open("chunk.txt", "w")
    # f1.write(string)
    # f1.close()


def get_chunks_old(file_name):
    chunks_dict = {}
    data = []
    sentences = []
    chunk_len = 2
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # count, passwd = line.split(' ')
            passwd = line
            count = 1

            sen = ['START']
            for i in range(len(passwd)-chunk_len):
                ch = passwd[i:i+chunk_len]
                if ch in chunks_dict:
                    chunks_dict[ch] += 1
                else:
                    chunks_dict[ch] = count
                sen.append(ch)
            data.append(sen)
            sen.append('ENDend')
            sentences.append(' '.join(sen))

    f1 = open("chunk_csdn.txt", "w")
    f1.write('\n'.join(sentences))
    f1.close()

    # using items() to get all items
    # lambda function is passed in key to perform sort by key
    # passing 2nd element of items()
    # adding "reversed = True" for reversed order
    string = ''
    res = {key: val for key, val in sorted(chunks_dict.items(), key=lambda ele: ele[1], reverse=True)}
    for k in res.keys():
        string += str(res[k]) + " " + str(k) + "\n"
    f1 = open("chunk.txt", "w")
    f1.write(string)
    f1.close()

# 降序排列训练集口令，并取前n个，写入top_csdn.txt
def top_pass(n=1000.0):
    with open(f"data/{vname}/train_csdn.txt") as f:
        lines = f.readlines()
    pass_dict = {}
    for line in lines:
        # 两种格式，一种是前面是重复次数，之后是口令，另一种就只是口令
        # if len(line.split()) == 1:
        #     continue
        line = line.strip()
        if line in pass_dict:
            pass_dict[line] += 1
        else:
            pass_dict[line] = 1
    res = {key: val for key, val in sorted(pass_dict.items(), key=lambda ele: ele[1], reverse=True)}
    r = list(res.keys())
    if n < 1:
        # 使用占比
        n = int(len(r) * n)
    else:
        n = int(n)
    #  降序排列训练集口令，并取前n个
    r = r[:n]
    f = open(f"data/{vname}/top_csdn.txt", "w")
    f.write('\n'.join(r))
    f.close()
    return

# 相当于恢复原始数据集？ 对口令进行频数次重复
def repeated_train():
    with open(f"data/{vname}/train_csdn.txt") as f:
        lines = f.readlines()
    trains = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        num_pass = line.split()
        if len(num_pass) != 2:
            continue
        num, passwd = num_pass
        num = int(num)
        for i in range(num):
            trains.append(passwd)
    random.shuffle(trains)
    f = open(f"data/{vname}/repeated_train.txt", "w")
    f.write('\n'.join(trains))
    f.close()

# 对原集合进行去重，降序排列之后存到unique_csdn,然后打乱顺序按比例分成训练集和测试集
def unique_pass():
    with open("data/csdn.txt") as f:
        lines = f.readlines()
    pass_dict = {}
    for line in lines:
        line = line.strip()
        # 计算口令出现频数
        if line in pass_dict:
            pass_dict[line] += 1
        else:
            pass_dict[line] = 1
    #   进行降序排列
    res = {key: val for key, val in sorted(pass_dict.items(), key=lambda ele: ele[1], reverse=True)}

    keys = list(res.keys())
    # 打乱keys中的顺序
    random.shuffle(keys)
    random.shuffle(keys)
    # 一部分作为训练集（80%），一部分作为测试集（20%）
    count = int(len(keys) * train_test_percent)
    # 前80%为训练集
    train_keys = set(keys[:count])
    string = ''
    train_string = ''
    tests_string = ''
    f = open(f"data/{vname}/unique_csdn.txt", "w")
    ftrain = open(f"data/{vname}/train_csdn.txt", "w")
    ftests = open(f"data/{vname}/test_csdn.txt", "w")
    for k in res.keys():
        item = f"{res[k]}\t{k}\n" # 频数， 口令
        string += item
        if k in train_keys:
            train_string += item
        else:
            tests_string += item

    f.write(string)  # 总集（去重后）
    ftrain.write(train_string)  # 训练集（去重后）
    ftests.write(tests_string)  # 测试集（去重后）
    f.close()
    ftrain.close()
    ftests.close()
    return

# 打乱顺序，按比例分成训练集和测试集
def split_dataset():
    with open("data/csdn.txt") as f:
        lines = f.readlines()
    # 将口令顺序打乱
    random.shuffle(lines)
    random.shuffle(lines)

    train_num = int(train_test_percent*len(lines))
    # 前80%为训练集，其余为测试集（未去重）
    train_string = ''.join(lines[:train_num])
    tests_string = ''.join(lines[train_num:])
    ftrain = open(f"data/{vname}/train_csdn.txt", "w")
    ftests = open(f"data/{vname}/test_csdn.txt", "w")

    ftrain.write(train_string)
    ftests.write(tests_string)
    ftrain.close()
    ftests.close()
    return


if __name__ == '__main__':
    # unique_pass()
    # split_dataset()
    # repeated_train()
    # word_in_pass()

    # s = passwd2chunk('dearbook')
    # s = match_year_month_day("ads2022312r31ss")
    # s = match_year_month("hoylove2009")
    # print(s)
    # if vname == 'v1':
    #     get_chunks(f'data/{vname}/repeated_train.txt')
    # else:
    #     get_chunks(f'data/{vname}/train_csdn.txt')
    # top_pass(100000000)
    # count_chunk("myspace", "baseline")
    r = passwd2chunk('love@1234')
    print(r)

    # a = ['guaiwu588', 'xinyuan1978', '19791112', 'csdn1981315']
    # for p in a:
    #     r = passwd2chunk(p)
    #     print(r)
