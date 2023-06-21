import math
import sys

import tqdm
from gensim.models import Word2Vec,KeyedVectors,FastText
import numpy as np
from zxcvbn import zxcvbn

import chunc2vec

vname = 'v3'
add_train_pass = True
use_online_crack_info = False


def read_file(file_name, item_ind, sep=None, n=1000, count_dict=False):
    with open(file_name) as f:
        lines = f.readlines()
        # 取文件中的前n个元素
        lines = lines[:n]
    ch = []
    cdict = {}
    # 另一种格式，第一列保存的是口令的重复数
    format_1 = lines[0].split()
    format_1 = len(format_1) == 2
    # format为true表示是频数和口令，false只有口令
    if format_1 and item_ind == 0:
        item_ind = 1
    for line in lines:
        # 删除首尾空格
        line = line.strip()
        if len(line) == 0:
            continue
        if format_1: # 格式1：频数， 口令
            if sep is not None:
                sp = line.split(sep)
            else:
                sp = line.split()
            if item_ind >= len(sp):
                continue
            item = sp[item_ind]
        else:
            item = line # 取口令
        if count_dict:
            # 支持两种格式
            if format_1:
                value = int(sp[item_ind - 1]) # 在格式1中取频数
            else:
                value = 1
            if item in cdict:
                cdict[item] += value
            else:
                cdict[item] = value

        else:
            ch.append(item)
    if count_dict:
        return cdict # 返回有频数的字典
    return ch   # 返回口令list

# 按概率排布表示
def top_chunk_prob():
    chunk_dict = read_file(f"data/{vname}/rockyou/chunk_baseline.txt", 1, n=1000000, count_dict=True)
    # chunk_dict = read_file(f"data/{vname}/chunk_level.txt", 1, n=1000000, count_dict=True)
    # chunk_dict = read_file(f"data/{vname}/chunk.txt", 1, n=1000000, count_dict=True)
    all_count = 0
    for key in chunk_dict:
        all_count += chunk_dict[key]
    for key in chunk_dict:
        chunk_dict[key] /= all_count
    result = {}
    for key in chunk_dict:
        if chunk_dict[key] > 0:
            result[key] = chunk_dict[key]
    return result


def top_chunk(n=3000):
    # return read_file(f"data/{vname}/rockyou/chunk_baseline.txt", 1, n=n)
    return read_file(f"data/{vname}/chunk_level.txt", 1, n=n)
    # return read_file(f"data/{vname}/chunk.txt", 1, n=n)

#按频数降序排列，取前n个
def top_pass(n=500000):
    print(f"top password: {n}")
    return read_file(f"data/{vname}/top_csdn.txt", 0, n=n)
    # return read_file(f"data/{vname}/rockyou/top_rockyou.txt", 0, n=n)


def test_pass(n=3000000):
    return read_file(f"data/{vname}/rockyou/test_rockyou.txt", 0, n=n, count_dict=True)
    # return read_file(f"data/{vname}/test_csdn.txt", 0, n=n, count_dict=True)


chunk_map = {}


def get_chunk(passwd, chunk_len=2, num=None):
    global chunk_map
    # 从分词之后的文件中读取 密码以及其对应的分词存到字典。
    if len(chunk_map) == 0:
        # with open(f"data/{vname}/chunk_csdn.txt") as f:
        # with open(f"data/{vname}/chunk_csdn_level.txt") as f:
        with open(f"data/{vname}/rockyou/chunk_rockyou_baseline.txt") as f:
            passwds = f.readlines()
            if num is not None:
                passwds = passwds[:num]
            progress = tqdm.tqdm(passwds)
            for wd in progress:
                key = wd.replace(" ", "")
                value = wd.split()
                chunk_map[key.strip()] = value
    # 字典里有该密码
    if passwd in chunk_map:
        return chunk_map[passwd]
    sen = []
    # 滑动窗口取chunk， 窗口长度为chunk_len
    for i in range(len(passwd) - chunk_len):
        ch = passwd[i:i + chunk_len]
        sen.append(ch)
    return sen


def get_pass_score(chunk_prob, passwd_chunks):
    prob = 100000000000
    for k in passwd_chunks:
        if k in chunk_prob:
            prob *= chunk_prob[k]
        else:
            prob *= 0.0001
    return prob


def get_chunk_ind(ordered_chunk, pass_chunks):
    res = []
    max_id = 1000100
    for c in pass_chunks:
        if c not in ordered_chunk:
            res.append(max_id)
            max_id -= 1
            continue
        ind = ordered_chunk.index(c)
        res.append(ind)
    return res

# 使用种子口令直接在测试集中猜
def base_line():
    m = Word2Vec.load(f"model/{vname}/csdn_fasttext.model")
    # m = Word2Vec.load(f"model/{vname}/csdn.model")
    top_ch = top_chunk()
    seeds = top_pass()
    test_pass_dict = test_pass()

    unique_count = 0
    count = 0
    all_pass_num = 0
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    for pass_ind, passwd in enumerate(seeds):
        # 长度小于5的不处理
        if len(passwd) < 5:
            continue
        newps = passwd
        all_pass_num += 1
        if newps in test_pass_dict:
            count += test_pass_dict[newps]
            unique_count += 1
        if pass_ind % 100 == 0:
            print(f"{count}\t{unique_count}\t{all_pass_num}")

    print(f"{count}\t{unique_count}\t{all_pass_num}")
#     298842	67314	499185


def load_center():
    res = []
    with open("data/kmeans_10.txt") as f:
        d = f.read()
        d = d.split("]\n [")
        for k in d:
            k = k.split()
            k = [float(i) for i in k]
            res.append(k)
    return res


def get_word_vector(model, chunks):
    passwd_vec = [0 for k in range(100)]
    for ch in chunks:
        if ch not in model.wv:
            continue
        s = model.wv[ch]
        passwd_vec += s
    return passwd_vec


def distance(centers, point):
    def numpy_euclidian_distance(point_1, point_2):
        array_1, array_2 = np.array(point_1), np.array(point_2)
        squared_distance = np.sum(np.square(array_1 - array_2))
        d = np.sqrt(squared_distance)
        return d

    mind = 100000000
    for c in centers:
        d = numpy_euclidian_distance(c, point)
        if d < mind:
            mind = d

    # print(mind)
    return mind


def juelei():
    from matplotlib import pyplot
    from sklearn.decomposition import PCA
    import sklearn.cluster as sc

    get_chunk("passwd")
    m = Word2Vec.load(f"model/{vname}/csdn.model")
    seeds = top_pass()
    # seeds = seeds[:10000]   # for testing
    progress = tqdm.tqdm(seeds)

    # get_chunk("passwd", num=10000) # for testing
    all_vec = np.ndarray(shape=(len(seeds), 100))
    for ind, passwd in enumerate(progress):
        if len(passwd) < 5:
            continue
        chunks = get_chunk(passwd)
        passwd_vec = [0 for k in range(100)]
        for ch in chunks:
            if ch not in m.wv:
                continue
            s = m.wv[ch]
            passwd_vec += s
        all_vec[ind] = passwd_vec

    X = all_vec
    model = sc.KMeans(n_clusters=10)# k-means 聚类
    # 模型拟合与聚类预测
    # 模型拟合
    # 为每个示例分配一个集群
    yhat = model.fit_predict(X)
    # new1查看各个类数量
    print(np.unique(yhat, return_counts=True))
    # 检索唯一群集
    clusters = np.unique(yhat)
    # 为每个群集的样本创建散点图

    for cluster in clusters:
        # 获取此群集的示例的行索引
        row_ix = np.where(yhat == cluster)
        # print(seeds[row_ix])
        # 创建这些样本的散布
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # 绘制散点图
    pyplot.show()
    print(model.cluster_centers_)


    # X = all_vec
    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)
    # # create a scatter plot of the projection
    # pyplot.scatter(result[:, 0], result[:, 1])
    # words = seeds
    # for i, word in enumerate(words):
    #     # $ 显示会出问题
    #     if '$' in word:
    #         continue
    #     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()

def get_cross_entropy(chunks_prob, seed_chunks, newp_chunks):
    s = len(seed_chunks)
    h = 0.0
    for i in range(0, s):
        if seed_chunks[i] in chunks_prob:
            seedp = chunks_prob[seed_chunks[i]]
        else:
            seedp = 0.0000001
        if newp_chunks[i] in chunks_prob:
            newp = chunks_prob[newp_chunks[i]]
        else:
            newp = 0.0000001
        h -= seedp * math.log(newp)
    return h


# 生成猜测的口令集, 放到vec_guess.txt
def compute(model_name, model):
    chunk_similar_map = {}

    def get_sorted_similar(chunk, num):
        if chunk in chunk_similar_map:
            return [k for k in chunk_similar_map[chunk]]
        similar = m.wv.similar_by_word(chunk, num * 5)
        chunk_replace = [[similar[k][0], 1000000] for k in range(num*5)]
        for c in chunk_replace:
            if c[0] in top_ch:
                c[1] = top_ch.index(c[0])
        chunk_replace = sorted(chunk_replace, key=lambda x: x[1])
        #  取出现频率排名在一千以内的
        chunk_replace = [k[0] for k in chunk_replace if k[1] < 100000]
        # 加入到字典中，方便下次直接查询
        chunk_similar_map[chunk] = chunk_replace
        return [k for k in chunk_replace]

    if model == 'word2vec':
        m = Word2Vec.load(f"model/{vname}/word2vec/{model_name}.model")
        f = open(f"data/{vname}/word2vec/vec_guess-{model_name}.txt", "w")
    if model == 'fasttext':
        m = FastText.load(f"model/{vname}/fasttext/{model_name}.model")
        f = open(f"data/{vname}/fasttext/vec_guess-{model_name}.txt", "w")
    if model == 'glove':
        m = KeyedVectors.load_word2vec_format(f'./model/{vname}/glove/{model_name}.txt', binary=False)
        f = open(f"data/{vname}/glove/vec_guess-{model_name}.txt", "w")


    top_ch = top_chunk(10000)

    chunk_prob = top_chunk_prob()

    seeds = top_pass(1000000)
    # seeds = seeds[:200000]  # 一共就 400多万
    # seeds = ['cheng1203', 'my723628', 'liuxiang123']
    # kmeans_centers = load_center()
    test_pass_dict = {}
    if use_online_crack_info:
        test_pass_dict = test_pass()

    # unique_count = 0
    # count = 0
    guess_set = set()
    guess_list = []
    # 首次读取chunk，会进行一些初始化
    get_chunk("passwd")

    # if add_train_pass:
    #     for passwd in seeds:
    #         if passwd not in guess_set:
    #             # 将训练口令加入猜测集
    #             guess_set.add(passwd)
    #             guess_list.append(passwd)


    # f = open(f"data/{vname}/vec_guess-{fname}.txt", "w")

    # print("guessing passwords")
    progress = tqdm.tqdm(seeds)
    for passwd in progress:
        if len(passwd) < 5:
            continue

        # 将种子口令也加入猜测集
        if add_train_pass and passwd not in guess_set:
            guess_set.add(passwd)
            guess_list.append(passwd)

        # 获取当前密码的chunk
        chunks = get_chunk(passwd)
        # chunks 对应的下标
        inds = get_chunk_ind(top_ch, chunks)
        # 最大表示最不可能出现，所以需要替换掉
        # 最小表示出现可能性大，其相似的 chunk 也可能出现
        # 实验表明 最大要好于最小
        if len(inds) > 1:
            # 降序排列， 对于当前口令所得到的chunks，取在所有chunk中出现频率排名前二的。
            inds_copy = sorted(inds, reverse=True)
            maxi = inds_copy[0]
            maxi2 = inds_copy[1]
            mini = inds_copy[-1]
            max_ind = inds.index(maxi)
            max_ind2 = inds.index(maxi2)
            min_ind = inds.index(mini)
        else:
            # 只有一个chunk
            continue
        similar_num = 35
        # similar_num = 15
        for ind in {max_ind, max_ind2}:
            ch = chunks[ind]
            # 要替换的chunk不在top_chunk里，认为该chunk不应被替换
            if ch not in top_ch:
                continue

            # 每 100000 次，相似的多一些
            if len(guess_set) > 10 and len(guess_set) % 100000 == 0:
                similar_num += 5
            # 多计算一倍
            use_additional = False
            # s = m.wv.similar_by_word(ch, similar_num*2)
            # ch_replace = [s[k][0] for k in range(similar_num)]
            # 获取相似chunks
            ch_replace = get_sorted_similar(ch, similar_num)
            # 依据生成的chunk对原口令进行替换，从而生成新口令
            while len(ch_replace) > 0:
                ch_rep = ch_replace.pop(0)
                newps = passwd.replace(ch, ch_rep)
                # 新口令已经存在
                if newps in guess_set:
                    continue
                #  新口令比原口令长太多了，忽略
                if len(newps) > 30 and len(passwd) < 20:
                    continue
                # 处理新口令的chunks得分
                newps_chunks = []
                for c in chunks:
                    if c == ch:
                        newps_chunks.append(ch_rep)
                    else:
                        newps_chunks.append(c)
                #  计算得分
                passwd_score = get_pass_score(chunk_prob, newps_chunks)
                # passwd_score = get_cross_entropy(chunk_prob, chunks, newps_chunks)
                # #  计算距离，用距离筛选一些口令，实验效果不好，先留到这
                # chunks[ind] = s[k][0]
                # newps_vec = get_word_vector(m, chunks)
                # passwd_dist = distance(kmeans_centers, newps_vec)
                # if passwd_dist > 20:
                #     continue

                # 加入新的猜测口令
                guess_set.add(newps)
                # 格式化方式存储
                # guess_list.append(f"{newps}\t{passwd}\t{ch}\t{ch_rep}\t{passwd_score:.5f}")
                f.write(f"{newps}\t{passwd}\t{ch}\t{ch_rep}\t{passwd_score:.5f}\n")
                #
                # if not use_additional and use_online_crack_info and newps in test_pass_dict:
                #     # 增加一倍的变形
                #     ch_replace.extend([s[k+similar_num][0] for k in range(similar_num)])
                #     use_additional = True

    # all_guess = '\n'.join(guess_list)

    # f.write(all_guess)
    f.close()

    # all_pass_num = len(guess_set)
    # print(f"all_pass_num: {all_pass_num}")
    # for ind, newps in enumerate(guess_list):
    #     if newps in test_pass_dict:
    #         count += test_pass_dict[newps]
    #         unique_count += 1
    #         print(f"{ind}\t{count}\t{unique_count}\t\t{newps}")

    # print(f"{count}\t{unique_count}\t{all_pass_num}")

# 未猜对的
def pass_not_guessed():

    f = open(f"data/{vname}/vec_guess.txt")
    guess_list = f.readlines()
    guess_list = guess_list[:9000000]
    guess_set = set()
    progress = tqdm.tqdm(guess_list)
    for newps in progress:
        newps = newps.split()
        guess_set.add(newps[0])
    f.close()
    test_pass_dict = test_pass()

    d = {}

    for newps in test_pass_dict:
        # 保证口令只出现一次
        if newps not in guess_set:
            if newps not in d:
                d[newps] = test_pass_dict[newps]


    for k in d:
        print(f"{k}: {d[k]}")
    # print(f"{count}\t{unique_count}\t{all_pass_num}")

# 猜测测试集
def pass2vec_score(fname, model):
    write_string = ''
    size_count_string = ''
    fwrite = open(f'./result/{fname}_withoutseeds.txt', "w")

    test_pass_dict = test_pass() # 测试口令集，带计数
    count = 0   # 猜中总数
    unique_count = 0    # 去重后的猜中总数
    skip_count = 0

    pass_set = set()    #
    prev_unqiue = 0
    uesd_pass_count = 0     # 猜测次数

    size_count = np.zeros(50, int)

    nk = 0
    v_list = [1, 2, 3, 4, 5, 6, 7, 8, 9,
              10, 20, 30, 40, 50, 60, 70, 80, 90,
              100, 200, 300, 400, 500, 600, 700, 800, 900,
              1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
              10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
              100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
              1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000,
              9500000]

    zxcvbn_score_count = np.zeros(5,int)    # 计算zxcvbn的每个强度分别有多少猜测口令

    # 直接猜种子在测试集中的出现情况
    # seeds = top_pass()
    # for newps in seeds:
    #     # 保证口令只出现一次
    #     if newps in pass_set:
    #         continue
    #
    #     uesd_pass_count += 1 # 用于猜测的口令数量
    #     pass_set.add(newps)
    #     if newps in test_pass_dict:
    #         count += test_pass_dict[newps] # 猜中口令数量
    #         unique_count += 1               # 去重后的猜中口令数量
    #         size_count[len(newps)] += test_pass_dict[newps]
    #         # compute zxcvbn score
    #         score_zxcvbn = zxcvbn(newps)
    #         zxcvbn_score_count[score_zxcvbn['score']]+= test_pass_dict[newps]
    #     if uesd_pass_count == v_list[nk]:   # 每猜一千次,打印一次
    #         for x in size_count:
    #             size_count_string += str(x)
    #             size_count_string += '\t'
    #         size_count_string += '\n'
    #
    #         print(f"{uesd_pass_count}\t{unique_count}\t{unique_count - prev_unqiue}\t{count}")
    #         write_string += f"{uesd_pass_count}\t{unique_count}\t{unique_count - prev_unqiue}\t{count}\n"
    #         prev_unqiue = unique_count
    #         nk += 1


    # 获取猜测集
    # f = open(f"data/{vname}/vec_guess-{fname}.txt")
    # if model == 'word2vec':
    #     f = open(f"data/{vname}/word2vec/vec_guess-{fname}.txt")
    # if model == 'glove':
    #     f = open(f"data/{vname}/glove/vec_guess-{fname}.txt")
    # if model == 'fasttext':
    #     f = open(f"data/{vname}/fasttext/vec_guess-{fname}.txt")
    # if model == 'simbert'
    f = open(f"data/{vname}/{model}/vec_guess-{fname}.txt")
    # 按得分排序进行猜测，先猜测得分高的
    for score in [(100, 10000000000000), (10, 100), (1, 10), (0, 1)]:
        while True:
            line = f.readline()
            # if line is empty end of file is reached
            if not line:
                break
            newps = line.split()
            # 只读有五个部分的行， 比如songwei	zhangwei	zhang	song	26149.88014
            # (f"{newps}\t{passwd}\t{ch}\t{ch_rep}\t{passwd_score:.5f}")
            if len(newps) != 5:
                continue
            # 保证新口令只进行一次猜测，newps[0]是替换chunk的新口令
            if newps[0] in pass_set:
                continue
            ss = float(newps[-1])  # ss表示新口令的得分
            if ss < score[0] or ss > score[1]: #在得分范围之外
                continue

            # 通过psm过滤强度过大的口令,只有一点点效果，可以忽略
            # if score_differ(newps[0], newps[1]) > 0:
            #     skip_count += 1
            #     continue

            newps = newps[0]
            uesd_pass_count += 1 #用于猜测的口令数量
            pass_set.add(newps) # 加入新口令
            # 新口令在测试集中
            if newps in test_pass_dict:
                count += test_pass_dict[newps] # 新口令在测试集中的数量, 加入到总的猜中数里
                unique_count += 1   # 口令唯一数加一
                # 口令长度计数
                size_count[len(newps)] += test_pass_dict[newps]
                # print(f"{ind}\t{count}\t{unique_count}\t\t{newps}")
                # compute zxcvbn score
                score_zxcvbn = zxcvbn(newps)
                zxcvbn_score_count[score_zxcvbn['score']]+= test_pass_dict[newps]
            if nk < len(v_list):
                if uesd_pass_count == v_list[nk]: # 每一千次打印一次
                    for x in size_count:
                        size_count_string += str(x)
                        size_count_string += '\t'
                    size_count_string += '\n'
                    # print(f"{ind}\t{unique_count}\t{count}\t{newps}")
                    print(f"{uesd_pass_count}\t{unique_count}\t{unique_count-prev_unqiue}\t{count}")
                    write_string += f"{uesd_pass_count}\t{unique_count}\t{unique_count - prev_unqiue}\t{count}\n"
                    prev_unqiue = unique_count
                    nk+=1
            if uesd_pass_count > 10000000:
                break
        f.seek(0, 0)
    print(f"tt{uesd_pass_count}")
        # 写结果数据
    fcount = open(f'./result/size_count-{fname}_withoutseeds.txt', 'w')
    fcount.write(size_count_string)
    fcount.close()
    fwrite.write(write_string)
    fwrite.close()
    print(zxcvbn_score_count)

    # print(f"skip count: {skip_count}")
    # print(f"{count}\t{unique_count}\t{all_pass_num}")

def pass2vec_score_sort(fname):
    write_string = ''
    size_count_string = ''
    fwrite = open(f'./result/{fname}-sort-cross.txt', "w")

    test_pass_dict = test_pass()
    count = 0
    unique_count = 0
    skip_count = 0

    pass_set = set()
    guess_set = set()
    prev_unqiue = 0
    uesd_pass_count = 0

    size_count = np.zeros(50, int)

    nk = 0
    v_list = [1, 2, 3, 4, 5, 6, 7, 8, 9,
              10, 20, 30, 40, 50, 60, 70, 80, 90,
              100, 200, 300, 400, 500, 600, 700, 800, 900,
              1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
              10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
              100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
              1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000,
              10000000]

    # 直接猜种子在测试集中的出现情况
    seeds = top_pass()
    for newps in seeds:
        # 保证口令只出现一次
        if newps in pass_set:
            continue

        uesd_pass_count += 1  # 用于猜测的口令数量
        pass_set.add(newps)
        guess_set.add(newps)
        if newps in test_pass_dict:
            count += test_pass_dict[newps]  # 猜中口令数量
            unique_count += 1  # 去重后的猜中口令数量
            size_count[len(newps)] += test_pass_dict[newps]
        if uesd_pass_count == v_list[nk]:  # 每猜一千次,打印一次
            for x in size_count:
                size_count_string += str(x)
                size_count_string += '\t'
            size_count_string += '\n'

            print(f"{uesd_pass_count}\t{unique_count}\t{unique_count - prev_unqiue}\t{count}")
            write_string += f"{uesd_pass_count}\t{unique_count}\t{unique_count - prev_unqiue}\t{count}\n"
            prev_unqiue = unique_count
            nk += 1

    # 获取猜测集
    # f = open(f"data/{vname}/vec_guess-{fname}.txt")
    # f = open(f"data/{vname}/glove/vec_guess-{fname}.txt")
    # f = open(f"data/{vname}/word2vec/vec_guess-{fname}.txt")
    f = open(f"data/{vname}/fasttext/vec_guess-{fname}-cross.txt")

    # sort
    l = []
    while True:
        line = f.readline()
        # if line is empty end of file is reached
        if not line:
            break
        newps = line.split()
        # 只读有五个部分的行， 比如songwei	zhangwei	zhang	song	26149.88014
        # (f"{newps}\t{passwd}\t{ch}\t{ch_rep}\t{passwd_score:.5f}")
        if len(newps) != 5:
            continue
        if newps[0] in guess_set:
            continue
        ss = float(newps[-1])
        p = newps[0]
        newpassword = [p,ss]
        guess_set.add(p)  # 加入新口令
        l.append(newpassword)
    sorted(l, key=lambda x:x[1], reverse=True)

    # guess
    for guess in l:
        pwd = guess[0]
        if pwd in pass_set:
            continue
        uesd_pass_count += 1  # 用于猜测的口令数量
        pass_set.add(pwd)  # 加入新口令
        # 新口令在测试集中
        if pwd in test_pass_dict:
            count += test_pass_dict[pwd]  # 新口令在测试集中的数量, 加入到总的猜中数里
            unique_count += 1  # 口令唯一数加一
            # 口令长度计数
            size_count[len(pwd)] += test_pass_dict[pwd]
        if nk <= len(v_list):
            if uesd_pass_count == v_list[nk]:  # 每一千次打印一次
                for x in size_count:
                    size_count_string += str(x)
                    size_count_string += '\t'
                size_count_string += '\n'
                print(f"{uesd_pass_count}\t{unique_count}\t{unique_count - prev_unqiue}\t{count}")
                write_string += f"{uesd_pass_count}\t{unique_count}\t{unique_count - prev_unqiue}\t{count}\n"
                prev_unqiue = unique_count
                nk += 1
        if uesd_pass_count >= 10000000:
            break


        # 写结果数据
    fcount = open(f'./result/size_count-{fname}-all-sort-cross.txt', 'w')
    fcount.write(size_count_string)
    fcount.close()
    fwrite.write(write_string)
    fwrite.close()

def pcfg():
    # pcfg = read_file("guess.txt", item_ind=0, n=10000)
    test_pass_dict = test_pass()
    count = 0
    unique_count = 0
    # pcfg_passwd = open(f"../pcfg_cracker/guess_rockyou_{vname}.txt")
    passgan_passwd = open(f"./Passgan/gens_rockyou_PASSGAN_l25.txt")
    size_count = np.zeros(50, int)
    size_count_string = ''
    zxcvbn_score_count = np.zeros(5, int)

    write_str = ''
    # res = open("./result/pcfg_result_rockyou.txt","w")
    res = open("./result/passgan_result_rockyou.txt","w")
    nk = 0
    v_list = [1, 2, 3, 4, 5, 6, 7, 8, 9,
              10, 20, 30, 40, 50, 60, 70, 80, 90,
              100, 200, 300, 400, 500, 600, 700, 800, 900,
              1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
              10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
              100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
              1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000,
              9500000]

    pset = set()
    plist = []
    # 保证唯一性
    # for p in pcfg_passwd:

    for p in passgan_passwd:
        p = p.strip()
        if len(p) == 0:
            continue
        if p not in pset:
            pset.add(p)
            plist.append(p)

    num = 0
    lenp = 0
    for p in plist:
        p = p.strip()
        if len(p) == 0:
            continue
        num += 1
        if p in test_pass_dict:
            count += test_pass_dict[p]
            unique_count += 1
            score_zxcvbn = zxcvbn(p)
            zxcvbn_score_count[score_zxcvbn['score']] += test_pass_dict[p]
            if len(p) < 50:
                lenp = max(lenp, len(p))
                size_count[len(p)] += test_pass_dict[p]
        if nk < len(v_list) and  num == v_list[nk] :
            for x in size_count:
                size_count_string += str(x)
                size_count_string += '\t'
            size_count_string += '\n'
            write_str+=f"{num}\t{unique_count}\t00\t{count}\n"
            print(f"{num}\t{unique_count}\t{count}")
            nk += 1
        if num > 9500000:
            break
#         9705000	129347	360483
    res.write(write_str)
    res.close()
    # size_f = open('./result/size_count-pcfg-rockyou.txt','w')
    size_f = open('./result/size_count-passgan-rockyou.txt','w')
    size_f.write(size_count_string)
    size_f.close()
    print(zxcvbn_score_count)
    print(lenp)
    # passgan zxcvbn: 128758 157321   4651     34      0


def score_differ(passwd1, passwd2):
    r1 = zxcvbn(passwd1)
    r2 = zxcvbn(passwd2)
    return r1['score'] - r2['score']


if __name__ == '__main__':
    # r = score_differ("123@11123@09", 'complicatedcomplicatedcomplicatedcomplicated')
    # print(r)
    # c = top_chunk_prob()
    # print(c)
    #
    # c = top_pass()
    # print(c)
    # base_line() #fasttext:298842	67314	499185

    #  log_out

    # for mini_count in (10, 20):

    size = 10
    window = 3
    iter = 50
    mini_count = 5
    model = 'simbert'
    fname = f'rockyou-{model}-level'
    pass2vec_score(fname, model)

    # for model in ['fasttext', 'glove', 'word2vec']:
    #     print('-------------------------------------------------------------------------------\n')
    #     fname = f'rockyou-{model}-s{size}-w{window}-e{iter}-m{mini_count}_baseline'
    #     print(fname)
    #     # chunc2vec.chunk2vec_fasttext(f'data/{vname}/rockyou/chunk_rockyou_level.txt', size, window, mini_count, iter, fname)
    #     # chunc2vec.chunk2vec(f'data/{vname}/rockyou/chunk_rockyou_level.txt',fname)
    #     print('\nchun2vec done!\n')
    #
    #     compute(fname, model)
    #     print('compute done \n')
    #     pass2vec_score(fname, model)  # fasttext:9800000	103384	7	337487
    #
    #     # pass2vec_score_sort(fname)
    #     print('pass2vec_score done \n')
    #     print('-------------------------------------------------------------------------------\n')

    # compute('')
    # fname = f'glove-s{size}-w{window}-e{iter}-m{mini_count}_level'
    # # [ 422825 1143264  140382   23932    3137]
    # pass2vec_score(fname, 'glove')
    # pass_not_guessed()
    # pcfg()
    # [374313 621612  49670   5077     77]
    # juelei()

