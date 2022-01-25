# fast preprocessing function on dataset folder


def check_contain_chinese(check_str: str):
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def load_stop_train(stop_word_path: str):
    stop_dict = {}
    for line in open(stop_word_path).readlines():
        line = line.strip()
        stop_dict[line] = 1
    return stop_dict


def open_file(file: str, encoding: str):
    res = []
    with open(file, encoding=encoding) as f:
        for _ in f.readlines():
            res.append(_.strip())
    return res


def load_data_from_index(index: str, root_folder: str):
    with open(index, encoding='utf-8') as f:
        _files = f.readlines()
    for _ in range(len(_files)):
        tag_script = _files[_].strip().replace('../', root_folder).split(' ')
        try:
            words = open_file(tag_script[1], encoding='gbk')
        except UnicodeDecodeError:
            continue
            try:
                words = open_file(tag_script[1], encoding='utf-8')  # for English script
            except UnicodeDecodeError:
                continue
        yield words, tag_script[0]


build_splitted_scripts_and_dictionary = False
if build_splitted_scripts_and_dictionary:
    import jieba
    import gensim
    script, label = [], []
    stop_words = load_stop_train('../dataset/ts/stop_words.txt')
    for item in load_data_from_index('../dataset/trec06c/full/index', '../dataset/trec06c/'):
        temp = []
        for _ in range(len(item[0])):
            seg = jieba.cut(item[0][_], cut_all=True)
            for word in seg:
                if not check_contain_chinese(word) or word in stop_words:
                    continue
                else:
                    temp.append(word)
        script.append(temp)
        label.append(item[1])
    gensim.corpora.Dictionary(script).save('../dataset/ts/dictionary')  # build dictionary


if __name__ == '__main__':
    pass
