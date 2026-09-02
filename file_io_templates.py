"""纯 Python 文件、终端输入输出、词频和 TF-IDF 模板。

本文件只允许且只使用以下两个导入：
    import csv
    import math

文件读写依赖 Python 内置 open()，终端输入输出依赖内置 input() 与 print()。
"""

import csv
import math


# ------------------------------ TXT 文件读写 ------------------------------

def read_text(path, encoding="utf-8"):
    """一次性读取整个 TXT 文件；适合中小文件。"""
    with open(path, "r", encoding=encoding) as file:
        return file.read()


def read_nonempty_lines(path, encoding="utf-8"):
    """逐行读取 TXT，去掉每行首尾空白，并跳过空行。"""
    lines = []
    with open(path, "r", encoding=encoding) as file:
        for line in file:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def write_text(path, content, encoding="utf-8"):
    """覆盖写入 TXT；文件存在时，旧内容会被替换。"""
    with open(path, "w", encoding=encoding) as file:
        file.write(content)


def append_text(path, content, encoding="utf-8"):
    """追加写入 TXT；保留已有内容。"""
    with open(path, "a", encoding=encoding) as file:
        file.write(content)


# ------------------------------ 终端输入输出 ------------------------------

def read_ints():
    """读取一行空格分隔的整数，例如输入 '1 2 3'，返回 [1, 2, 3]。"""
    return list(map(int, input().split()))


def read_matrix(rows):
    """读取 rows 行整数矩阵；每行使用空格分隔。"""
    matrix = []
    for _ in range(rows):
        matrix.append(read_ints())
    return matrix


def solve_sum_from_input():
    """机试模板：读入 n 和下一行的 n 个整数，输出它们的和。

    示例输入：
        5
        1 2 3 4 5
    """
    _ = int(input())  # n 可用于校验；这里不影响求和逻辑。
    numbers = read_ints()
    print(sum(numbers))


# ------------------------------ CSV 文件读写 ------------------------------

def read_csv_records(path, encoding="utf-8-sig"):
    """读取 CSV，返回“每行一个字典”的列表。

    utf-8-sig 能兼容许多由 Excel 导出的带 BOM UTF-8 CSV 文件。
    """
    with open(path, "r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def write_csv_records(path, records, fieldnames, encoding="utf-8-sig"):
    """将字典列表写入 CSV；fieldnames 决定列的顺序。"""
    with open(path, "w", encoding=encoding, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def clean_csv_records(
    records,
    required_fields=(),
    deduplicate_by=(),
    missing_value="",
):
    """CSV 清洗模板：去空格、删除关键列缺失的行，并可按指定列去重。

    参数示例：
        clean_csv_records(rows, required_fields=("name",), deduplicate_by=("id",))
    """
    cleaned = []
    seen = set()

    for record in records:
        # 1. None 转为空字符串，所有值统一转字符串并清除两端空格。
        normalized = {}
        for key, value in record.items():
            key = str(key).strip()
            normalized[key] = missing_value if value is None else str(value).strip()

        # 2. 关键字段为空，直接舍弃该行。
        has_missing_required_field = False
        for field in required_fields:
            if not normalized.get(field, ""):
                has_missing_required_field = True
                break
        if has_missing_required_field:
            continue

        # 3. 指定业务主键后，用元组作为集合键实现去重。
        if deduplicate_by:
            unique_key = tuple(normalized.get(field, missing_value) for field in deduplicate_by)
            if unique_key in seen:
                continue
            seen.add(unique_key)

        cleaned.append(normalized)

    return cleaned


def convert_column_to_float(records, column, default=None):
    """把 CSV 某一列转为 float；无法转换时写入 default。"""
    converted = []
    for record in records:
        row = dict(record)
        try:
            # 常见金额格式可能带逗号，例如 '1,234.50'。
            row[column] = float(str(row.get(column, "")).replace(",", ""))
        except ValueError:
            row[column] = default
        converted.append(row)
    return converted


# ------------------------------ 文本清洗、词频、TF-IDF ------------------------------

def normalize_text(text):
    """基础英文文本清洗：小写化、移除标点、合并多余空白。

    不使用 re，而是逐字符判断。中文没有天然空格，若做“词”频统计，需要先自行
    把文本切成词；本函数不会替你完成中文分词。
    """
    characters = []
    previous_is_space = False

    for character in text.lower():
        # 保留字母、数字、下划线及中文字符；标点一律视为分隔符。
        is_chinese = "\u4e00" <= character <= "\u9fff"
        if character.isalnum() or character == "_" or is_chinese:
            characters.append(character)
            previous_is_space = False
        elif not previous_is_space:
            characters.append(" ")
            previous_is_space = True

    return "".join(characters).strip()


def tokenize(text):
    """按空白分词，适合英文或已经人工分好词的文本。"""
    return normalize_text(text).split()


def word_frequency(text, top_k=None):
    """统计词频，返回 {词: 次数}；top_k 可限制只保留频率最高的词。"""
    frequencies = {}
    for word in tokenize(text):
        frequencies[word] = frequencies.get(word, 0) + 1

    if top_k is None:
        return frequencies

    # 频率降序、词典序升序，使并列频率时输出稳定。
    ordered_items = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered_items[:top_k])


def term_frequency(tokens):
    """计算一篇文档的 TF：某词出现次数 / 文档总词数。"""
    if not tokens:
        return {}

    counts = {}
    for word in tokens:
        counts[word] = counts.get(word, 0) + 1

    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


def inverse_document_frequency(documents):
    """计算平滑 IDF：log((N + 1) / (DF + 1)) + 1。

    N 是总文档数；DF 是出现该词的文档数。同词在一篇文档中只能贡献一次 DF。
    """
    document_frequency = {}

    for document in documents:
        for word in set(document):
            document_frequency[word] = document_frequency.get(word, 0) + 1

    document_count = len(documents)
    idf = {}
    for word, frequency in document_frequency.items():
        idf[word] = math.log((document_count + 1) / (frequency + 1)) + 1
    return idf


def tf_idf(texts):
    """计算多篇文本的 TF-IDF，返回“每篇一个 {词: 权重} 字典”的列表。"""
    documents = [tokenize(text) for text in texts]
    idf = inverse_document_frequency(documents)
    result = []

    for document in documents:
        tf = term_frequency(document)
        scores = {}
        for word, frequency in tf.items():
            # TF 表示词在当前文档的重要性，IDF 表示词在所有文档中的稀有程度。
            scores[word] = frequency * idf[word]
        result.append(scores)

    return result


if __name__ == "__main__":
    text = "Python is simple. Python is useful for data processing!"
    print("词频：", word_frequency(text))
    print("TF-IDF：", tf_idf(["python data", "python machine learning", "data analysis"]))

    # 文件读写示例：
    # write_text("example.txt", "第一行\n")
    # append_text("example.txt", "第二行\n")
    # print(read_nonempty_lines("example.txt"))
