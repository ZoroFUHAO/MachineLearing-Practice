"""Pandas 入门示例，以及数据处理类机试/OA 模板。

提醒：传统算法机试通常不提供 pandas，必须先确认评测环境允许导入。此文件后半部分
更适合数据分析笔试、数据工程 OA 和本地数据处理脚本。
"""

from __future__ import annotations

from io import StringIO
import sys

import pandas as pd


# ------------------------------ 入门示例 ------------------------------

def simple_examples() -> dict[str, object]:
    """返回常用 Pandas 操作的结果，阅读或交互运行时可逐项查看。

    内容包括：创建 DataFrame、选列/选行、条件过滤、新列、缺失值、聚合、合并。
    """
    students = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "David"],
            "class": ["A", "A", "B", "B"],
            "score": [91, 78, None, 88],
        }
    )

    # 1. 选择：df["col"] 选一列；loc 按标签选行列；iloc 按位置选行列。
    names = students["name"]
    first_two_rows = students.iloc[:2]
    selected_columns = students.loc[:, ["name", "score"]]

    # 2. 过滤：布尔条件产生布尔 Series，再用它筛行。
    passed = students[students["score"] >= 60]

    # 3. 清洗/新列：fillna 填缺失值，assign 返回带新列的新 DataFrame。
    cleaned = students.copy()
    cleaned["score"] = cleaned["score"].fillna(cleaned["score"].median())
    cleaned = cleaned.assign(passed=cleaned["score"] >= 60)

    # 4. groupby + agg：按班级计算人数、平均分和最高分。
    class_summary = (
        cleaned.groupby("class", as_index=False)
        .agg(student_count=("name", "count"), mean_score=("score", "mean"), max_score=("score", "max"))
        .sort_values("class")
    )

    # 5. merge：按共同键把两张表横向连接，类似 SQL JOIN。
    city = pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "city": ["北京", "上海", "深圳"]})
    student_with_city = cleaned.merge(city, on="name", how="left")

    return {
        "students": students,
        "names": names,
        "first_two_rows": first_two_rows,
        "selected_columns": selected_columns,
        "passed": passed,
        "cleaned": cleaned,
        "class_summary": class_summary,
        "student_with_city": student_with_city,
    }


# ------------------------------ 数据清洗模板 ------------------------------

def clean_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    """销售数据清洗模板。

    假定输入列为 order_id、customer、amount、created_at。实际题目中要先阅读字段含义，
    不要不加判断地填充或删除缺失值，否则很容易篡改业务语义。
    """
    required_columns = {"order_id", "customer", "amount", "created_at"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"缺少列：{sorted(required_columns - set(data.columns))}")

    result = data.copy()

    # 1. 字符串规范化：去前后空格，避免 'A001' 与 ' A001 ' 无法去重。
    result["order_id"] = result["order_id"].astype("string").str.strip()
    result["customer"] = result["customer"].astype("string").str.strip()

    # 2. 类型转换：非法金额/日期转为 NaN/NaT，而不是直接报错中断。
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
    result["created_at"] = pd.to_datetime(result["created_at"], errors="coerce")

    # 3. 删除业务主键或关键指标不合法的记录，再根据主键去重。
    result = result.dropna(subset=["order_id", "amount", "created_at"])
    result = result.drop_duplicates(subset=["order_id"], keep="last")

    # 4. 示例规则：负金额视为退款则保留；若业务不允许负数，应在此处过滤或单独处理。
    return result.reset_index(drop=True)


def top_n_per_group(
    data: pd.DataFrame, group_column: str, score_column: str, n: int
) -> pd.DataFrame:
    """机试高频：每组取分数最高的 n 行，分数相同按原始行顺序。

    核心套路：先稳定排序（kind="stable"），再 groupby(...).head(n)。
    不要对每一组写 Python for 循环，Pandas 的分组操作通常更清晰也更快。
    """
    if n <= 0:
        raise ValueError("n 必须为正整数")
    if group_column not in data.columns or score_column not in data.columns:
        raise ValueError("分组列或分数列不存在")

    return (
        data.sort_values([group_column, score_column], ascending=[True, False], kind="stable")
        .groupby(group_column, group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def user_daily_summary(logs: pd.DataFrame) -> pd.DataFrame:
    """机试/OA 模板：统计每个用户每天的事件数、总金额、平均金额。

    输入列：user_id、timestamp、amount。此题覆盖三件常考能力：时间解析、派生日期列、
    多指标 groupby 聚合。非法时间和金额会被剔除。
    """
    required_columns = {"user_id", "timestamp", "amount"}
    if not required_columns.issubset(logs.columns):
        raise ValueError(f"缺少列：{sorted(required_columns - set(logs.columns))}")

    result = logs.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
    result = result.dropna(subset=["user_id", "timestamp", "amount"])
    result["date"] = result["timestamp"].dt.date

    return (
        result.groupby(["user_id", "date"], as_index=False)
        .agg(
            event_count=("amount", "size"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
        )
        .sort_values(["user_id", "date"])
        .reset_index(drop=True)
    )


def pivot_monthly_report(data: pd.DataFrame) -> pd.DataFrame:
    """机试/OA 模板：把“城市-月份-金额”的长表转为交叉报表。

    输入列：city、month、amount；输出行是城市、列是月份、值是金额总和。pivot_table
    是 Excel 透视表和 SQL 条件聚合的 Pandas 对应写法。
    """
    required_columns = {"city", "month", "amount"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"缺少列：{sorted(required_columns - set(data.columns))}")

    result = data.copy()
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
    result = result.dropna(subset=["city", "month", "amount"])
    return pd.pivot_table(
        result,
        index="city",
        columns="month",
        values="amount",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()


# ------------------------------ 从标准输入读取 CSV 的完整解法 ------------------------------

def solve_group_top_2_from_stdin() -> None:
    """完整机试解法：从标准输入读 CSV，并输出每组分数前两名。

    输入格式（CSV，第一行是表头）：
        group,name,score
        A,Alice,91
        A,Bob,88
        A,Carol,95
        B,David,80

    输出也是 CSV。运行方式：
        python pandas_templates.py top2 < input.csv
    """
    data = pd.read_csv(sys.stdin)
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["group", "name", "score"])
    answer = top_n_per_group(data, "group", "score", 2)
    answer.to_csv(sys.stdout, index=False)


def solve_daily_summary_from_stdin() -> None:
    """完整机试解法：从标准输入读事件 CSV，输出用户每日汇总。

    输入列必须为 user_id,timestamp,amount；运行方式：
        python pandas_templates.py daily < logs.csv
    """
    logs = pd.read_csv(sys.stdin)
    answer = user_daily_summary(logs)
    answer.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "top2":
        solve_group_top_2_from_stdin()
    elif len(sys.argv) == 2 and sys.argv[1] == "daily":
        solve_daily_summary_from_stdin()
    else:
        # 简单示例：直接运行可查看分组聚合结果。
        examples = simple_examples()
        print("按班级汇总：")
        print(examples["class_summary"])

        # 内存中的 CSV 示例；真实场景可改为 pd.read_csv("data.csv")。
        demo_csv = StringIO("user_id,timestamp,amount\nu1,2026-08-01 10:00,12\nu1,2026-08-01 12:00,8\nu2,2026-08-02 09:00,20\n")
        print("\n用户每日汇总：")
        print(user_daily_summary(pd.read_csv(demo_csv)))
