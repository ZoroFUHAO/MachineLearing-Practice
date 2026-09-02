"""常见排序算法的教学实现。

每个函数都接收一个整数列表，并返回一个新的升序列表；传入的列表不会被修改。
"""


def quick_sort(numbers: list[int]) -> list[int]:
    """快速排序。

    实现思路：选取一个基准值（pivot），把小于等于基准值的元素放到左边，
    大于基准值的元素放到右边；基准值就位后，再分别处理左右两个区间。

    时间复杂度：平均 O(n log n)，最坏 O(n^2)（每次划分都极不均衡时）。
    空间复杂度：平均 O(log n)，最坏 O(n)（递归调用栈）；本实现原地划分，
    除结果副本外不额外创建与输入等长的数组。
    """
    result = numbers.copy()

    def partition(left: int, right: int) -> int:
        """以右端元素为基准，将区间划分成两个部分。"""
        pivot = result[right]
        smaller_end = left

        for current in range(left, right):
            if result[current] <= pivot:
                result[smaller_end], result[current] = (
                    result[current],
                    result[smaller_end],
                )
                smaller_end += 1

        result[smaller_end], result[right] = result[right], result[smaller_end]
        return smaller_end

    def sort_range(left: int, right: int) -> None:
        if left >= right:
            return

        pivot_index = partition(left, right)
        sort_range(left, pivot_index - 1)
        sort_range(pivot_index + 1, right)

    sort_range(0, len(result) - 1)
    return result


def merge_sort(numbers: list[int]) -> list[int]:
    """归并排序。

    实现思路：不断把序列二分，直到每个子序列只剩一个元素；再把两个已有序的
    子序列按从小到大的顺序合并，最终得到完整有序序列。

    时间复杂度：最好、平均、最坏均为 O(n log n)。
    空间复杂度：O(n)，合并时需要临时数组。
    """
    if len(numbers) <= 1:
        return numbers.copy()

    middle = len(numbers) // 2
    left = merge_sort(numbers[:middle])
    right = merge_sort(numbers[middle:])
    merged: list[int] = []
    left_index = 0
    right_index = 0

    # 两个子序列均有序，每次取更小的当前元素即可保持结果有序。
    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1

    # 将尚未取完的一侧直接追加；该侧本身已经有序。
    merged.extend(left[left_index:])
    merged.extend(right[right_index:])
    return merged


def selection_sort(numbers: list[int]) -> list[int]:
    """选择排序。

    实现思路：依次确定每个位置应放的元素。在未排序区间中找出最小值，
    再与未排序区间的第一个元素交换。

    时间复杂度：最好、平均、最坏均为 O(n^2)，因为始终要扫描未排序区间。
    空间复杂度：O(1)（不计返回结果的副本）。
    """
    result = numbers.copy()

    for start in range(len(result) - 1):
        min_index = start

        for current in range(start + 1, len(result)):
            if result[current] < result[min_index]:
                min_index = current

        if min_index != start:
            result[start], result[min_index] = result[min_index], result[start]

    return result


def insertion_sort(numbers: list[int]) -> list[int]:
    """插入排序。

    实现思路：把列表左侧维护成有序区间。每次取出下一个元素，从右向左移动
    比它大的元素，直到找到合适位置后将该元素插入。

    时间复杂度：最好 O(n)（原列表已有序），平均和最坏 O(n^2)。
    空间复杂度：O(1)（不计返回结果的副本）。
    """
    result = numbers.copy()

    for current in range(1, len(result)):
        value = result[current]
        insert_at = current - 1

        while insert_at >= 0 and result[insert_at] > value:
            result[insert_at + 1] = result[insert_at]
            insert_at -= 1

        result[insert_at + 1] = value

    return result


def bubble_sort(numbers: list[int]) -> list[int]:
    """冒泡排序。

    实现思路：反复比较相邻元素，若顺序错误就交换；每完成一轮，未排序区间中
    的最大元素会“冒”到右端。因此下一轮可少比较一个位置。

    时间复杂度：最好 O(n)（使用提前结束优化时），平均和最坏 O(n^2)。
    空间复杂度：O(1)（不计返回结果的副本）。
    """
    result = numbers.copy()

    for end in range(len(result) - 1, 0, -1):
        swapped = False

        for current in range(end):
            if result[current] > result[current + 1]:
                result[current], result[current + 1] = (
                    result[current + 1],
                    result[current],
                )
                swapped = True

        # 一轮比较没有交换，说明整个列表已经有序。
        if not swapped:
            break

    return result


if __name__ == "__main__":
    sample = [5, 1, 4, 2, 8, 5, -3]
    algorithms = [quick_sort, merge_sort, selection_sort, insertion_sort, bubble_sort]

    for algorithm in algorithms:
        print(f"{algorithm.__name__}: {algorithm(sample)}")
