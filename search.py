"""常见搜索算法模板。

图使用邻接表表示，例如：
graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
"""

from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TypeVar


Node = TypeVar("Node", bound=Hashable)
Item = TypeVar("Item")


def bfs(graph: Mapping[Node, Iterable[Node]], start: Node) -> list[Node]:
    """广度优先搜索（BFS），返回从 start 可到达节点的访问顺序。

    实现思路：使用队列保存“下一层”待访问的节点。每取出一个节点，就把尚未
    访问过的相邻节点加入队尾，因此总是先访问距离起点更近的节点。

    时间复杂度：O(V + E)，V 为可达节点数，E 为这些节点之间的边数。
    空间复杂度：O(V)，用于队列和 visited 集合。
    """
    visited = {start}
    queue: deque[Node] = deque([start])
    order: list[Node] = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph.get(node, ()):
            if neighbor not in visited:
                visited.add(neighbor)  # 入队时标记，避免同一节点重复入队。
                queue.append(neighbor)

    return order


def bfs_shortest_path(
    graph: Mapping[Node, Iterable[Node]], start: Node, target: Node
) -> list[Node] | None:
    """在无权图中用 BFS 找 start 到 target 的最短路径；不可达时返回 None。

    实现思路：BFS 第一次访问到的节点，经过的边数一定最少。用 parents 记录
    每个节点从哪个前驱到达，找到 target 后即可从终点反向还原路径。

    时间复杂度：O(V + E)。
    空间复杂度：O(V)。
    """
    parents: dict[Node, Node | None] = {start: None}
    queue: deque[Node] = deque([start])

    while queue:
        node = queue.popleft()
        if node == target:
            path: list[Node] = []
            current: Node | None = target

            while current is not None:
                path.append(current)
                current = parents[current]

            return path[::-1]

        for neighbor in graph.get(node, ()):
            if neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)

    return None


def dfs_recursive(graph: Mapping[Node, Iterable[Node]], start: Node) -> list[Node]:
    """深度优先搜索（递归版），返回从 start 可到达节点的访问顺序。

    实现思路：访问当前节点后，立刻递归访问一个未访问的相邻节点；直到无路可走
    再回退，继续尝试其他分支。

    时间复杂度：O(V + E)。
    空间复杂度：O(V)，包括 visited 集合和最坏 O(V) 深度的递归调用栈。
    """
    visited: set[Node] = set()
    order: list[Node] = []

    def visit(node: Node) -> None:
        visited.add(node)
        order.append(node)

        for neighbor in graph.get(node, ()):
            if neighbor not in visited:
                visit(neighbor)

    visit(start)
    return order


def dfs_iterative(graph: Mapping[Node, Iterable[Node]], start: Node) -> list[Node]:
    """深度优先搜索（迭代版），返回从 start 可到达节点的访问顺序。

    实现思路：用栈替代递归调用栈。每次弹出栈顶节点，并把未访问的相邻节点压栈，
    后压入的节点会先被访问，体现“优先向深处搜索”。

    时间复杂度：O(V + E)。
    空间复杂度：O(V)，用于栈和 visited 集合。
    """
    visited: set[Node] = set()
    stack = [start]
    order: list[Node] = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        for neighbor in graph.get(node, ()):
            if neighbor not in visited:
                stack.append(neighbor)

    return order


def binary_search(numbers: Sequence[int], target: int) -> int:
    """二分查找，返回 target 的一个下标；不存在时返回 -1。

    使用前提：numbers 必须已经按升序排列。每轮比较中间元素，并丢弃不可能包含
    target 的一半区间，因此搜索范围会持续减半。

    时间复杂度：O(log n)。
    空间复杂度：O(1)。
    """
    left = 0
    right = len(numbers) - 1

    while left <= right:
        middle = (left + right) // 2

        if numbers[middle] == target:
            return middle
        if numbers[middle] < target:
            left = middle + 1
        else:
            right = middle - 1

    return -1


def permutations_backtracking(items: Sequence[Item]) -> list[list[Item]]:
    """回溯法模板：生成 items 的所有排列。

    实现思路：path 保存当前已作出的选择，used 记录已选元素。每一层尝试一个
    尚未使用的元素；递归返回后撤销本层选择，再尝试下一个选择。这正是回溯的
    “选择 -> 递归 -> 撤销选择”模板。处理组合、N 皇后等问题时，替换终止条件、
    可选项和剪枝条件即可。

    时间复杂度：O(n * n!)，共有 n! 个排列，复制每个结果需要 O(n)。
    空间复杂度：O(n)（不计输出结果），用于 path、used 和递归调用栈。
    """
    result: list[list[Item]] = []
    path: list[Item] = []
    used = [False] * len(items)

    def backtrack() -> None:
        if len(path) == len(items):
            result.append(path.copy())
            return

        for index, item in enumerate(items):
            if used[index]:
                continue

            # 选择：将 item 加入当前解。
            used[index] = True
            path.append(item)

            # 递归：继续构造下一层选择。
            backtrack()

            # 撤销选择：恢复状态，使下一次循环能尝试其他分支。
            path.pop()
            used[index] = False

    backtrack()
    return result


if __name__ == "__main__":
    demo_graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": ["F"],
        "F": [],
    }

    print("BFS:", bfs(demo_graph, "A"))
    print("BFS shortest path:", bfs_shortest_path(demo_graph, "A", "F"))
    print("DFS recursive:", dfs_recursive(demo_graph, "A"))
    print("DFS iterative:", dfs_iterative(demo_graph, "A"))
    print("Binary search:", binary_search([1, 3, 5, 7, 9], 7))
    print("Backtracking:", permutations_backtracking([1, 2, 3]))
