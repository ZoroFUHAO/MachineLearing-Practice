"""图论经典算法模板。

约定：
1. 普通图使用邻接表，例如 {"A": [("B", 3), ("C", 1)]}，其中元组为
   (相邻节点, 边权)。无权图可把边权统一视为 1。
2. 边集使用 (起点, 终点, 权重) 的三元组。
3. Kruskal、Prim、割点与桥都针对无向图：一条无向边应在两个端点的邻接表中
   各出现一次；Kruskal 的边集则只写一次。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
import heapq
from itertools import count
from math import inf
from typing import TypeVar


Node = TypeVar("Node", bound=Hashable)
Weight = int | float
WeightedGraph = Mapping[Node, Iterable[tuple[Node, Weight]]]
Edge = tuple[Node, Node, Weight]


def dijkstra(graph: WeightedGraph[Node], start: Node) -> dict[Node, Weight]:
    """Dijkstra 单源最短路：返回 start 到各可达节点的最短距离。

    适用条件：所有边权必须非负；有负权边时应使用 Bellman-Ford。
    思路：每次从小根堆取出当前距离最小的节点，该距离已确定；随后用该节点
    松弛相邻边。堆内可能保留旧距离，弹出时跳过即可。

    时间复杂度：O((V + E) log V)。空间复杂度：O(V)。
    """
    distance: dict[Node, Weight] = {start: 0}
    # counter 避免距离相等时比较不可比较的节点（例如不同类型的节点）。
    sequence = count()
    heap: list[tuple[Weight, int, Node]] = [(0, next(sequence), start)]

    while heap:
        current_distance, _, node = heapq.heappop(heap)
        if current_distance != distance.get(node):
            continue  # 堆中的旧记录。

        for neighbor, weight in graph.get(node, ()):
            if weight < 0:
                raise ValueError("Dijkstra 不适用于负权边")

            new_distance = current_distance + weight
            if new_distance < distance.get(neighbor, inf):
                distance[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, next(sequence), neighbor))

    return distance


def bellman_ford(
    vertices: Iterable[Node], edges: Iterable[Edge[Node]], start: Node
) -> tuple[dict[Node, Weight], bool]:
    """Bellman-Ford 单源最短路，返回 (距离字典, 是否存在可达负环)。

    适用条件：可有负权边。若存在从 start 可达的负权环，最短路径没有定义。
    思路：重复 V - 1 轮遍历全部边并进行松弛；简单路径最多包含 V - 1 条边。
    再进行第 V 轮松弛，若仍能变小，则存在可达负环。

    时间复杂度：O(VE)。空间复杂度：O(V)。
    """
    vertex_list = list(vertices)
    edge_list = list(edges)
    distance: dict[Node, Weight] = {vertex: inf for vertex in vertex_list}
    distance[start] = 0

    # 最多进行 V - 1 轮；若某轮无更新，可提前结束。
    for _ in range(max(0, len(vertex_list) - 1)):
        updated = False
        for source, target, weight in edge_list:
            if distance.get(source, inf) == inf:
                continue
            if distance[source] + weight < distance.get(target, inf):
                distance[target] = distance[source] + weight
                updated = True
        if not updated:
            break

    has_negative_cycle = any(
        distance.get(source, inf) != inf
        and distance[source] + weight < distance.get(target, inf)
        for source, target, weight in edge_list
    )
    return distance, has_negative_cycle


def floyd_warshall(
    distance: Sequence[Sequence[Weight]],
) -> list[list[Weight]]:
    """Floyd-Warshall 多源最短路，返回所有点对最短路矩阵。

    输入是邻接矩阵：distance[i][j] 为 i 到 j 的边权；无边填 inf，对角线填 0。
    思路：依次允许每个节点 k 作为中转点，尝试用 i -> k -> j 更新 i -> j。
    若结果中某个 result[i][i] < 0，表示图中存在负权环。

    时间复杂度：O(V^3)。空间复杂度：O(V^2)。适合点数较少但需任意两点距离的场景。
    """
    result = [list(row) for row in distance]
    size = len(result)
    if any(len(row) != size for row in result):
        raise ValueError("Floyd-Warshall 的输入必须是方阵")

    for middle in range(size):
        for source in range(size):
            if result[source][middle] == inf:
                continue
            for target in range(size):
                through_middle = result[source][middle] + result[middle][target]
                if through_middle < result[source][target]:
                    result[source][target] = through_middle

    return result


class UnionFind:
    """并查集：支持快速维护若干不相交集合的连通性。

    思路：parent 保存每个节点的代表元。find 使用路径压缩，union 使用按大小合并，
    可高效回答“两个节点是否属于同一集合”。

    单次均摊时间复杂度：O(alpha(n))，可近似看作 O(1)。空间复杂度：O(n)。
    """

    def __init__(self, items: Iterable[Node] = ()) -> None:
        self.parent: dict[Node, Node] = {}
        self.size: dict[Node, int] = {}
        for item in items:
            self.add(item)

    def add(self, item: Node) -> None:
        """添加一个独立节点；已存在时不做任何操作。"""
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item: Node) -> Node:
        """查找 item 所在集合的代表元，并进行路径压缩。"""
        if item not in self.parent:
            self.add(item)

        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: Node, second: Node) -> bool:
        """合并两个集合；若原本不连通并成功合并，返回 True。"""
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False

        # 小树接到大树下，以降低树高。
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        return True

    def connected(self, first: Node, second: Node) -> bool:
        """判断两个节点是否连通。"""
        return self.find(first) == self.find(second)


def kruskal_mst(vertices: Iterable[Node], edges: Iterable[Edge[Node]]) -> tuple[Weight, list[Edge[Node]]]:
    """Kruskal 最小生成树，返回 (总权重, 选中的边)。

    适用条件：无向带权连通图。若图不连通，返回的是最小生成森林。
    思路：按边权从小到大枚举；只有当一条边的两个端点尚不连通时才选它，
    这样不会形成环。连通性判断由并查集完成。

    时间复杂度：O(E log E)。空间复杂度：O(V)。
    """
    union_find = UnionFind(vertices)
    chosen: list[Edge[Node]] = []
    total_weight: Weight = 0

    for source, target, weight in sorted(edges, key=lambda edge: edge[2]):
        if union_find.union(source, target):
            chosen.append((source, target, weight))
            total_weight += weight

    return total_weight, chosen


def prim_mst(graph: WeightedGraph[Node], start: Node) -> tuple[Weight, list[Edge[Node]]]:
    """Prim 最小生成树，返回 (总权重, 选中的边)。

    适用条件：无向带权连通图；若图不连通，本函数会抛出 ValueError。
    思路：从 start 开始维护“已选节点”与“未选节点”之间的候选边，每次选择权重
    最小的一条边，把一个新节点加入树中。

    时间复杂度：O(E log V)。空间复杂度：O(V + E)。
    """
    # graph 的键应列出所有节点（包括只有入边的节点）。
    vertices = set(graph)
    vertices.add(start)
    selected = {start}
    sequence = count()
    heap: list[tuple[Weight, int, Node, Node]] = []
    chosen: list[Edge[Node]] = []
    total_weight: Weight = 0

    for neighbor, weight in graph.get(start, ()):
        heapq.heappush(heap, (weight, next(sequence), start, neighbor))

    while heap and len(selected) < len(vertices):
        weight, _, source, target = heapq.heappop(heap)
        if target in selected:
            continue

        selected.add(target)
        chosen.append((source, target, weight))
        total_weight += weight

        for neighbor, next_weight in graph.get(target, ()):
            if neighbor not in selected:
                heapq.heappush(
                    heap, (next_weight, next(sequence), target, neighbor)
                )

    if len(selected) != len(vertices):
        raise ValueError("图不连通，无法得到一棵覆盖所有节点的最小生成树")

    return total_weight, chosen


def topological_sort(graph: Mapping[Node, Iterable[Node]]) -> list[Node] | None:
    """Kahn 算法进行拓扑排序；若有环则返回 None。

    适用条件：有向无环图（DAG）。拓扑序保证每条边 u -> v 中，u 都排在 v 前面。
    思路：反复从入度为 0 的节点中取出一个节点并删除其出边；若最后仍有节点
    未处理，说明它们互相依赖而形成了环。

    时间复杂度：O(V + E)。空间复杂度：O(V)。
    """
    indegree: dict[Node, int] = {}
    for node, neighbors in graph.items():
        indegree.setdefault(node, 0)
        for neighbor in neighbors:
            indegree[neighbor] = indegree.get(neighbor, 0) + 1

    queue: deque[Node] = deque(node for node, degree in indegree.items() if degree == 0)
    order: list[Node] = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph.get(node, ()):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == len(indegree) else None


def dag_shortest_path(graph: WeightedGraph[Node], start: Node) -> dict[Node, Weight]:
    """DAG 单源最短路，返回 start 到各可达节点的最短距离。

    适用条件：图必须为 DAG，边权可以为负。思路：先拓扑排序，再按拓扑序松弛
    每条边；处理某节点时，其所有前驱都已经处理完毕。

    时间复杂度：O(V + E)。空间复杂度：O(V)。
    """
    unweighted_graph: dict[Node, list[Node]] = {}
    for node, neighbors in graph.items():
        neighbor_list = list(neighbors)
        unweighted_graph[node] = [neighbor for neighbor, _ in neighbor_list]

    order = topological_sort(unweighted_graph)
    if order is None:
        raise ValueError("DAG 最短路要求输入图无环")

    distance: dict[Node, Weight] = {start: 0}
    for node in order:
        if node not in distance:
            continue
        for neighbor, weight in graph.get(node, ()):
            new_distance = distance[node] + weight
            if new_distance < distance.get(neighbor, inf):
                distance[neighbor] = new_distance

    return distance


def tarjan_scc(graph: Mapping[Node, Iterable[Node]]) -> list[list[Node]]:
    """Tarjan 算法：求有向图的强连通分量（SCC）。

    强连通分量内任意两点相互可达。思路：DFS 中为节点记录发现时间 index 和
    low-link 值；当 node 的 low-link 等于自身 index 时，它是一个 SCC 的根，
    从栈中弹出直到 node 的所有节点构成一个分量。

    时间复杂度：O(V + E)。空间复杂度：O(V)。
    """
    nodes: set[Node] = set(graph)
    for neighbors in graph.values():
        nodes.update(neighbors)

    next_index = 0
    indices: dict[Node, int] = {}
    low_link: dict[Node, int] = {}
    stack: list[Node] = []
    in_stack: set[Node] = set()
    components: list[list[Node]] = []

    def visit(node: Node) -> None:
        nonlocal next_index
        indices[node] = next_index
        low_link[node] = next_index
        next_index += 1
        stack.append(node)
        in_stack.add(node)

        for neighbor in graph.get(node, ()):
            if neighbor not in indices:
                visit(neighbor)
                low_link[node] = min(low_link[node], low_link[neighbor])
            elif neighbor in in_stack:
                low_link[node] = min(low_link[node], indices[neighbor])

        if low_link[node] == indices[node]:
            component: list[Node] = []
            while True:
                member = stack.pop()
                in_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(component)

    for node in nodes:
        if node not in indices:
            visit(node)

    return components


def tarjan_articulation_points_and_bridges(
    graph: Mapping[Node, Iterable[Node]],
) -> tuple[set[Node], list[tuple[Node, Node]]]:
    """Tarjan 算法：求无向图的割点和桥。

    割点：删除该点后，连通分量数量增加的点；桥：删除该边后，图会变得更不连通。
    思路：DFS 记录每个节点的发现时间 disc 与子树能回到的最早祖先 low。对于
    树边 u -> v：low[v] >= disc[u] 时 u 是割点（根节点另有子树数大于 1 的条件）；
    low[v] > disc[u] 时 (u, v) 是桥。

    时间复杂度：O(V + E)。空间复杂度：O(V)。
    """
    time = 0
    discovery: dict[Node, int] = {}
    low: dict[Node, int] = {}
    articulation_points: set[Node] = set()
    bridges: list[tuple[Node, Node]] = []

    def visit(node: Node, parent: Node | None) -> None:
        nonlocal time
        discovery[node] = time
        low[node] = time
        time += 1
        child_count = 0

        for neighbor in graph.get(node, ()):
            if neighbor == parent:
                continue

            if neighbor not in discovery:
                child_count += 1
                visit(neighbor, node)
                low[node] = min(low[node], low[neighbor])

                if parent is not None and low[neighbor] >= discovery[node]:
                    articulation_points.add(node)
                if low[neighbor] > discovery[node]:
                    bridges.append((node, neighbor))
            else:
                low[node] = min(low[node], discovery[neighbor])

        if parent is None and child_count > 1:
            articulation_points.add(node)

    all_nodes: set[Node] = set(graph)
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    for node in all_nodes:
        if node not in discovery:
            visit(node, None)

    return articulation_points, bridges


if __name__ == "__main__":
    weighted_graph = {
        "A": [("B", 4), ("C", 1)],
        "B": [("D", 1)],
        "C": [("B", 2), ("D", 5)],
        "D": [],
    }
    edges = [("A", "B", 4), ("A", "C", 1), ("C", "B", 2), ("B", "D", 1), ("C", "D", 5)]
    undirected_graph = {
        "A": [("B", 4), ("C", 1)],
        "B": [("A", 4), ("C", 2), ("D", 1)],
        "C": [("A", 1), ("B", 2), ("D", 5)],
        "D": [("B", 1), ("C", 5)],
    }

    print("Dijkstra:", dijkstra(weighted_graph, "A"))
    print("Bellman-Ford:", bellman_ford(weighted_graph, edges, "A"))
    print("Floyd-Warshall:", floyd_warshall([[0, 3, inf], [inf, 0, 2], [4, inf, 0]]))
    print("Kruskal:", kruskal_mst(undirected_graph, edges))
    print("Prim:", prim_mst(undirected_graph, "A"))
    print("Topological sort:", topological_sort({"A": ["C"], "B": ["C"], "C": ["D"], "D": []}))
    print("DAG shortest path:", dag_shortest_path(weighted_graph, "A"))
    print("Tarjan SCC:", tarjan_scc({"A": ["B"], "B": ["C", "D"], "C": ["A"], "D": ["E"], "E": ["D"]}))
    print("Cut points and bridges:", tarjan_articulation_points_and_bridges({"A": ["B"], "B": ["A", "C", "D"], "C": ["B"], "D": ["B", "E"], "E": ["D"]}))
