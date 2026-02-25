import heapq as hp


def astar(grid, start, goal):
    rows = len(grid)
    cols = len(grid[0])

    def in_bounds(row, col):
        return 0 <= row < rows and 0 <= col <cols

    def is_open(row, col):
        return grid[row][col] == 0

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    came_from = {start: None}
    g_score = {start: 0}
    hp.heappush(open_set, (heuristic(start, goal), 0, start))

    while open_set:
        _, _, current = hp.heappop(open_set)
        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        row, col = current
        neighbors = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        for next_cell in neighbors:
            next_row, next_col = next_cell
            if not in_bounds(next_row, next_col):
                continue
            if not is_open(next_row, next_col):
                continue

            new_g = g_score[current] + 1
            if next_cell not in g_score or new_g < g_score[next_cell]:
                #fixed:keep only shorter routes to each cell
                came_from[next_cell] = current
                g_score[next_cell] = new_g
                new_f = new_g + heuristic(next_cell, goal)
                hp.heappush(open_set, (new_f, new_g, next_cell))

    return None
