"""
面向对象大作业第一阶段
姓名：柳黛鲜
学号：2025214239
"""


import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import copy
import json
import os


# ==========================================
#  Model 层：游戏核心逻辑
# ==========================================

class BoardGame:
    """
    棋类游戏基类，负责通用的状态管理、悔棋、存档等功能
    体现了面向对象的继承和多态
    """
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self, size=15):
        self.size = size
        self.board = [[self.EMPTY for _ in range(size)] for _ in range(size)]
        self.current_player = self.BLACK
        self.history = []  # 栈结构，用于悔棋
        self.game_over = False
        self.winner = None
        self.msg_callback = None  # 用于向UI发送消息的回调

    def set_callback(self, callback):
        self.msg_callback = callback

    def log(self, message):
        if self.msg_callback:
            self.msg_callback(message)
        print(f"[System]: {message}")

    def switch_player(self):
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK

    def save_state(self):
        """保存当前状态到历史栈，用于悔棋"""
        state = {
            'board': copy.deepcopy(self.board),
            'player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner
        }
        self.history.append(state)

    def undo(self):
        """悔棋操作"""
        if not self.history:
            return False, "无棋可悔"

        prev_state = self.history.pop()
        self.board = prev_state['board']
        self.current_player = prev_state['player']
        self.game_over = prev_state['game_over']
        self.winner = prev_state['winner']
        return True, "悔棋成功"

    def is_valid_coord(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def place_stone(self, x, y):
        """抽象方法，由子类实现"""
        raise NotImplementedError

    def check_win(self):
        """抽象方法，由子类实现"""
        raise NotImplementedError

    def to_dict(self):
        """序列化"""
        return {
            'type': self.__class__.__name__,
            'size': self.size,
            'board': self.board,
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner
        }

    def load_from_dict(self, data):
        """反序列化"""
        self.size = data['size']
        self.board = data['board']
        self.current_player = data['current_player']
        self.game_over = data['game_over']
        self.winner = data['winner']
        self.history = []  # 读档后清空悔棋栈


class GomokuGame(BoardGame):
    """五子棋逻辑实现"""

    def place_stone(self, x, y):
        if self.game_over:
            return False, "游戏已结束"

        if not self.is_valid_coord(x, y):
            return False, "坐标不合法"

        if self.board[y][x] != self.EMPTY:
            return False, "此处已有棋子"

        self.save_state()
        self.board[y][x] = self.current_player

        if self.check_win(x, y):
            self.game_over = True
            self.winner = self.current_player
            return True, f"游戏结束，{'黑方' if self.winner == self.BLACK else '白方'}获胜！"

        if self.is_full():
            self.game_over = True
            return True, "棋盘已满，平局！"

        self.switch_player()
        return True, "落子成功"

    def is_full(self):
        for row in self.board:
            if self.EMPTY in row:
                return False
        return True

    def check_win(self, x, y):
        """判断五子连珠"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        color = self.board[y][x]

        for dx, dy in directions:
            count = 1
            # 正向查找
            nx, ny = x + dx, y + dy
            while self.is_valid_coord(nx, ny) and self.board[ny][nx] == color:
                count += 1
                nx, ny = nx + dx, ny + dy

            # 反向查找
            nx, ny = x - dx, y - dy
            while self.is_valid_coord(nx, ny) and self.board[ny][nx] == color:
                count += 1
                nx, ny = nx - dx, ny - dy

            if count >= 5:
                return True
        return False


class GoGame(BoardGame):
    """围棋逻辑实现：包含气、提子、禁着点判断"""

    def __init__(self, size=19):
        super().__init__(size)
        # 简单处理：记录死子数量，用于简单胜负判断
        self.captured_counts = {self.BLACK: 0, self.WHITE: 0}

    def place_stone(self, x, y):
        if self.game_over:
            return False, "游戏已结束"

        # 围棋虚着 (Pass) - x,y = -1, -1 代表虚着
        if x == -1 and y == -1:
            self.save_state()
            self.switch_player()
            return True, f"{'黑方' if self.current_player == self.BLACK else '白方'} 虚着（Pass）"

        if not self.is_valid_coord(x, y):
            return False, "坐标不合法"

        if self.board[y][x] != self.EMPTY:
            return False, "此处已有棋子"

        # 逻辑：
        # 1. 假设落子
        # 2. 检查是否有对方棋子气为0 -> 提子
        # 3. 检查自己气是否为0 -> 若未提子且自己无气，则为自杀（非法）
        # 4. 劫争检查 (Ko Rule) - 简化版：这里暂不深度实现全局同型判断，仅做基础规则

        # 临时模拟棋盘用于判断
        temp_board = copy.deepcopy(self.board)
        temp_board[y][x] = self.current_player
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK

        captured_stones = []

        # 检查四周对手棋子
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if self.is_valid_coord(nx, ny) and temp_board[ny][nx] == opponent:
                # 如果这个对手棋子所在的群没有气了
                if not self.has_liberties(temp_board, nx, ny):
                    captured_stones.extend(self.get_group(temp_board, nx, ny))

        # 移除死子
        for cx, cy in captured_stones:
            temp_board[cy][cx] = self.EMPTY

        # 检查自己是否有气（自杀规则）
        if not self.has_liberties(temp_board, x, y):
            return False, "禁着点：不能自杀"

        # 劫争判断（简单）：不能回到上一步的局面
        if self.history:
            last_board = self.history[-1]['board']
            if temp_board == last_board:
                return False, "打劫：不能立即回提"

        # === 确认落子 ===
        self.save_state()
        self.board = temp_board
        if captured_stones:
            self.captured_counts[self.current_player] += len(captured_stones)
            msg = f"落子成功，提吃 {len(captured_stones)} 子"
        else:
            msg = "落子成功"

        self.switch_player()
        return True, msg

    def get_group(self, board, x, y):
        """BFS获取相连的同色棋子"""
        color = board[y][x]
        group = []
        visited = set()
        queue = [(x, y)]
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited: continue
            visited.add((cx, cy))
            group.append((cx, cy))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if self.is_valid_coord(nx, ny) and board[ny][nx] == color:
                    queue.append((nx, ny))
        return group

    def has_liberties(self, board, x, y):
        """检查某一子所在的群是否有气"""
        color = board[y][x]
        visited = set()
        queue = [(x, y)]
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited: continue
            visited.add((cx, cy))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if self.is_valid_coord(nx, ny):
                    if board[ny][nx] == self.EMPTY:
                        return True  # 只要找到一个空位就有气
                    elif board[ny][nx] == color and (nx, ny) not in visited:
                        queue.append((nx, ny))
        return False

    def count_score(self):
        """
        简化的数子法判断胜负。
        实际围棋规则复杂（数地/数子），此处采用：棋盘上子数 + 提子数
        """
        black_score = sum(row.count(self.BLACK) for row in self.board) + self.captured_counts[self.BLACK]
        white_score = sum(row.count(self.WHITE) for row in self.board) + self.captured_counts[self.WHITE]
        # 贴目（Komidashi），通常白棋+6.5或7.5，此处简化为+0以做演示，或者直接比数量
        return black_score, white_score

    def judge_winner(self):
        b, w = self.count_score()
        self.game_over = True
        if b > w:
            self.winner = self.BLACK
            return f"黑方获胜 (黑:{b}, 白:{w})"
        else:
            self.winner = self.WHITE
            return f"白方获胜 (黑:{b}, 白:{w})"


# ==========================================
#  View & Controller 层：GUI实现
# ==========================================

class GameWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Python对战棋坛 - 五子棋 & 围棋")
        self.root.geometry("900x700")

        # 全局配置
        self.cell_size = 35
        self.margin = 30
        self.game = None  # 当前游戏实例
        self.show_hints = True

        self._init_ui()
        self.start_menu()

    def _init_ui(self):
        # 1. 右侧控制面板
        self.panel = tk.Frame(self.root, width=200, bg="#f0f0f0")
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)

        # 2. 左侧棋盘区域
        self.canvas = tk.Canvas(self.root, bg="#DEB887")  # 木纹色
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # 3. 控制面板组件
        tk.Label(self.panel, text="控制台", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

        self.info_label = tk.Label(self.panel, text="请开始游戏", bg="#f0f0f0", fg="blue", wraplength=180)
        self.info_label.pack(pady=10)

        # 按钮组
        btn_opts = {'width': 15, 'pady': 5}
        tk.Button(self.panel, text="新游戏 (五子棋)", command=lambda: self.start_game('gomoku'), **btn_opts).pack(pady=5)
        tk.Button(self.panel, text="新游戏 (围棋)", command=lambda: self.start_game('go'), **btn_opts).pack(pady=5)

        tk.Frame(self.panel, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=5, pady=10)

        self.btn_pass = tk.Button(self.panel, text="围棋虚着 (Pass)", command=self.pass_move, state=tk.DISABLED, **btn_opts)
        self.btn_pass.pack(pady=5)

        tk.Button(self.panel, text="悔棋", command=self.undo_move, **btn_opts).pack(pady=5)
        tk.Button(self.panel, text="认负", command=self.resign, **btn_opts).pack(pady=5)
        tk.Button(self.panel, text="判断胜负(围棋)", command=self.judge_go_win, **btn_opts).pack(pady=5)

        tk.Frame(self.panel, height=2, bd=1, relief=tk.SUNKEN).pack(fill=tk.X, padx=5, pady=10)

        tk.Button(self.panel, text="保存局面", command=self.save_game, **btn_opts).pack(pady=5)
        tk.Button(self.panel, text="读取局面", command=self.load_game, **btn_opts).pack(pady=5)

        # 提示开关
        self.chk_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.panel, text="显示操作反馈", variable=self.chk_var, bg="#f0f0f0").pack(pady=10)

    def start_menu(self):
        # 初始弹窗选择
        pass

    def start_game(self, game_type):
        # 询问棋盘大小
        size = simpledialog.askinteger("设置", "请输入棋盘大小 (8-19):", minvalue=8, maxvalue=19,
                                       initialvalue=15 if game_type == 'gomoku' else 19)
        if not size: return

        if game_type == 'gomoku':
            self.game = GomokuGame(size)
            self.btn_pass.config(state=tk.DISABLED)
        else:
            self.game = GoGame(size)
            self.btn_pass.config(state=tk.NORMAL)

        self.game.set_callback(self.update_info)
        self.update_info(f"游戏开始: {'五子棋' if game_type == 'gomoku' else '围棋'}")
        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        if not self.game: return

        size = self.game.size
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # 计算网格
        grid_w = min(w, h) - 2 * self.margin
        self.cell_size = grid_w / (size - 1)

        # 绘制网格线
        for i in range(size):
            start = self.margin + i * self.cell_size
            # 横线
            self.canvas.create_line(self.margin, start, self.margin + grid_w, start)
            # 竖线
            self.canvas.create_line(start, self.margin, start, self.margin + grid_w)

            # 绘制坐标文字 (可选)
            self.canvas.create_text(self.margin - 15, start, text=str(i + 1), font=("Arial", 8))
            self.canvas.create_text(start, self.margin - 15, text=chr(65 + i), font=("Arial", 8))

        # 绘制天元和星位 (针对19路或15路)
        if size in [15, 19]:
            star_points = []
            if size == 19:
                star_points = [(3, 3), (15, 3), (3, 15), (15, 15), (9, 9), (9, 3), (9, 15), (3, 9), (15, 9)]
            elif size == 15:
                star_points = [(3, 3), (11, 3), (3, 11), (11, 11), (7, 7)]

            for sx, sy in star_points:
                cx = self.margin + sx * self.cell_size
                cy = self.margin + sy * self.cell_size
                self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="black")

        # 绘制棋子
        for y in range(size):
            for x in range(size):
                val = self.game.board[y][x]
                if val != BoardGame.EMPTY:
                    cx = self.margin + x * self.cell_size
                    cy = self.margin + y * self.cell_size
                    color = "black" if val == BoardGame.BLACK else "white"
                    outline = "white" if val == BoardGame.BLACK else "black"
                    self.canvas.create_oval(cx - self.cell_size / 2.2, cy - self.cell_size / 2.2,
                                            cx + self.cell_size / 2.2, cy + self.cell_size / 2.2,
                                            fill=color, outline=outline)

        # 标记最后一步
        if self.game.history:
            # 这里简化处理，不专门存最后一步坐标，可以通过比较 board 差异或者增强 history 数据结构来实现
            # 为作业简单起见，暂不绘制最后一步红点
            pass

    def on_canvas_click(self, event):
        if not self.game or self.game.game_over:
            return

        # 转换坐标
        click_x = event.x - self.margin
        click_y = event.y - self.margin

        # 四舍五入到最近的交叉点
        x = round(click_x / self.cell_size)
        y = round(click_y / self.cell_size)

        success, msg = self.game.place_stone(x, y)
        if success:
            self.draw_board()
            self.update_info(msg)
            if self.game.game_over:
                messagebox.showinfo("游戏结束", msg)
        else:
            if self.chk_var.get():  # 如果开启提示
                self.update_info(f"错误: {msg}")
                messagebox.showwarning("非法操作", msg)

    def pass_move(self):
        if not isinstance(self.game, GoGame) or self.game.game_over:
            return
        success, msg = self.game.place_stone(-1, -1)
        self.update_info(msg)

    def undo_move(self):
        if not self.game: return
        success, msg = self.game.undo()
        if success:
            self.draw_board()
            self.update_info(f"悔棋: {msg}")
        else:
            self.update_info(msg)

    def resign(self):
        if not self.game or self.game.game_over: return
        winner = "白方" if self.game.current_player == BoardGame.BLACK else "黑方"
        self.game.game_over = True
        self.game.winner = BoardGame.WHITE if self.game.current_player == BoardGame.BLACK else BoardGame.BLACK
        self.update_info(f"投子认负，{winner}获胜")
        messagebox.showinfo("结束", f"投子认负，{winner}获胜")

    def judge_go_win(self):
        if not isinstance(self.game, GoGame):
            messagebox.showinfo("提示", "只有围棋需要系统判分")
            return

        # 围棋可以在中途强制结算
        msg = self.game.judge_winner()
        self.update_info(msg)
        messagebox.showinfo("结果", msg)

    def save_game(self):
        if not self.game: return
        filename = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                data = self.game.to_dict()
                with open(filename, 'w') as f:
                    json.dump(data, f)
                self.update_info("游戏已保存")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load_game(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                game_type = data['type']
                if game_type == 'GomokuGame':
                    self.game = GomokuGame(data['size'])
                    self.btn_pass.config(state=tk.DISABLED)
                elif game_type == 'GoGame':
                    self.game = GoGame(data['size'])
                    self.btn_pass.config(state=tk.NORMAL)

                self.game.load_from_dict(data)
                self.game.set_callback(self.update_info)
                self.draw_board()
                self.update_info("存档读取成功")
            except Exception as e:
                messagebox.showerror("错误", f"读取失败: {str(e)}")

    def update_info(self, msg):
        current = "黑方" if self.game and self.game.current_player == BoardGame.BLACK else "白方"
        if self.game and self.game.game_over:
            status = f"[结束] {msg}"
        else:
            status = f"[当前: {current}] {msg}"
        self.info_label.config(text=status)

    def run(self):
        # 窗口大小调整时重绘
        self.canvas.bind("<Configure>", lambda e: self.draw_board())
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = GameWindow(root)
    app.run()