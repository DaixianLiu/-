"""
面向对象大作业第二阶段
姓名：柳黛鲜
学号：2025214239
"""


import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import copy
import json
import os
import random
import time
import math
import socket
import threading
import queue


# ==========================================
#  Utils & Managers: 用户管理与网络
# ==========================================

class UserManager:
    """用户账户管理类，负责登录、注册和战绩记录"""
    FILE_PATH = "users.json"

    def __init__(self):
        self.users = self._load_users()
        self.current_user = None

    def _load_users(self):
        if not os.path.exists(self.FILE_PATH):
            return {}
        try:
            with open(self.FILE_PATH, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _save_users(self):
        with open(self.FILE_PATH, 'w') as f:
            json.dump(self.users, f)

    def login(self, username, password):
        if username in self.users and self.users[username]['password'] == password:
            self.current_user = username
            return True, "登录成功"
        return False, "用户名或密码错误"

    def register(self, username, password):
        if username in self.users:
            return False, "用户名已存在"
        self.users[username] = {
            'password': password,
            'wins': 0,
            'matches': 0
        }
        self._save_users()
        self.current_user = username
        return True, "注册成功并自动登录"

    def update_stats(self, is_win):
        if self.current_user:
            self.users[self.current_user]['matches'] += 1
            if is_win:
                self.users[self.current_user]['wins'] += 1
            self._save_users()

    def get_stats(self, username=None):
        target = username if username else self.current_user
        if target and target in self.users:
            u = self.users[target]
            return f"{target} (胜/局: {u['wins']}/{u['matches']})"
        return "游客"


class NetworkManager:
    """简易网络对战管理器"""

    def __init__(self, message_callback):
        self.sock = None
        self.is_host = False
        self.connected = False
        self.callback = message_callback  # 回调函数，用于处理接收到的消息
        self.msg_queue = queue.Queue()
        self.running = False

    def start_host(self, port=8888):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(('0.0.0.0', port))
            server.listen(1)
            self.is_host = True
            threading.Thread(target=self._accept_client, args=(server,), daemon=True).start()
            return True, f"正在监听端口 {port}..."
        except Exception as e:
            return False, str(e)

    def _accept_client(self, server):
        try:
            conn, addr = server.accept()
            self.sock = conn
            self.connected = True
            self.callback("CONNECTED", f"已连接: {addr}")
            self._start_listening()
        except:
            pass

    def connect_host(self, ip, port=8888):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((ip, port))
            self.is_host = False
            self.connected = True
            self.callback("CONNECTED", f"已连接主机 {ip}")
            self._start_listening()
            return True, "连接成功"
        except Exception as e:
            return False, str(e)

    def _start_listening(self):
        self.running = True
        threading.Thread(target=self._receive_loop, daemon=True).start()

    def _receive_loop(self):
        while self.running and self.sock:
            try:
                data = self.sock.recv(1024).decode('utf-8')
                if not data: break
                # 可能发生粘包，这里做简单处理，假设每条消息以\n结尾
                for msg in data.split('\n'):
                    if msg:
                        try:
                            payload = json.loads(msg)
                            self.callback("DATA", payload)
                        except:
                            pass
            except:
                break
        self.connected = False
        self.callback("DISCONNECTED", "连接断开")

    def send(self, data):
        if self.connected and self.sock:
            try:
                msg = json.dumps(data) + '\n'
                self.sock.send(msg.encode('utf-8'))
            except:
                pass


# ==========================================
#  Model 层：游戏核心逻辑 (扩展)
# ==========================================

class BoardGame:
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self, size=15):
        self.size = size
        self.board = [[self.EMPTY for _ in range(size)] for _ in range(size)]
        self.current_player = self.BLACK
        self.history = []
        self.move_record = []  # 记录每一步 (player, x, y)，用于回放
        self.game_over = False
        self.winner = None
        self.msg_callback = None

    def set_callback(self, callback):
        self.msg_callback = callback

    def log(self, message):
        if self.msg_callback:
            self.msg_callback(message)

    def switch_player(self):
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK

    def save_state(self):
        state = {
            'board': copy.deepcopy(self.board),
            'player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner
        }
        self.history.append(state)

    def record_move(self, x, y):
        self.move_record.append({
            'player': self.current_player,
            'x': x,
            'y': y
        })

    def undo(self):
        if not self.history: return False, "无棋可悔"
        prev_state = self.history.pop()
        self.board = prev_state['board']
        self.current_player = prev_state['player']
        self.game_over = prev_state['game_over']
        self.winner = prev_state['winner']
        if self.move_record: self.move_record.pop()
        return True, "悔棋成功"

    def is_valid_coord(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def place_stone(self, x, y):
        raise NotImplementedError

    def get_valid_moves(self):
        """获取当前玩家所有合法落子点，供AI使用"""
        moves = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == self.EMPTY:
                    moves.append((x, y))
        return moves

    def check_win(self):
        raise NotImplementedError

    def to_dict(self):
        return {
            'type': self.__class__.__name__,
            'size': self.size,
            'board': self.board,
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'move_record': self.move_record
        }

    def load_from_dict(self, data):
        self.size = data['size']
        self.board = data['board']
        self.current_player = data['current_player']
        self.game_over = data['game_over']
        self.winner = data['winner']
        self.move_record = data.get('move_record', [])
        self.history = []


class ReversiGame(BoardGame):
    """黑白棋 (Othello) 逻辑"""

    def __init__(self, size=8):
        super().__init__(size)
        # 初始布局
        mid = size // 2
        self.board[mid - 1][mid - 1] = self.WHITE
        self.board[mid][mid] = self.WHITE
        self.board[mid - 1][mid] = self.BLACK
        self.board[mid][mid - 1] = self.BLACK

    def place_stone(self, x, y):
        if self.game_over: return False, "游戏已结束"

        # 弃权逻辑 (Pass)
        if x == -1 and y == -1:
            self.save_state()
            self.record_move(-1, -1)
            self.switch_player()
            # 检查下一位是否有棋可走，若双方都无，则结束
            if not self.get_valid_moves() and not self.has_valid_moves(
                    self.WHITE if self.current_player == self.BLACK else self.BLACK):
                self.judge_winner()
                return True, "双方无子可下，游戏结束"
            return True, "无处落子，被迫弃权"

        if not self.is_valid_coord(x, y): return False, "坐标不合法"
        if self.board[y][x] != self.EMPTY: return False, "此处已有棋子"

        flipped = self.get_flipped_stones(x, y, self.current_player)
        if not flipped:
            return False, "必须夹住对方棋子才能落子"

        self.save_state()
        self.board[y][x] = self.current_player
        for fx, fy in flipped:
            self.board[fy][fx] = self.current_player

        self.record_move(x, y)

        self.switch_player()

        # 检查对方是否有子可下，若无，轮回到自己；若自己也无，结束
        if not self.get_valid_moves():
            self.log("对方无子可下，轮空")
            self.switch_player()
            if not self.get_valid_moves():
                self.judge_winner()
                return True, "双方无子可下，游戏结束"

        if self.is_full():
            self.judge_winner()
            return True, "棋盘已满，游戏结束"

        return True, "落子成功"

    def get_flipped_stones(self, x, y, player):
        """计算落子后会翻转的棋子"""
        opponent = self.WHITE if player == self.BLACK else self.BLACK
        flipped = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dx, dy in directions:
            temp = []
            nx, ny = x + dx, y + dy
            while self.is_valid_coord(nx, ny) and self.board[ny][nx] == opponent:
                temp.append((nx, ny))
                nx += dx
                ny += dy
            # 如果末端是己方棋子，则中间的可以翻转
            if self.is_valid_coord(nx, ny) and self.board[ny][nx] == player:
                flipped.extend(temp)
        return flipped

    def get_valid_moves(self):
        moves = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] == self.EMPTY:
                    if self.get_flipped_stones(x, y, self.current_player):
                        moves.append((x, y))
        return moves

    def has_valid_moves(self, player):
        """辅助检查某方是否有棋可走"""
        original_player = self.current_player
        self.current_player = player
        moves = self.get_valid_moves()
        self.current_player = original_player
        return len(moves) > 0

    def is_full(self):
        for row in self.board:
            if self.EMPTY in row: return False
        return True

    def count_score(self):
        b = sum(row.count(self.BLACK) for row in self.board)
        w = sum(row.count(self.WHITE) for row in self.board)
        return b, w

    def judge_winner(self):
        b, w = self.count_score()
        self.game_over = True
        if b > w:
            self.winner = self.BLACK
        elif w > b:
            self.winner = self.WHITE
        else:
            self.winner = None  # 平局


class GomokuGame(BoardGame):
    def place_stone(self, x, y):
        if self.game_over: return False, "游戏已结束"
        if not self.is_valid_coord(x, y): return False, "坐标不合法"
        if self.board[y][x] != self.EMPTY: return False, "此处已有棋子"

        self.save_state()
        self.board[y][x] = self.current_player
        self.record_move(x, y)  # 新增记录

        if self.check_win(x, y):
            self.game_over = True
            self.winner = self.current_player
            return True, "获胜"
        if self.is_full():
            self.game_over = True
            return True, "平局"

        self.switch_player()
        return True, "落子成功"

    def is_full(self):
        for row in self.board:
            if self.EMPTY in row: return False
        return True

    def check_win(self, x, y):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        color = self.board[y][x]
        for dx, dy in directions:
            count = 1
            nx, ny = x + dx, y + dy
            while self.is_valid_coord(nx, ny) and self.board[ny][nx] == color:
                count += 1
                nx, ny = nx + dx, ny + dy
            nx, ny = x - dx, y - dy
            while self.is_valid_coord(nx, ny) and self.board[ny][nx] == color:
                count += 1
                nx, ny = nx - dx, ny - dy
            if count >= 5: return True
        return False


class GoGame(BoardGame):
    def __init__(self, size=19):
        super().__init__(size)
        self.captured_counts = {self.BLACK: 0, self.WHITE: 0}

    def place_stone(self, x, y):
        if self.game_over: return False, "已结束"
        # Pass
        if x == -1 and y == -1:
            self.save_state()
            self.record_move(-1, -1)
            self.switch_player()
            return True, "虚着"

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

        # 通过检查
        self.save_state()
        self.board[y][x] = self.current_player  # 简化示意
        self.record_move(x, y)  # 新增记录
        self.switch_player()
        return True, "落子成功"

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
#  AI 引擎
# ==========================================

class AIEngine:
    @staticmethod
    def get_move(game, level):
        """
        根据游戏类型和AI等级返回最佳落子 (x, y)
        Level 1: 随机
        Level 2: 启发式
        Level 3: MCTS / 强力规则
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return (-1, -1)  # Pass

        if level == 1:
            return random.choice(valid_moves)

        elif level == 2:
            if isinstance(game, ReversiGame):
                return AIEngine._reversi_greedy(game, valid_moves)
            elif isinstance(game, GomokuGame):
                return AIEngine._gomoku_heuristic(game, valid_moves)
            else:
                return random.choice(valid_moves)  # 围棋复杂，fallback到随机

        elif level == 3:
            # 简单 MCTS 模拟
            return AIEngine._mcts_search(game, valid_moves)

        return random.choice(valid_moves)

    @staticmethod
    def _reversi_greedy(game, moves):
        # 黑白棋贪心：优先角点，其次翻转最多，避免C点X点
        best_score = -9999
        best_move = moves[0]
        size = game.size
        corners = {(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)}

        for x, y in moves:
            score = 0
            if (x, y) in corners:
                score += 100
            # 模拟翻转数量
            flipped = len(game.get_flipped_stones(x, y, game.current_player))
            score += flipped
            if score > best_score:
                best_score = score
                best_move = (x, y)
        return best_move

    @staticmethod
    def _gomoku_heuristic(game, moves):
        # 简单评分：优先选中央，或者能连成更多子的
        center = game.size // 2
        best_move = moves[0]
        min_dist = 999
        for x, y in moves:
            dist = abs(x - center) + abs(y - center)
            if dist < min_dist:
                min_dist = dist
                best_move = (x, y)
        return best_move

    @staticmethod
    def _mcts_search(game, moves, simulations=50):
        # 简化版蒙特卡洛：对每个候选点进行多次随机模拟
        best_win_rate = -1
        best_move = moves[0]

        player = game.current_player

        for mx, my in moves:
            wins = 0
            for _ in range(simulations):
                # 复制游戏状态进行模拟
                # 注意：为了性能，实际MCTS通常不深拷贝整个对象，这里为演示简化
                sim_game = copy.deepcopy(game)
                sim_game.msg_callback = None  # 禁用日志
                sim_game.place_stone(mx, my)

                # 随机走步直到结束或一定步数
                steps = 0
                while not sim_game.game_over and steps < 20:
                    v_moves = sim_game.get_valid_moves()
                    if not v_moves:
                        # 尝试Pass
                        sim_game.place_stone(-1, -1)
                        if sim_game.game_over: break
                        continue
                    rx, ry = random.choice(v_moves)
                    sim_game.place_stone(rx, ry)
                    steps += 1

                # 判胜负
                if isinstance(sim_game, ReversiGame):
                    b, w = sim_game.count_score()
                    if (player == BoardGame.BLACK and b > w) or (player == BoardGame.WHITE and w > b):
                        wins += 1
                elif sim_game.winner == player:
                    wins += 1

            rate = wins / simulations
            if rate > best_win_rate:
                best_win_rate = rate
                best_move = (mx, my)

        return best_move


# ==========================================
#  View & Controller
# ==========================================

class GameWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Python综合对战平台")
        self.root.geometry("1000x750")

        self.user_mgr = UserManager()
        self.net_mgr = NetworkManager(self.on_net_message)

        self.game = None
        self.mode = "PvP"  # PvP, PvE, EvE, Net, Replay
        self.ai_level = 1
        self.replay_step = 0
        self.saved_record = []

        self.cell_size = 35
        self.margin = 30

        self._init_ui()
        self.show_login_dialog()

    def _init_ui(self):
        # 顶部菜单栏/状态栏
        self.top_frame = tk.Frame(self.root, height=30)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        self.lbl_user = tk.Label(self.top_frame, text="未登录", fg="darkblue")
        self.lbl_user.pack(side=tk.RIGHT, padx=10)

        # 左右布局
        self.panel = tk.Frame(self.root, width=220, bg="#f0f0f0")
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(self.root, bg="#DEB887")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

        # 控制面板
        tk.Label(self.panel, text="游戏控制", font=("Arial", 14, "bold")).pack(pady=10)
        self.info_label = tk.Label(self.panel, text="准备就绪", wraplength=200)
        self.info_label.pack(pady=5)

        # 模式选择
        tk.Label(self.panel, text="模式:").pack(anchor='w', padx=10)
        self.var_mode = tk.StringVar(value="PvP")
        modes = [("人人对战", "PvP"), ("人机对战", "PvE"), ("机机对战", "EvE")]
        for text, val in modes:
            tk.Radiobutton(self.panel, text=text, variable=self.var_mode, value=val, command=self.set_mode).pack(
                anchor='w', padx=20)

        # AI难度
        tk.Label(self.panel, text="AI等级:").pack(anchor='w', padx=10)
        self.var_ai = tk.IntVar(value=1)
        for i in range(1, 4):
            tk.Radiobutton(self.panel, text=f"Level {i}", variable=self.var_ai, value=i).pack(anchor='w', padx=20)

        # 游戏按钮
        f_btn = tk.Frame(self.panel)
        f_btn.pack(pady=10)
        tk.Button(f_btn, text="五子棋", command=lambda: self.start_game('gomoku')).grid(row=0, column=0, padx=2)
        tk.Button(f_btn, text="围棋", command=lambda: self.start_game('go')).grid(row=0, column=1, padx=2)
        tk.Button(f_btn, text="黑白棋", command=lambda: self.start_game('reversi')).grid(row=1, column=0, columnspan=2,
                                                                                      pady=2)

        # 常用操作
        tk.Button(self.panel, text="悔棋", command=self.undo_move).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(self.panel, text="存档/录像", command=self.save_game).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(self.panel, text="读取/回放", command=self.load_game).pack(fill=tk.X, padx=10, pady=2)

        # 回放控制区
        self.f_replay = tk.LabelFrame(self.panel, text="回放控制")
        self.f_replay.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(self.f_replay, text="< 上一步", command=lambda: self.step_replay(-1)).pack(side=tk.LEFT)
        tk.Button(self.f_replay, text="下一步 >", command=lambda: self.step_replay(1)).pack(side=tk.RIGHT)

        # 网络控制区
        self.f_net = tk.LabelFrame(self.panel, text="网络对战")
        self.f_net.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(self.f_net, text="作为主机", command=self.net_host).pack(fill=tk.X)
        tk.Button(self.f_net, text="连接主机", command=self.net_connect).pack(fill=tk.X)

    def show_login_dialog(self):
        d = tk.Toplevel(self.root)
        d.title("登录/注册")
        d.geometry("300x200")

        tk.Label(d, text="用户名:").pack()
        e_user = tk.Entry(d)
        e_user.pack()
        tk.Label(d, text="密码:").pack()
        e_pwd = tk.Entry(d, show="*")
        e_pwd.pack()

        def do_login():
            ok, msg = self.user_mgr.login(e_user.get(), e_pwd.get())
            if ok:
                messagebox.showinfo("成功", msg)
                self.update_user_display()
                d.destroy()
            else:
                messagebox.showerror("失败", msg)

        def do_register():
            ok, msg = self.user_mgr.register(e_user.get(), e_pwd.get())
            messagebox.showinfo("结果", msg)
            if ok:
                self.update_user_display()
                d.destroy()

        tk.Button(d, text="登录", command=do_login).pack(pady=5)
        tk.Button(d, text="注册", command=do_register).pack(pady=5)
        # 允许直接关闭做游客

    def update_user_display(self):
        self.lbl_user.config(text=self.user_mgr.get_stats())

    def set_mode(self):
        self.mode = self.var_mode.get()
        # 如果是EvE，可能需要自动开始
        if self.mode == "EvE" and self.game and not self.game.game_over:
            self.run_ai_step()

    def start_game(self, gtype):
        # 1. 如果是 Client 模式且已连接，禁止手动开始游戏，等待 Host
        if self.mode == "Net" and not self.net_mgr.is_host and self.net_mgr.connected:
            messagebox.showinfo("提示", "您是客户端，请等待主机开始游戏。")
            return

        size = 15
        if gtype == 'reversi':
            size = 8
        elif gtype == 'go':
            size = 19

        # 允许用户调整
        res = simpledialog.askinteger("设置", f"棋盘大小 ({'黑白棋默认8' if gtype == 'reversi' else '8-19'}):",
                                      minvalue=8, maxvalue=19, initialvalue=size)
        if not res: return
        size = res

        self.init_game_internal(gtype, size)

        # 2. 如果是 Host 模式，发送开始指令给 Client
        if self.mode == "Net" and self.net_mgr.is_host and self.net_mgr.connected:
            self.net_mgr.send({'type': 'start', 'game': gtype, 'size': size})

    def init_game_internal(self, gtype, size):
        """内部初始化游戏逻辑，供本地和网络调用"""
        if gtype == 'gomoku':
            self.game = GomokuGame(size)
        elif gtype == 'go':
            self.game = GoGame(size)
        elif gtype == 'reversi':
            self.game = ReversiGame(size)

        self.game.set_callback(self.update_info)

        # 如果不是网络同步启动，就读取单选框；如果是网络启动，保持 Net 模式
        if self.mode != "Net":
            self.mode = self.var_mode.get()

        self.replay_step = 0
        self.saved_record = []

        self.draw_board()
        self.update_info(f"开始 {gtype} ({self.mode})")

        if self.mode == "EvE":
            self.root.after(500, self.run_ai_step)

    def draw_board(self):
        self.canvas.delete("all")
        if not self.game: return

        size = self.game.size
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        grid_w = min(w, h) - 2 * self.margin
        self.cell_size = grid_w / (size - 1) if size > 1 else 0

        # 绘制网格
        for i in range(size):
            p = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, p, self.margin + grid_w, p)
            self.canvas.create_line(p, self.margin, p, self.margin + grid_w)

        # 绘制棋子
        for y in range(size):
            for x in range(size):
                v = self.game.board[y][x]
                if v != 0:
                    cx = self.margin + x * self.cell_size
                    cy = self.margin + y * self.cell_size
                    color = "black" if v == 1 else "white"
                    outline = "white" if v == 1 else "black"
                    self.canvas.create_oval(cx - self.cell_size / 2.2, cy - self.cell_size / 2.2,
                                            cx + self.cell_size / 2.2, cy + self.cell_size / 2.2,
                                            fill=color, outline=outline)

    def on_click(self, event):
        if not self.game or self.game.game_over: return
        if self.mode == "Replay": return

        # 如果是 AI 回合，忽略点击
        is_ai_turn = (self.mode == "PvE" and self.game.current_player == BoardGame.WHITE) or (self.mode == "EvE")
        if is_ai_turn: return

        # --- 网络对战落子权限检查 ---
        if self.mode == "Net":
            # 约定：主机执黑，从机执白
            is_host = self.net_mgr.is_host
            current_player = self.game.current_player

            # 如果我是主机(黑)，但现在轮到白棋，禁止操作
            if is_host and current_player == BoardGame.WHITE:
                return
            # 如果我是从机(白)，但现在轮到黑棋，禁止操作
            if not is_host and current_player == BoardGame.BLACK:
                return
        # ---------------------------

        cx = round((event.x - self.margin) / self.cell_size)
        cy = round((event.y - self.margin) / self.cell_size)

        self.human_move(cx, cy)

    def human_move(self, x, y):
        """处理本地玩家落子"""
        success, msg = self.game.place_stone(x, y)
        if success:
            self.draw_board()
            self.update_info(msg)
            self.check_game_over()

            # 网络同步：只有在本地成功落子后，才发送给对手
            if self.mode == "Net":
                self.net_mgr.send({'type': 'move', 'x': x, 'y': y})

            # 触发AI
            if not self.game.game_over and self.mode == "PvE":
                self.root.after(500, self.run_ai_step)
        else:
            self.update_info(msg)

    def apply_remote_move(self, x, y):
        """处理来自网络的对手落子"""
        if not self.game: return

        # 直接调用 place_stone 更新本地棋盘状态
        success, msg = self.game.place_stone(x, y)
        if success:
            self.draw_board()
            self.update_info(f"对手落子: {msg}")
            self.check_game_over()
        else:
            # 理论上不应发生，除非同步出错
            self.update_info(f"同步错误: {msg}")

    def run_ai_step(self):
        if not self.game or self.game.game_over: return

        level = self.var_ai.get()
        x, y = AIEngine.get_move(self.game, level)

        success, msg = self.game.place_stone(x, y)
        self.draw_board()
        self.update_info(f"AI落子({x},{y}): {msg}")
        self.check_game_over()

        if self.mode == "EvE" and not self.game.game_over:
            self.root.after(100, self.run_ai_step)

    def check_game_over(self):
        if self.game.game_over:
            winner = self.game.winner
            # 更新战绩
            if self.mode in ["PvP", "PvE", "Net"] and self.user_mgr.current_user:
                is_win = (winner == BoardGame.BLACK)  # 假设玩家总是黑方或先手
                self.user_mgr.update_stats(is_win)
                self.update_user_display()
            messagebox.showinfo("结束", f"游戏结束")

    def save_game(self):
        if not self.game: return
        fname = filedialog.asksaveasfilename(defaultextension=".json")
        if fname:
            data = self.game.to_dict()
            with open(fname, 'w') as f:
                json.dump(data, f)
            messagebox.showinfo("成功", "录像已保存")

    def load_game(self):
        fname = filedialog.askopenfilename()
        if fname:
            with open(fname, 'r') as f:
                data = json.load(f)

            # 恢复游戏类型
            gtype = data.get('type')
            size = data.get('size')
            if gtype == 'ReversiGame':
                self.game = ReversiGame(size)
            elif gtype == 'GoGame':
                self.game = GoGame(size)
            else:
                self.game = GomokuGame(size)

            # 进入回放模式
            self.mode = "Replay"
            self.info_label.config(text="进入回放模式")
            self.saved_record = data.get('move_record', [])
            self.game.move_record = self.saved_record
            self.replay_step = 0

            # 重置棋盘到初始状态以便回放
            self.game.board = [[0] * size for _ in range(size)]
            if isinstance(self.game, ReversiGame):
                mid = size // 2
                self.game.board[mid - 1][mid - 1] = 2
                self.game.board[mid][mid] = 2
                self.game.board[mid - 1][mid] = 1
                self.game.board[mid][mid - 1] = 1
            self.game.current_player = BoardGame.BLACK
            self.draw_board()

    def step_replay(self, direction):
        if self.mode != "Replay" or not self.saved_record: return

        if direction == 1:  # Next
            if self.replay_step < len(self.saved_record):
                move = self.saved_record[self.replay_step]
                self.game.current_player = move['player']
                # 强制落子，不检查合法性(因为是录像)或重新执行逻辑
                # 为简单起见，这里重新执行 place_stone，确保吃子等逻辑生效
                self.game.place_stone(move['x'], move['y'])
                self.replay_step += 1
                self.draw_board()
        elif direction == -1:  # Prev
            # 回放倒退比较复杂（需要undo），这里简单重置再跑
            if self.replay_step > 0:
                target = self.replay_step - 1
                # 重置
                self.load_game_internal_reset()
                # 快进到 target
                for i in range(target):
                    m = self.saved_record[i]
                    self.game.current_player = m['player']
                    self.game.place_stone(m['x'], m['y'])
                self.replay_step = target
                self.draw_board()

    def load_game_internal_reset(self):
        # 辅助函数：清空棋盘
        size = self.game.size
        self.game.board = [[0] * size for _ in range(size)]
        if isinstance(self.game, ReversiGame):
            mid = size // 2
            self.game.board[mid - 1][mid - 1] = 2
            self.game.board[mid][mid] = 2
            self.game.board[mid - 1][mid] = 1
            self.game.board[mid][mid - 1] = 1
        self.game.current_player = BoardGame.BLACK

    def undo_move(self):
        if self.game:
            self.game.undo()
            self.draw_board()

    # --- 网络部分 ---
    def net_host(self):
        self.mode = "Net"
        ok, msg = self.net_mgr.start_host()
        self.update_info(msg)

    def net_connect(self):
        ip = simpledialog.askstring("IP", "输入主机IP:")
        if ip:
            self.mode = "Net"
            ok, msg = self.net_mgr.connect_host(ip)
            self.update_info(msg)

    def on_net_message(self, mtype, data):
        # 处理网络消息，注意要在主线程执行UI更新
        if mtype == "DATA":
            dtype = data.get('type')
            if dtype == 'move':
                # 收到对手落子，使用 apply_remote_move 避免死循环
                self.root.after(0, lambda: self.apply_remote_move(data['x'], data['y']))
            elif dtype == 'start':
                # 收到开始游戏指令
                self.root.after(0, lambda: self.init_game_internal(data['game'], data['size']))

        elif mtype == "CONNECTED":
            self.root.after(0, lambda: self.update_info(str(data)))

    def update_info(self, msg):
        self.info_label.config(text=str(msg))


if __name__ == "__main__":
    root = tk.Tk()
    app = GameWindow(root)
    # 绑定窗口大小变化重绘
    root.bind("<Configure>", lambda e: app.draw_board())
    root.mainloop()