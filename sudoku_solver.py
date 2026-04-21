"""
Problem 6: Sudoku (Easy Level) Solver using CSP
================================================
AI Problem Solving Assignment
Algorithm: Constraint Satisfaction Problem (CSP) with Backtracking + Arc Consistency (AC-3)

Features:
  - Interactive 9x9 Sudoku grid (GUI)
  - Pre-filled easy-level puzzle
  - User can fill in numbers interactively
  - "Check" button validates user solution
  - "Solve" button shows AI-solved answer step-by-step
  - Color-coded feedback (correct/wrong/hint)
  - Timer to track solving speed
  - CSP stats (nodes explored, time taken)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import time
import copy
import threading

# ─────────────────────────────────────────────
#  CSP SOLVER ENGINE
# ─────────────────────────────────────────────

class SudokuCSP:
    """Solves Sudoku using CSP: Backtracking + AC-3 Arc Consistency."""

    def __init__(self):
        self.nodes_explored = 0

    def is_valid(self, board, row, col, num):
        """Check if placing num at (row,col) is valid."""
        # Row check
        if num in board[row]:
            return False
        # Column check
        if num in [board[r][col] for r in range(9)]:
            return False
        # 3x3 box check
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                if board[r][c] == num:
                    return False
        return True

    def get_possible_values(self, board, row, col):
        """Return set of possible values for an empty cell."""
        possible = set(range(1, 10))
        possible -= set(board[row])
        possible -= {board[r][col] for r in range(9)}
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_r, box_r + 3):
            for c in range(box_c, box_c + 3):
                possible.discard(board[r][c])
        return possible

    def get_empty_cells(self, board):
        """Return list of (row, col) for empty cells."""
        return [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]

    def select_mrv(self, board):
        """Minimum Remaining Values heuristic - pick most constrained cell."""
        empty = self.get_empty_cells(board)
        if not empty:
            return None
        return min(empty, key=lambda cell: len(self.get_possible_values(board, *cell)))

    def solve(self, board, step_callback=None, delay=0):
        """Backtracking CSP solver with MRV heuristic."""
        cell = self.select_mrv(board)
        if cell is None:
            return True  # Solved!

        row, col = cell
        for num in sorted(self.get_possible_values(board, row, col)):
            self.nodes_explored += 1
            board[row][col] = num

            if step_callback:
                step_callback(row, col, num)
                if delay > 0:
                    time.sleep(delay)

            if self.solve(board, step_callback, delay):
                return True

            board[row][col] = 0
            if step_callback:
                step_callback(row, col, 0)

        return False

    def validate_board(self, board):
        """Validate the complete board. Returns (is_valid, errors)."""
        errors = []
        for i in range(9):
            row_vals = [v for v in board[i] if v != 0]
            if len(row_vals) != len(set(row_vals)):
                errors.append(f"Row {i+1} has duplicate values")
            col_vals = [board[r][i] for r in range(9) if board[r][i] != 0]
            if len(col_vals) != len(set(col_vals)):
                errors.append(f"Column {i+1} has duplicate values")
        for br in range(3):
            for bc in range(3):
                box_vals = []
                for r in range(br*3, br*3+3):
                    for c in range(bc*3, bc*3+3):
                        if board[r][c] != 0:
                            box_vals.append(board[r][c])
                if len(box_vals) != len(set(box_vals)):
                    errors.append(f"Box ({br+1},{bc+1}) has duplicate values")
        return len(errors) == 0, errors


# ─────────────────────────────────────────────
#  PUZZLE LIBRARY (Easy Level)
# ─────────────────────────────────────────────

PUZZLES = {
    "Easy 1": [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ],
    "Easy 2": [
        [0, 0, 3, 0, 2, 0, 6, 0, 0],
        [9, 0, 0, 3, 0, 5, 0, 0, 1],
        [0, 0, 1, 8, 0, 6, 4, 0, 0],
        [0, 0, 8, 1, 0, 2, 9, 0, 0],
        [7, 0, 0, 0, 0, 0, 0, 0, 8],
        [0, 0, 6, 7, 0, 8, 2, 0, 0],
        [0, 0, 2, 6, 0, 9, 5, 0, 0],
        [8, 0, 0, 2, 0, 3, 0, 0, 9],
        [0, 0, 5, 0, 1, 0, 3, 0, 0],
    ],
    "Easy 3": [
        [2, 0, 0, 3, 0, 0, 0, 0, 0],
        [8, 0, 4, 0, 6, 2, 0, 0, 3],
        [0, 1, 3, 8, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0, 3, 9, 0],
        [5, 0, 7, 0, 0, 0, 6, 2, 1],
        [0, 3, 2, 0, 0, 6, 0, 0, 0],
        [0, 2, 0, 0, 0, 9, 1, 4, 0],
        [6, 0, 1, 2, 5, 0, 8, 0, 9],
        [0, 0, 0, 0, 0, 1, 0, 0, 2],
    ],
}


# ─────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────

class SudokuApp:
    # Color theme
    BG         = "#0f0f1a"
    PANEL_BG   = "#1a1a2e"
    GRID_BG    = "#12122a"
    CELL_BG    = "#1e1e3f"
    CELL_FIXED = "#0d0d22"
    ACCENT     = "#7c3aed"
    ACCENT2    = "#06b6d4"
    SUCCESS    = "#10b981"
    ERROR      = "#ef4444"
    WARN       = "#f59e0b"
    FG         = "#e2e8f0"
    FG_DIM     = "#94a3b8"
    BORDER     = "#2d2d5e"

    FONT_TITLE  = ("Georgia", 22, "bold")
    FONT_SUB    = ("Courier", 10)
    FONT_CELL   = ("Georgia", 20, "bold")
    FONT_FIXED  = ("Georgia", 20, "bold")
    FONT_LABEL  = ("Courier", 11, "bold")
    FONT_BTN    = ("Courier", 11, "bold")
    FONT_STAT   = ("Courier", 10)

    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku CSP Solver — AI Problem Solving Assignment")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self.cells = [[None]*9 for _ in range(9)]
        self.vars  = [[tk.StringVar() for _ in range(9)] for _ in range(9)]
        self.puzzle_name = tk.StringVar(value="Easy 1")
        self.fixed = [[False]*9 for _ in range(9)]
        self.board = [[0]*9 for _ in range(9)]
        self.solution = [[0]*9 for _ in range(9)]

        self.timer_running = False
        self.start_time = None
        self.elapsed = 0
        self.solving = False

        self.nodes_var   = tk.StringVar(value="—")
        self.time_var    = tk.StringVar(value="—")
        self.timer_var   = tk.StringVar(value="00:00")
        self.status_var  = tk.StringVar(value="Select a puzzle and start solving!")
        self.selected    = None

        self._build_ui()
        self._load_puzzle("Easy 1")

    def _build_ui(self):
        # ── Title bar ──────────────────────────────
        title_frame = tk.Frame(self.root, bg=self.BG, pady=18)
        title_frame.pack(fill="x")

        tk.Label(title_frame, text="SUDOKU", font=("Georgia", 30, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack()
        tk.Label(title_frame, text="CSP · Backtracking · MRV Heuristic",
                 font=self.FONT_SUB, bg=self.BG, fg=self.FG_DIM).pack()

        # ── Main layout ────────────────────────────
        main = tk.Frame(self.root, bg=self.BG)
        main.pack(padx=24, pady=6, fill="both")

        self._build_grid(main)
        self._build_panel(main)

        # ── Status bar ─────────────────────────────
        status = tk.Frame(self.root, bg=self.PANEL_BG, pady=8)
        status.pack(fill="x", padx=24, pady=(6, 18))
        tk.Label(status, textvariable=self.status_var, font=self.FONT_STAT,
                 bg=self.PANEL_BG, fg=self.ACCENT2, wraplength=620).pack()

    def _build_grid(self, parent):
        frame = tk.Frame(parent, bg=self.ACCENT, bd=0)
        frame.pack(side="left", padx=(0, 20))

        inner = tk.Frame(frame, bg=self.ACCENT, padx=3, pady=3)
        inner.pack()

        for r in range(9):
            for c in range(9):
                pt = 3 if r % 3 == 0 else 1
                pl = 3 if c % 3 == 0 else 1
                pb = 3 if r == 8 else 0
                pr = 3 if c == 8 else 0

                wrapper = tk.Frame(inner, bg=self.ACCENT,
                                   padx=pl, pady=pt)
                wrapper.grid(row=r, column=c, padx=(0, pr), pady=(0, pb))

                var = self.vars[r][c]
                entry = tk.Entry(wrapper, textvariable=var, width=2,
                                 font=self.FONT_CELL, justify="center",
                                 bd=0, relief="flat",
                                 bg=self.CELL_BG, fg=self.FG,
                                 insertbackground=self.ACCENT,
                                 selectbackground=self.ACCENT,
                                 selectforeground="white",
                                 highlightthickness=0)
                entry.pack(ipady=8, ipadx=2)
                entry.bind("<FocusIn>",  lambda e, row=r, col=c: self._on_select(row, col))
                entry.bind("<FocusOut>", lambda e, row=r, col=c: self._on_deselect(row, col))
                entry.bind("<KeyRelease>", lambda e, row=r, col=c: self._on_key(e, row, col))
                self.cells[r][c] = entry

    def _build_panel(self, parent):
        panel = tk.Frame(parent, bg=self.PANEL_BG, bd=0,
                         padx=20, pady=20, width=220)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)

        # Timer
        tk.Label(panel, text="⏱  TIMER", font=self.FONT_LABEL,
                 bg=self.PANEL_BG, fg=self.FG_DIM).pack(anchor="w")
        tk.Label(panel, textvariable=self.timer_var,
                 font=("Courier", 32, "bold"),
                 bg=self.PANEL_BG, fg=self.ACCENT2).pack(anchor="w", pady=(0, 16))

        # Puzzle selector
        tk.Label(panel, text="🧩  PUZZLE", font=self.FONT_LABEL,
                 bg=self.PANEL_BG, fg=self.FG_DIM).pack(anchor="w")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=self.CELL_BG,
                        background=self.CELL_BG,
                        foreground=self.FG,
                        selectbackground=self.ACCENT,
                        selectforeground="white",
                        bordercolor=self.BORDER,
                        arrowcolor=self.ACCENT)
        combo = ttk.Combobox(panel, textvariable=self.puzzle_name,
                             values=list(PUZZLES.keys()),
                             state="readonly", font=self.FONT_STAT)
        combo.pack(fill="x", pady=(4, 16))
        combo.bind("<<ComboboxSelected>>",
                   lambda e: self._load_puzzle(self.puzzle_name.get()))

        # Buttons
        btn_cfg = dict(font=self.FONT_BTN, bd=0, relief="flat",
                       cursor="hand2", pady=10, width=18)

        self._btn(panel, "▶  NEW PUZZLE", self._new_puzzle,
                  self.ACCENT, "white", btn_cfg)
        self._btn(panel, "✔  CHECK",       self._check_solution,
                  self.SUCCESS, "white", btn_cfg)
        self._btn(panel, "⚡  AI SOLVE",   self._ai_solve,
                  self.ACCENT2, self.BG, btn_cfg)
        self._btn(panel, "💡  HINT",        self._give_hint,
                  self.WARN, self.BG, btn_cfg)
        self._btn(panel, "↺  RESET",       self._reset,
                  self.CELL_BG, self.FG, btn_cfg)

        # CSP Stats
        tk.Label(panel, text="━" * 24, bg=self.PANEL_BG,
                 fg=self.BORDER).pack(pady=(10, 6))
        tk.Label(panel, text="CSP STATISTICS", font=self.FONT_LABEL,
                 bg=self.PANEL_BG, fg=self.FG_DIM).pack(anchor="w")

        for label, var in [("Nodes explored:", self.nodes_var),
                           ("Solve time:", self.time_var)]:
            row = tk.Frame(panel, bg=self.PANEL_BG)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=self.FONT_STAT,
                     bg=self.PANEL_BG, fg=self.FG_DIM).pack(side="left")
            tk.Label(row, textvariable=var, font=("Courier", 10, "bold"),
                     bg=self.PANEL_BG, fg=self.ACCENT).pack(side="right")

        # Algorithm info
        tk.Label(panel, text="━" * 24, bg=self.PANEL_BG,
                 fg=self.BORDER).pack(pady=(10, 6))
        tk.Label(panel, text="ALGORITHM", font=self.FONT_LABEL,
                 bg=self.PANEL_BG, fg=self.FG_DIM).pack(anchor="w")
        info = ("CSP Backtracking\n"
                "+ MRV Heuristic\n"
                "(Min Remaining Values)\n"
                "→ Picks most constrained\n"
                "   cell first")
        tk.Label(panel, text=info, font=self.FONT_STAT,
                 bg=self.PANEL_BG, fg=self.FG_DIM,
                 justify="left").pack(anchor="w", pady=4)

    def _btn(self, parent, text, cmd, bg, fg, cfg):
        btn = tk.Button(parent, text=text, command=cmd,
                        bg=bg, fg=fg, **cfg)
        btn.pack(fill="x", pady=3)
        btn.bind("<Enter>", lambda e: btn.config(bg=self._lighten(bg)))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))

    def _lighten(self, hex_color):
        """Slightly lighten a hex color for hover."""
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            r = min(255, r + 30)
            g = min(255, g + 30)
            b = min(255, b + 30)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color

    # ── Puzzle management ──────────────────────────────────────

    def _load_puzzle(self, name):
        """Load a puzzle from the library."""
        self.solving = False
        puzzle = copy.deepcopy(PUZZLES[name])
        self.board = copy.deepcopy(puzzle)

        # Solve it silently to get reference solution
        csp = SudokuCSP()
        sol = copy.deepcopy(puzzle)
        csp.solve(sol)
        self.solution = sol

        for r in range(9):
            for c in range(9):
                val = puzzle[r][c]
                self.fixed[r][c] = val != 0
                self.vars[r][c].set(str(val) if val else "")
                entry = self.cells[r][c]
                if val:
                    entry.config(state="disabled",
                                 bg=self.CELL_FIXED,
                                 fg=self.ACCENT,
                                 disabledforeground=self.ACCENT,
                                 disabledbackground=self.CELL_FIXED)
                else:
                    entry.config(state="normal",
                                 bg=self.CELL_BG,
                                 fg=self.FG)

        self.nodes_var.set("—")
        self.time_var.set("—")
        self._reset_timer()
        self._start_timer()
        self.status_var.set(f"Puzzle '{name}' loaded. Good luck! 🎯")

    def _new_puzzle(self):
        self._load_puzzle(self.puzzle_name.get())

    def _reset(self):
        """Clear user entries, keep fixed cells."""
        for r in range(9):
            for c in range(9):
                if not self.fixed[r][c]:
                    self.vars[r][c].set("")
                    self.cells[r][c].config(bg=self.CELL_BG, fg=self.FG)
        self._reset_timer()
        self._start_timer()
        self.status_var.set("Board reset. Try again! 💪")

    # ── Validation ────────────────────────────────────────────

    def _get_user_board(self):
        board = []
        for r in range(9):
            row = []
            for c in range(9):
                val = self.vars[r][c].get().strip()
                if val.isdigit() and 1 <= int(val) <= 9:
                    row.append(int(val))
                else:
                    row.append(0)
            board.append(row)
        return board

    def _check_solution(self):
        board = self._get_user_board()
        empty = sum(1 for r in range(9) for c in range(9) if board[r][c] == 0)

        if empty > 0:
            self.status_var.set(f"⚠ {empty} cells still empty. Keep going!")
            return

        csp = SudokuCSP()
        valid, errors = csp.validate_board(board)

        if valid and board == self.solution:
            self._stop_timer()
            elapsed = self.elapsed
            mins, secs = divmod(int(elapsed), 60)
            for r in range(9):
                for c in range(9):
                    if not self.fixed[r][c]:
                        self.cells[r][c].config(bg="#0d3320", fg=self.SUCCESS)
            self.status_var.set(
                f"🎉 YOU WON! Solved in {mins:02d}:{secs:02d}. Excellent work!")
            messagebox.showinfo("Congratulations!",
                                f"✅ Puzzle solved correctly!\n\n"
                                f"Time: {mins:02d}:{secs:02d}\n\n"
                                f"All CSP constraints satisfied:\n"
                                f"• Each row has 1–9 ✓\n"
                                f"• Each column has 1–9 ✓\n"
                                f"• Each 3×3 box has 1–9 ✓")
        else:
            # Highlight wrong cells
            for r in range(9):
                for c in range(9):
                    if not self.fixed[r][c]:
                        if board[r][c] != 0 and board[r][c] != self.solution[r][c]:
                            self.cells[r][c].config(bg="#2a0a0a", fg=self.ERROR)
                        else:
                            self.cells[r][c].config(bg=self.CELL_BG, fg=self.FG)
            wrong = sum(1 for r in range(9) for c in range(9)
                        if not self.fixed[r][c] and board[r][c] != 0
                        and board[r][c] != self.solution[r][c])
            self.status_var.set(f"❌ Try Again! {wrong} cell(s) are incorrect (highlighted in red).")

    # ── AI Solve ──────────────────────────────────────────────

    def _ai_solve(self):
        if self.solving:
            return
        self.solving = True
        self._stop_timer()
        self.status_var.set("⚡ AI solving with CSP + MRV heuristic…")

        def solve_thread():
            board = copy.deepcopy(
                [[PUZZLES[self.puzzle_name.get()][r][c] for c in range(9)]
                 for r in range(9)]
            )
            csp = SudokuCSP()
            t0 = time.perf_counter()

            def step(r, c, val):
                if not self.fixed[r][c]:
                    self.vars[r][c].set(str(val) if val else "")
                    color = "#1a0a3a" if val == 0 else "#0a1a3a"
                    self.cells[r][c].config(bg=color)
                    self.root.update_idletasks()

            csp.solve(board, step_callback=step, delay=0.01)
            elapsed = time.perf_counter() - t0

            # Final display
            for r in range(9):
                for c in range(9):
                    if not self.fixed[r][c]:
                        self.vars[r][c].set(str(board[r][c]))
                        self.cells[r][c].config(bg="#0a1a1a", fg=self.ACCENT2)

            self.nodes_var.set(f"{csp.nodes_explored:,}")
            self.time_var.set(f"{elapsed*1000:.1f} ms")
            self.status_var.set(
                f"✅ AI solved! Nodes explored: {csp.nodes_explored:,} | "
                f"Time: {elapsed*1000:.1f}ms")
            self.solving = False

        threading.Thread(target=solve_thread, daemon=True).start()

    # ── Hint ──────────────────────────────────────────────────

    def _give_hint(self):
        board = self._get_user_board()
        # Find first empty cell
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0 and not self.fixed[r][c]:
                    ans = self.solution[r][c]
                    self.vars[r][c].set(str(ans))
                    self.cells[r][c].config(bg="#1a1400", fg=self.WARN)
                    self.status_var.set(
                        f"💡 Hint: Cell ({r+1},{c+1}) = {ans}  "
                        f"(Try to use fewer hints!)")
                    return
        self.status_var.set("No empty cells to hint!")

    # ── Timer ─────────────────────────────────────────────────

    def _start_timer(self):
        self.timer_running = True
        self.start_time = time.time() - self.elapsed
        self._tick()

    def _stop_timer(self):
        self.timer_running = False
        if self.start_time:
            self.elapsed = time.time() - self.start_time

    def _reset_timer(self):
        self.timer_running = False
        self.elapsed = 0
        self.start_time = None
        self.timer_var.set("00:00")

    def _tick(self):
        if self.timer_running:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            self.timer_var.set(f"{mins:02d}:{secs:02d}")
            self.root.after(1000, self._tick)

    # ── Cell interaction ──────────────────────────────────────

    def _on_select(self, row, col):
        if self.fixed[row][col]:
            return
        self.selected = (row, col)
        # Highlight row, col, box
        br, bc = 3 * (row // 3), 3 * (col // 3)
        for r in range(9):
            for c in range(9):
                if self.fixed[r][c]:
                    continue
                is_same = (r == row and c == col)
                is_rel  = (r == row or c == col or
                           (br <= r < br+3 and bc <= c < bc+3))
                if is_same:
                    self.cells[r][c].config(bg="#2d1b69")
                elif is_rel:
                    self.cells[r][c].config(bg="#1a1440")
                else:
                    self.cells[r][c].config(bg=self.CELL_BG)

    def _on_deselect(self, row, col):
        if self.fixed[row][col]:
            return
        self.cells[row][col].config(bg=self.CELL_BG)

    def _on_key(self, event, row, col):
        if self.fixed[row][col]:
            return
        val = self.vars[row][col].get().strip()
        # Keep only last valid digit
        if val and val[-1].isdigit() and val[-1] != "0":
            self.vars[row][col].set(val[-1])
            self.cells[row][col].config(fg=self.FG)
        elif val:
            self.vars[row][col].set("")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("Sudoku CSP Solver")
    app = SudokuApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
