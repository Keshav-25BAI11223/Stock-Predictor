import yfinance as yf
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
import threading
import io
import contextlib

# --- THEME COLORS ---
COLORS = {
    "bg": "#050505",
    "sidebar": "#0F0F0F",
    "card": "#161616",
    "accent": "#0078D4",
    "text": "#E0E0E0",
    "grey": "#666666",
    "green": "#00FF66",
    "red": "#FF3333"
}

analyzer = SentimentIntensityAnalyzer()

class SmoothScrollCanvas(tk.Canvas):
    """High-performance scrollable canvas with momentum interpolation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self._vel = 0
        self._pos = 0

    def _on_mousewheel(self, event):
        self._vel += (event.delta / 120) * -0.02
        self._animate()

    def _animate(self):
        if abs(self._vel) > 0.0001:
            self._pos = max(0, min(1, self.yview()[0] + self._vel))
            self.yview_moveto(self._pos)
            self._vel *= 0.85  # Friction
            self.after(10, self._animate)

class QuantTerminalPro:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Stock Predictor")
        self.root.geometry("1450x950")
        self.root.configure(bg=COLORS["bg"])
        
        self.watchlist = ["RELIANCE.NS", "BHEL.NS", "MRF.NS"]
        self.inr_rate = self.fetch_inr_rate()
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        self.setup_layout()

    def fetch_inr_rate(self):
        try: return yf.Ticker("USDINR=X").fast_info['lastPrice']
        except: return 83.2

    def setup_layout(self):
        # --- LEFT SIDEBAR (20%) ---
        self.sidebar = tk.Frame(self.root, bg=COLORS["sidebar"], width=320)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Search
        tk.Label(self.sidebar, text="QUICK SEARCH", bg=COLORS["sidebar"], fg=COLORS["accent"], font=("Segoe UI", 10, "bold")).pack(pady=(40, 5), padx=25, anchor="w")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.search_logic)
        self.entry = tk.Entry(self.sidebar, textvariable=self.search_var, bg="#222", fg="white", insertbackground="white", font=("Segoe UI", 12), borderwidth=0)
        self.entry.pack(fill="x", padx=25, pady=5)
        
        self.res_pane = tk.Frame(self.root, bg="#1a1a1a", highlightthickness=1, highlightbackground=COLORS["accent"])

        # Watchlist
        tk.Label(self.sidebar, text="WATCHLIST (INR)", bg=COLORS["sidebar"], fg=COLORS["grey"], font=("Segoe UI", 9, "bold")).pack(pady=(30, 5), padx=25, anchor="w")
        self.list_box = tk.Frame(self.sidebar, bg=COLORS["sidebar"])
        self.list_box.pack(fill="both", expand=True, padx=15)
        self.render_watchlist()

        # Action
        self.action_btn = tk.Button(self.sidebar, text="START AI ENGINE", command=self.run_async_engine, bg=COLORS["accent"], fg="white", font=("Segoe UI", 11, "bold"), pady=18, borderwidth=0, cursor="hand2")
        self.action_btn.pack(side="bottom", fill="x", padx=25, pady=40)

        # --- MAIN AREA (80%) ---
        self.main = tk.Frame(self.root, bg=COLORS["bg"])
        self.main.pack(side="right", fill="both", expand=True)

        self.canvas = SmoothScrollCanvas(self.main, bg=COLORS["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main, orient="vertical", command=self.canvas.yview)
        self.view = tk.Frame(self.canvas, bg=COLORS["bg"])
        
        self.canvas.create_window((0, 0), window=self.view, anchor="nw", width=1100)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.view.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def search_logic(self, *args):
        q = self.search_var.get().upper()
        if len(q) < 2: self.res_pane.place_forget(); return
        threading.Thread(target=self.do_search, args=(q,), daemon=True).start()

    def do_search(self, q):
        try:
            res = yf.Search(q, max_results=5).quotes
            self.root.after(0, lambda: self.render_search(res))
        except: pass

    def render_search(self, results):
        for w in self.res_pane.winfo_children(): w.destroy()
        self.res_pane.place(x=25, y=115, width=270)
        self.res_pane.lift()
        for r in results:
            s = r['symbol']
            tk.Button(self.res_pane, text=f"{s}\n{r.get('shortname','')[:20]}", bg="#1a1a1a", fg="white", font=("Segoe UI", 9), anchor="w", justify="left", padx=10, pady=8, borderwidth=0, command=lambda sym=s: self.add_sym(sym)).pack(fill="x")

    def add_sym(self, s):
        if s not in self.watchlist:
            self.watchlist.append(s)
            self.render_watchlist()
        self.res_pane.place_forget()
        self.search_var.set("")

    def render_watchlist(self):
        for w in self.list_box.winfo_children(): w.destroy()
        for t in self.watchlist:
            f = tk.Frame(self.list_box, bg="#121212", pady=10, padx=12)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=t, fg="white", bg="#121212", font=("Segoe UI", 10, "bold")).pack(side="left")
            
            p_lab = tk.Label(f, text="₹...", fg=COLORS["grey"], bg="#121212", font=("Segoe UI", 9))
            p_lab.pack(side="left", padx=15)
            
            def load_price(ticker, lab):
                try:
                    tick = yf.Ticker(ticker)
                    mult = self.inr_rate if tick.fast_info['currency'] == 'USD' else 1.0
                    lab.config(text=f"₹{tick.fast_info['lastPrice']*mult:,.0f}")
                except: lab.config(text="N/A")
            threading.Thread(target=load_price, args=(t, p_lab), daemon=True).start()
            tk.Button(f, text="✕", bg="#121212", fg="#444", borderwidth=0, command=lambda s=t: self.rem_sym(s)).pack(side="right")

    def rem_sym(self, s):
        self.watchlist.remove(s)
        self.render_watchlist()

    def run_async_engine(self):
        self.action_btn.config(state="disabled", text="AI PROCESSING...")
        for w in self.view.winfo_children(): w.destroy()
        
        def start_tasks():
            futures = [self.executor.submit(self.ai_worker, t) for t in self.watchlist]
            for f in futures:
                data = f.result()
                if data: self.root.after(0, lambda d=data: self.inject_card(d))
            self.root.after(0, lambda: self.action_btn.config(state="normal", text="START AI ENGINE"))

        threading.Thread(target=start_tasks, daemon=True).start()

    def ai_worker(self, ticker):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                obj = yf.Ticker(ticker)
                df = obj.history(period="2y")
                if df.empty and "." not in ticker: 
                    ticker += ".NS"; obj = yf.Ticker(ticker); df = obj.history(period="2y")
            if df.empty: return None
            
            # Currency Normalization
            mult = self.inr_rate if obj.fast_info['currency'] == 'USD' else 1.0
            df['Close'] *= mult
            
            # AI Math (Log Returns = Stationarity)
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['SMA'] = df['Log_Ret'].rolling(10).mean()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            X = df[['Log_Ret', 'SMA']]
            y = df['Log_Ret'].shift(-1).dropna()
            X_sc = StandardScaler().fit_transform(X[:-1])
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_sc[:-20], y[:-20])
            
            preds_raw = model.predict(X_sc[-20:])
            last_val = df['Close'].iloc[-21]
            final_preds = []
            for r in preds_raw:
                last_val *= np.exp(r); final_preds.append(last_val)

            sent = np.mean([analyzer.polarity_scores(n.get('title',''))['compound'] for n in obj.news[:3]]) if obj.news else 0
            return {"ticker": ticker, "df": df, "preds": final_preds, "sent": sent}
        except: return None

    def inject_card(self, data):
        card = tk.Frame(self.view, bg=COLORS["card"], pady=25, padx=30)
        card.pack(fill="x", padx=50, pady=15)

        head = tk.Frame(card, bg=COLORS["card"])
        head.pack(fill="x")
        tk.Label(head, text=data['ticker'], fg=COLORS["accent"], bg=COLORS["card"], font=("Segoe UI", 18, "bold")).pack(side="left")
        
        last = data['df']['Close'].iloc[-1]
        targ = data['preds'][-1]
        mood = "BULLISH" if data['sent'] > 0.05 else "BEARISH" if data['sent'] < -0.05 else "NEUTRAL"
        m_col = COLORS["green"] if mood == "BULLISH" else COLORS["red"] if mood == "BEARISH" else COLORS["grey"]
        tk.Label(head, text=f"{mood} | Target ₹{targ:,.2f}", fg=m_col, bg=COLORS["card"], font=("Segoe UI", 12, "bold")).pack(side="right")

        fig = Figure(figsize=(11, 3.8), facecolor=COLORS["card"], dpi=90)
        ax = fig.add_subplot(111); ax.set_facecolor(COLORS["card"])
        dates = data['df'].index[-20:]
        ax.plot(dates, data['df']['Close'].tail(20), color=COLORS["accent"], label="Actual", marker="o", linewidth=2)
        ax.plot(dates, data['preds'], color="#FF0077", linestyle="--", label="AI Prediction", alpha=0.9)
        ax.tick_params(colors='white', labelsize=8); ax.grid(alpha=0.1, color="white")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        
        canvas = FigureCanvasTkAgg(fig, master=card); canvas.draw()
        canvas.get_tk_widget().pack(fill="both")
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantTerminalPro(root)
    root.mainloop()