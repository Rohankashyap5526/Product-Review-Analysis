"""
gui_app.py
----------
Full E-Commerce Analytics Application with:
  Tab 1 - Dashboard      : KPI cards + 5 embedded Matplotlib charts
  Tab 2 - Sentiment      : real-time review sentiment predictor + session tally
  Tab 3 - ML Predictor   : customer profile -> purchase probability gauge
  Tab 4 - Data Explorer  : searchable, filterable, sortable transaction table

Run:  python gui/gui_app.py
"""

import sys, os, re, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import tkinter as tk
from tkinter import ttk, messagebox

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sentiment_analysis import predict_sentiment

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join("data/ecommerce_data.csv")
MDL_DIR   = os.path.join("models/")

# ── Color palette ──────────────────────────────────────────────────────────────
BG      = "#0f1117"
SURFACE = "#1a1d27"
CARD    = "#22263a"
ACCENT  = "#6c63ff"
ACCENT2 = "#574fd6"
POS     = "#2ecc71"
NEU     = "#f39c12"
NEG     = "#e74c3c"
FG      = "#e8e8f0"
FG2     = "#9b9bb4"
BORDER  = "#2e3250"

PALETTE = ["#6c63ff","#2ecc71","#f39c12","#e74c3c","#3498db",
           "#9b59b6","#1abc9c","#e67e22","#e91e63","#00bcd4"]

MPL_RC = {
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   FG2,
    "xtick.color":       FG2,
    "ytick.color":       FG2,
    "text.color":        FG,
    "grid.color":        BORDER,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
}
plt.rcParams.update(MPL_RC)


# ══════════════════════════════════════════════════════════════════════════════
#  Cached data store
# ══════════════════════════════════════════════════════════════════════════════
class DataStore:
    _df = None

    @classmethod
    def load(cls):
        if cls._df is None:
            df = pd.read_csv(DATA_PATH)
            df.drop_duplicates(inplace=True)
            df["rating"]    = df["rating"].fillna(df["rating"].median())
            df["review"]    = df["review"].fillna("no review")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["year"]      = df["timestamp"].dt.year
            df["month"]     = df["timestamp"].dt.month
            df["total_amount"] = df["total_amount"].astype(float)
            cls._df = df
        return cls._df


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def make_card(parent, **kw):
    kw.setdefault("bg", CARD)
    kw.setdefault("highlightthickness", 1)
    kw.setdefault("highlightbackground", BORDER)
    return tk.Frame(parent, **kw)


def kpi_card(parent, label, value, color=ACCENT, width=175, height=88):
    f = make_card(parent, width=width, height=height)
    tk.Label(f, text=label, bg=CARD, fg=FG2,
             font=("Helvetica", 9)).pack(pady=(12, 2))
    tk.Label(f, text=value, bg=CARD, fg=color,
             font=("Helvetica", 18, "bold")).pack()
    return f


def embed_fig(parent, fig, row=0, col=0, rowspan=1, colspan=1, sticky="nsew"):
    c = FigureCanvasTkAgg(fig, master=parent)
    c.draw()
    c.get_tk_widget().grid(row=row, column=col, rowspan=rowspan,
                           columnspan=colspan, sticky=sticky,
                           padx=4, pady=4)
    return c


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Dashboard
# ══════════════════════════════════════════════════════════════════════════════
class DashboardTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._build()

    def _build(self):
        df = DataStore.load()

        # KPI row
        kpi_row = tk.Frame(self, bg=BG)
        kpi_row.pack(fill="x", padx=18, pady=(18, 8))

        kpis = [
            ("💰 Total Revenue",    f"${df['total_amount'].sum()/1e6:.2f}M",   POS),
            ("🧾 Total Orders",     f"{len(df):,}",                             ACCENT),
            ("👥 Unique Customers", f"{df['customer_id'].nunique():,}",         "#3498db"),
            ("⭐ Avg Rating",       f"{df['rating'].mean():.2f} / 5",          NEU),
            ("😊 Positive Reviews", f"{(df['sentiment']=='Positive').mean()*100:.1f}%", POS),
        ]
        for lbl, val, clr in kpis:
            kpi_card(kpi_row, lbl, val, color=clr).pack(side="left", padx=6)

        # Chart grid 3 x 2
        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="both", expand=True, padx=14, pady=4)
        for c in range(3): grid.columnconfigure(c, weight=1)
        for r in range(2): grid.rowconfigure(r, weight=1)

        embed_fig(grid, self._revenue_chart(df),    row=0, col=0)
        embed_fig(grid, self._category_chart(df),   row=0, col=1)
        embed_fig(grid, self._sentiment_pie(df),    row=0, col=2)
        embed_fig(grid, self._timeseries_chart(df), row=1, col=0, colspan=2, sticky="nsew")
        embed_fig(grid, self._ratings_chart(df),    row=1, col=2)

    def _revenue_chart(self, df):
        m = df.groupby(["year","month"])["total_amount"].sum().reset_index()
        m["period"] = pd.to_datetime(m[["year","month"]].assign(day=1))
        m.sort_values("period", inplace=True)
        fig = Figure(figsize=(4.2, 2.8), tight_layout=True)
        ax  = fig.add_subplot(111)
        ax.fill_between(m["period"], m["total_amount"], alpha=0.25, color=PALETTE[0])
        ax.plot(m["period"], m["total_amount"], color=PALETTE[0], lw=2)
        ax.set_title("Monthly Revenue", fontsize=9, fontweight="bold", pad=5)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.tick_params(axis="y", labelsize=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}K"))
        ax.grid(True, axis="y")
        return fig

    def _category_chart(self, df):
        cat = df.groupby("category")["total_amount"].sum().sort_values()
        fig = Figure(figsize=(4.2, 2.8), tight_layout=True)
        ax  = fig.add_subplot(111)
        ax.barh(cat.index, cat.values, color=PALETTE[:len(cat)],
                edgecolor="none", height=0.6)
        ax.set_title("Revenue by Category", fontsize=9, fontweight="bold", pad=5)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=6)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}K"))
        ax.grid(True, axis="x")
        return fig

    def _sentiment_pie(self, df):
        sc     = df["sentiment"].value_counts()
        colors = [POS if l=="Positive" else (NEG if l=="Negative" else NEU) for l in sc.index]
        fig = Figure(figsize=(4.2, 2.8), tight_layout=True)
        ax  = fig.add_subplot(111)
        wedges,_,auts = ax.pie(sc.values, labels=sc.index, colors=colors,
                               autopct="%1.1f%%", startangle=140,
                               wedgeprops={"edgecolor": BG, "linewidth": 2},
                               textprops={"fontsize": 7, "color": FG})
        for at in auts: at.set_fontsize(7)
        ax.set_title("Sentiment Split", fontsize=9, fontweight="bold", pad=5)
        return fig

    def _timeseries_chart(self, df):
        ts = (df.set_index("timestamp").resample("W")["total_amount"].sum()
                .reset_index().rename(columns={"timestamp":"week","total_amount":"rev"}))
        ts["roll4"] = ts["rev"].rolling(4, center=True).mean()
        fig = Figure(figsize=(8.8, 2.8), tight_layout=True)
        ax  = fig.add_subplot(111)
        ax.fill_between(ts["week"], ts["rev"], alpha=0.12, color=PALETTE[2])
        ax.plot(ts["week"], ts["rev"],   lw=0.9, alpha=0.5, color=PALETTE[2], label="Weekly")
        ax.plot(ts["week"], ts["roll4"], lw=2.2, color=PALETTE[0], label="4-Wk Avg")
        ax.set_title("Weekly Sales Trend", fontsize=9, fontweight="bold", pad=5)
        ax.tick_params(axis="x", labelsize=6, rotation=20)
        ax.tick_params(axis="y", labelsize=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.legend(fontsize=7); ax.grid(True, axis="y")
        return fig

    def _ratings_chart(self, df):
        rc = df["rating"].value_counts().sort_index()
        fig = Figure(figsize=(4.2, 2.8), tight_layout=True)
        ax  = fig.add_subplot(111)
        ax.bar(rc.index.astype(str), rc.values,
               color=PALETTE[:5], edgecolor="none", width=0.6)
        ax.set_title("Rating Distribution", fontsize=9, fontweight="bold", pad=5)
        ax.set_xlabel("Stars", fontsize=7); ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=7); ax.grid(True, axis="y")
        return fig


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Sentiment Analyzer
# ══════════════════════════════════════════════════════════════════════════════
class SentimentTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._history = []
        self._tally   = {"Positive": 0, "Neutral": 0, "Negative": 0}
        self._build()

    def _build(self):
        # Left – input panel
        left = tk.Frame(self, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(24, 12), pady=20)

        tk.Label(left, text="🔍  Customer Review Analyzer",
                 bg=BG, fg=FG, font=("Helvetica", 14, "bold")).pack(anchor="w")
        tk.Label(left, text="Paste a product review and get its sentiment in real time.",
                 bg=BG, fg=FG2, font=("Helvetica", 9)).pack(anchor="w", pady=(2, 10))

        wrap = tk.Frame(left, bg=CARD, highlightthickness=1,
                        highlightbackground=ACCENT)
        wrap.pack(fill="x")
        self._txt = tk.Text(wrap, height=9, wrap="word",
                            bg=CARD, fg=FG, insertbackground=FG,
                            font=("Helvetica", 11), relief="flat",
                            padx=10, pady=10, selectbackground=ACCENT)
        self._txt.pack(fill="both")
        self._ph = "Type or paste a customer review here…"
        self._txt.insert("1.0", self._ph); self._txt.config(fg=FG2)
        self._txt.bind("<FocusIn>",  self._clr_ph)
        self._txt.bind("<FocusOut>", self._set_ph)

        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=10)
        tk.Button(btn_row, text="  Predict Sentiment  →",
                  bg=ACCENT, fg="white", font=("Helvetica", 11, "bold"),
                  relief="flat", cursor="hand2",
                  activebackground=ACCENT2, activeforeground="white",
                  command=self._predict, padx=16, pady=9).pack(side="left")
        tk.Button(btn_row, text="  Clear  ",
                  bg=SURFACE, fg=FG2, font=("Helvetica", 10),
                  relief="flat", cursor="hand2",
                  command=self._clear, padx=12, pady=9).pack(side="left", padx=8)

        # Result card
        self._res = make_card(left)
        self._res.pack(fill="x", pady=4)
        self._emoji = tk.Label(self._res, text="", bg=CARD, font=("Helvetica", 40))
        self._emoji.pack(pady=(16, 2))
        self._slbl  = tk.Label(self._res, text="Awaiting input…",
                               bg=CARD, fg=FG2, font=("Helvetica", 16, "bold"))
        self._slbl.pack()
        self._scr   = tk.Label(self._res, text="", bg=CARD, fg=FG2, font=("Helvetica", 9))
        self._scr.pack(pady=(2, 16))

        # Right – history + tally donut
        right = tk.Frame(self, bg=BG)
        right.pack(side="right", fill="both", padx=(0, 24), pady=20)

        tk.Label(right, text="Prediction History",
                 bg=BG, fg=FG, font=("Helvetica", 10, "bold")).pack(anchor="w")
        hf = make_card(right)
        hf.pack(fill="both", expand=True, pady=(6, 12))
        self._hbox = tk.Text(hf, width=34, height=12, bg=CARD, fg=FG2,
                             font=("Courier", 9), relief="flat",
                             state="disabled", padx=8, pady=8)
        self._hbox.pack(fill="both", expand=True)

        tk.Label(right, text="Session Tally",
                 bg=BG, fg=FG, font=("Helvetica", 10, "bold")).pack(anchor="w")
        self._tfig   = Figure(figsize=(3.2, 2.4), tight_layout=True)
        self._tax    = self._tfig.add_subplot(111)
        self._tcanvas = FigureCanvasTkAgg(self._tfig, master=right)
        self._tcanvas.get_tk_widget().pack()
        self._draw_tally()

    def _clr_ph(self, _=None):
        if self._txt.get("1.0", "end-1c") == self._ph:
            self._txt.delete("1.0", "end"); self._txt.config(fg=FG)

    def _set_ph(self, _=None):
        if not self._txt.get("1.0", "end-1c").strip():
            self._txt.insert("1.0", self._ph); self._txt.config(fg=FG2)

    def _clear(self):
        self._txt.delete("1.0", "end"); self._set_ph()
        self._slbl.config(text="Awaiting input…", fg=FG2)
        self._emoji.config(text=""); self._scr.config(text="")
        self._res.config(highlightbackground=BORDER)

    def _predict(self):
        text = self._txt.get("1.0", "end-1c").strip()
        if not text or text == self._ph:
            messagebox.showwarning("Empty", "Please enter a review first."); return

        r     = predict_sentiment(text)
        label = r["label"]; score = r["compound"]
        cmap  = {"Positive": POS, "Neutral": NEU, "Negative": NEG}
        emap  = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
        color = cmap.get(label, FG); emoji = emap.get(label, "❓")

        self._res.config(highlightbackground=color)
        self._emoji.config(text=emoji, fg=color)
        self._slbl.config(text=label, fg=color)
        self._scr.config(text=f"Compound: {score:+.4f}  |  engine: {r['method']}")

        snip = (text[:40] + "…") if len(text) > 40 else text
        self._history.insert(0, f"{emoji} {label:<8}  {snip}\n")
        self._history = self._history[:20]
        self._hbox.config(state="normal")
        self._hbox.delete("1.0", "end")
        self._hbox.insert("1.0", "".join(self._history))
        self._hbox.config(state="disabled")

        self._tally[label] += 1
        self._draw_tally()

    def _draw_tally(self):
        self._tax.clear()
        vals   = list(self._tally.values())
        labels = list(self._tally.keys())
        colors = [POS, NEU, NEG]
        if sum(vals) == 0:
            vals = [1, 1, 1]; colors = [BORDER, BORDER, BORDER]
        _,_, auts = self._tax.pie(vals, labels=labels, colors=colors,
                                   autopct="%1.0f%%", startangle=90,
                                   wedgeprops={"edgecolor": BG, "linewidth": 2, "width": 0.55},
                                   textprops={"fontsize": 7, "color": FG})
        for at in auts: at.set_fontsize(7)
        self._tax.set_title("Session Sentiments", fontsize=8, fontweight="bold", color=FG)
        self._tcanvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – ML Purchase Predictor
# ══════════════════════════════════════════════════════════════════════════════
class MLPredictTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._model = self._encoders = self._scaler = None
        self._load_model()
        self._build()

    def _load_model(self):
        try:
            with open(os.path.join(MDL_DIR, "random_forest.pkl"), "rb") as f:
                self._model = pickle.load(f)
            with open(os.path.join(MDL_DIR, "artefacts.pkl"), "rb") as f:
                art = pickle.load(f)
            self._encoders = art["encoders"]
            self._scaler   = art["scaler"]
        except Exception:
            self._model = None

    def _build(self):
        df = DataStore.load()

        # Left – form
        left = tk.Frame(self, bg=BG)
        left.pack(side="left", fill="y", padx=(24, 12), pady=20)

        tk.Label(left, text="🤖  Purchase Likelihood Predictor",
                 bg=BG, fg=FG, font=("Helvetica", 14, "bold")).pack(anchor="w")
        tk.Label(left, text="Fill in a customer profile to predict re-purchase probability.",
                 bg=BG, fg=FG2, font=("Helvetica", 9)).pack(anchor="w", pady=(2, 12))

        form = make_card(left); form.pack(fill="x")
        self._vars = {}

        fields = [
            ("Age",             "spinbox", {"from_": 18, "to": 80, "width": 14}),
            ("Gender",          "combo",   df["gender"].unique().tolist()),
            ("Loyalty Tier",    "combo",   ["Bronze","Silver","Gold","Platinum"]),
            ("Category",        "combo",   sorted(df["category"].unique().tolist())),
            ("Payment Method",  "combo",   df["payment_method"].unique().tolist()),
            ("Price ($)",       "entry",   {}),
            ("Quantity",        "spinbox", {"from_": 1, "to": 20, "width": 14}),
            ("Rating (1-5)",    "combo",   ["1","2","3","4","5"]),
        ]
        for lbl, wtype, opts in fields:
            row = tk.Frame(form, bg=CARD); row.pack(fill="x", padx=14, pady=5)
            tk.Label(row, text=lbl, bg=CARD, fg=FG2,
                     font=("Helvetica", 9), width=16, anchor="w").pack(side="left")
            var = tk.StringVar()
            if wtype == "combo":
                w = ttk.Combobox(row, values=opts, textvariable=var,
                                 state="readonly", width=18, font=("Helvetica", 9))
                w.current(0)
            elif wtype == "spinbox":
                w = tk.Spinbox(row, textvariable=var, bg=SURFACE, fg=FG,
                               relief="flat", font=("Helvetica", 9),
                               insertbackground=FG, highlightthickness=1,
                               highlightbackground=BORDER, **opts)
                var.set(str(opts.get("from_", 1)))
            else:
                w = tk.Entry(row, textvariable=var, bg=SURFACE, fg=FG,
                             insertbackground=FG, relief="flat",
                             highlightthickness=1, highlightbackground=BORDER,
                             font=("Helvetica", 9), width=20)
                var.set("49.99")
            w.pack(side="left")
            self._vars[lbl] = var

        tk.Button(left, text="  Predict Purchase Likelihood  →",
                  bg=ACCENT, fg="white", font=("Helvetica", 11, "bold"),
                  relief="flat", cursor="hand2",
                  activebackground=ACCENT2, activeforeground="white",
                  command=self._predict, padx=16, pady=9).pack(pady=14)

        res = make_card(left); res.pack(fill="x")
        self._prob_lbl = tk.Label(res, text="—", bg=CARD, fg=ACCENT,
                                  font=("Helvetica", 32, "bold"))
        self._prob_lbl.pack(pady=(14, 2))
        self._pred_lbl = tk.Label(res, text="Fill form and click Predict",
                                  bg=CARD, fg=FG2, font=("Helvetica", 12))
        self._pred_lbl.pack(pady=(0, 14))

        # Right – model perf + gauge
        right = tk.Frame(self, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(0, 24), pady=20)

        tk.Label(right, text="Model Performance Summary",
                 bg=BG, fg=FG, font=("Helvetica", 10, "bold")).pack(anchor="w")
        c = FigureCanvasTkAgg(self._perf_fig(), master=right)
        c.draw(); c.get_tk_widget().pack(fill="both", expand=True, pady=6)

        tk.Label(right, text="Prediction Gauge",
                 bg=BG, fg=FG, font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self._gfig   = Figure(figsize=(4.5, 2.8), tight_layout=True)
        self._gax    = self._gfig.add_subplot(111)
        self._gcanvas = FigureCanvasTkAgg(self._gfig, master=right)
        self._gcanvas.get_tk_widget().pack(fill="x")
        self._draw_gauge(0.5)

    def _perf_fig(self):
        models  = ["Logistic\nRegression", "Decision\nTree", "Random\nForest"]
        metrics = {
            "Accuracy":  [0.732, 0.725, 0.732],
            "Precision": [0.723, 0.721, 0.723],
            "Recall":    [0.846, 0.830, 0.845],
            "F1":        [0.780, 0.772, 0.779],
        }
        fig = Figure(figsize=(4.5, 3.2), tight_layout=True)
        ax  = fig.add_subplot(111)
        x = np.arange(len(models)); w = 0.18
        for i, (mn, vals) in enumerate(metrics.items()):
            ax.bar(x + i*w, vals, width=w, label=mn,
                   color=PALETTE[i], alpha=0.88, edgecolor="none")
        ax.set_xticks(x + w*1.5); ax.set_xticklabels(models, fontsize=7)
        ax.set_ylim(0.6, 0.95); ax.set_ylabel("Score", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_title("Model Comparison", fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, ncol=2); ax.grid(True, axis="y")
        return fig

    def _draw_gauge(self, prob):
        ax = self._gax; ax.clear()
        theta      = np.linspace(np.pi, 0, 200)
        fill_theta = np.linspace(np.pi, np.pi - prob * np.pi, 200)
        clr        = POS if prob >= 0.6 else (NEU if prob >= 0.4 else NEG)
        ax.plot(np.cos(theta),      np.sin(theta),      lw=18, color=CARD,   solid_capstyle="round")
        ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=18, color=clr,    solid_capstyle="round")
        angle = np.pi - prob * np.pi
        ax.annotate("", xy=(0.55*np.cos(angle), 0.55*np.sin(angle)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=FG, lw=2.5))
        ax.text(0, -0.22, f"{prob*100:.1f}%", ha="center", va="center",
                fontsize=16, fontweight="bold", color=clr)
        ax.text(0, -0.46, "Purchase Probability", ha="center", va="center",
                fontsize=7, color=FG2)
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 1.1); ax.axis("off")
        self._gfig.set_facecolor(SURFACE); self._gcanvas.draw()

    def _predict(self):
        if self._model is None:
            messagebox.showerror("Model Missing",
                "Run:  python scripts/train_models.py  first."); return
        try:
            from sklearn.impute import SimpleImputer
            df  = DataStore.load()
            enc = self._encoders

            age      = int(self._vars["Age"].get())
            gender   = self._vars["Gender"].get()
            loyalty  = self._vars["Loyalty Tier"].get()
            category = self._vars["Category"].get()
            payment  = self._vars["Payment Method"].get()
            price    = float(self._vars["Price ($)"].get())
            quantity = int(self._vars["Quantity"].get())
            rating   = float(self._vars["Rating (1-5)"].get())
            now      = df["timestamp"].max()

            def enc_safe(le, v):
                return le.transform([v])[0] if v in le.classes_ else 0

            row = np.array([[
                age,
                enc_safe(enc["gender"],   gender),
                enc_safe(enc["location"], df["location"].mode()[0]),
                enc_safe(enc["loyalty"],  loyalty),
                enc_safe(enc["category"], category),
                enc_safe(enc["payment"],  payment),
                price, quantity, price * quantity, rating,
                now.year, now.month, now.dayofweek,
            ]])
            row = SimpleImputer(strategy="median").fit_transform(row)
            row = self._scaler.transform(row)
            prob = self._model.predict_proba(row)[0][1]
            clr  = POS if prob >= 0.5 else NEG
            pred = "✓  Will Re-Purchase" if prob >= 0.5 else "✗  Unlikely to Re-Purchase"
            self._prob_lbl.config(text=f"{prob*100:.1f}%", fg=clr)
            self._pred_lbl.config(text=pred, fg=clr)
            self._draw_gauge(prob)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
class DataExplorerTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._df_full = None
        self._cols    = ["transaction_id","customer_id","category","product",
                         "price","quantity","total_amount","rating",
                         "sentiment","payment_method","timestamp"]
        self._build()

    def _build(self):
        df = DataStore.load()
        self._df_full = df

        # Toolbar
        tb = tk.Frame(self, bg=BG)
        tb.pack(fill="x", padx=18, pady=(14, 6))
        tk.Label(tb, text="📁  Transaction Explorer",
                 bg=BG, fg=FG, font=("Helvetica", 13, "bold")).pack(side="left")

        tk.Label(tb, text="Search:", bg=BG, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(20, 4))
        self._sq = tk.StringVar(); self._sq.trace_add("write", self._refresh)
        tk.Entry(tb, textvariable=self._sq, bg=SURFACE, fg=FG,
                 insertbackground=FG, relief="flat",
                 highlightthickness=1, highlightbackground=ACCENT,
                 font=("Helvetica", 10), width=22).pack(side="left")

        tk.Label(tb, text="Category:", bg=BG, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(14, 4))
        self._cq = tk.StringVar(value="All")
        self._cq.trace_add("write", self._refresh)
        ttk.Combobox(tb, values=["All"] + sorted(df["category"].unique().tolist()),
                     textvariable=self._cq, state="readonly",
                     width=16, font=("Helvetica", 9)).pack(side="left")

        tk.Label(tb, text="Sentiment:", bg=BG, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(14, 4))
        self._sentq = tk.StringVar(value="All")
        self._sentq.trace_add("write", self._refresh)
        ttk.Combobox(tb, values=["All","Positive","Neutral","Negative"],
                     textvariable=self._sentq, state="readonly",
                     width=12, font=("Helvetica", 9)).pack(side="left")

        self._count_lbl = tk.Label(tb, text="", bg=BG, fg=FG2, font=("Helvetica", 9))
        self._count_lbl.pack(side="right")

        # Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("E.Treeview",
                        background=CARD, foreground=FG,
                        fieldbackground=CARD, rowheight=24, borderwidth=0)
        style.configure("E.Treeview.Heading",
                        background=SURFACE, foreground=FG2,
                        relief="flat", font=("Helvetica", 8, "bold"))
        style.map("E.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

        tf = tk.Frame(self, bg=BG)
        tf.pack(fill="both", expand=True, padx=18, pady=(0, 14))
        vsb = ttk.Scrollbar(tf, orient="vertical")
        hsb = ttk.Scrollbar(tf, orient="horizontal")
        self._tree = ttk.Treeview(tf, columns=self._cols, show="headings",
                                  style="E.Treeview",
                                  yscrollcommand=vsb.set,
                                  xscrollcommand=hsb.set)
        vsb.config(command=self._tree.yview)
        hsb.config(command=self._tree.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)

        widths = {"transaction_id":95,"customer_id":95,"category":105,
                  "product":130,"price":70,"quantity":60,"total_amount":95,
                  "rating":55,"sentiment":82,"payment_method":105,"timestamp":135}
        for col in self._cols:
            self._tree.heading(col, text=col.replace("_"," ").title(),
                               command=lambda c=col: self._sort(c))
            self._tree.column(col, width=widths.get(col, 90),
                              anchor="center" if col in ["price","quantity",
                              "total_amount","rating"] else "w")
        self._populate(df[self._cols])

    def _populate(self, df):
        self._tree.delete(*self._tree.get_children())
        for _, row in df.head(1000).iterrows():
            tag = row.get("sentiment","")
            self._tree.insert("", "end", values=[row[c] for c in self._cols], tags=(tag,))
        self._tree.tag_configure("Positive", foreground=POS)
        self._tree.tag_configure("Negative", foreground=NEG)
        self._tree.tag_configure("Neutral",  foreground=NEU)
        self._count_lbl.config(text=f"Showing {min(len(df),1000):,} of {len(df):,} rows")

    def _refresh(self, *_):
        q    = self._sq.get().lower().strip()
        cat  = self._cq.get()
        sent = self._sentq.get()
        df   = self._df_full[self._cols].copy()
        if cat  != "All": df = df[df["category"] == cat]
        if sent != "All": df = df[df["sentiment"] == sent]
        if q:
            mask = df.apply(lambda r: q in " ".join(r.astype(str)).lower(), axis=1)
            df   = df[mask]
        self._populate(df)

    def _sort(self, col):
        df = self._df_full[self._cols].copy()
        try: df = df.sort_values(col)
        except Exception: pass
        self._populate(df)


# ══════════════════════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🛒  E-Commerce Analytics Platform")
        self.geometry("1200x780")
        self.minsize(1100, 720)
        self.configure(bg=BG)
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=ACCENT, height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🛒  E-Commerce Customer Behavior Analytics Platform",
                 bg=ACCENT, fg="white",
                 font=("Helvetica", 14, "bold")).pack(side="left", padx=22, pady=14)
        tk.Label(hdr, text="Powered by Machine Learning  ·  Real-time Insights",
                 bg=ACCENT, fg="#c8c4ff",
                 font=("Helvetica", 9)).pack(side="right", padx=22)

        # Notebook
        style = ttk.Style(); style.theme_use("clam")
        style.configure("App.TNotebook", background=BG, borderwidth=0)
        style.configure("App.TNotebook.Tab",
                        background=SURFACE, foreground=FG2,
                        padding=[16, 8], font=("Helvetica", 10), borderwidth=0)
        style.map("App.TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])

        nb = ttk.Notebook(self, style="App.TNotebook")
        nb.pack(fill="both", expand=True)

        nb.add(DashboardTab(nb),   text="  📊 Dashboard  ")
        nb.add(SentimentTab(nb),   text="  🔍 Sentiment Analyzer  ")
        nb.add(MLPredictTab(nb),   text="  🤖 ML Predictor  ")
        nb.add(DataExplorerTab(nb),text="  📁 Data Explorer  ")

        # Status bar
        sb = tk.Frame(self, bg=SURFACE, height=24)
        sb.pack(fill="x", side="bottom")
        df = DataStore.load()
        tk.Label(sb,
                 text=(f"  Dataset: {len(df):,} transactions  ·  "
                       f"{df['customer_id'].nunique():,} customers  ·  "
                       f"{df['category'].nunique()} categories  ·  "
                       f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}"),
                 bg=SURFACE, fg=FG2, font=("Helvetica", 8)).pack(side="left", pady=4)


if __name__ == "__main__":
    print("Pre-loading dataset…")
    DataStore.load()
    print("Launching app…")
    App().mainloop()