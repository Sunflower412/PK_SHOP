import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


BASE_DIR = Path(__file__).resolve().parent
SALES_PATH = BASE_DIR / "data" / "PC_shop.csv"

FEATURES = [
    "avg_pc_price",
    "discount_percent",
    "is_new_product_launch",
    "website_traffic",
    "marketing_budget",
]
TARGET = "monthly_sales_units"

LABELS = {
    "avg_pc_price": "Цена",
    "discount_percent": "Скидка",
    "is_new_product_launch": "Новинка",
    "website_traffic": "Трафик",
    "marketing_budget": "Реклама",
    "monthly_sales_units": "Продажи",
}


def load_sales():
    rows = []

    with open(SALES_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({
                column: float(row[column])
                for column in FEATURES + [TARGET]
            })

    return rows


def average(values):
    return sum(values) / len(values) if values else 0


class ShopDashboard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.sales = load_sales()

        self.title("Панель магазина")
        self.geometry("1060x640")
        self.minsize(940, 560)

        self.configure(bg="#edf2f7")
        self.configure_style()
        self.create_layout()
        self.show_overview()

    def configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Sidebar.TFrame", background="#16213e")
        style.configure("Page.TFrame", background="#edf2f7")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("Title.TLabel", background="#edf2f7", foreground="#17202a", font=("Segoe UI", 18, "bold"))
        style.configure("CardName.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 10))
        style.configure("CardValue.TLabel", background="#ffffff", foreground="#111827", font=("Segoe UI", 18, "bold"))
        style.configure("Sidebar.TButton", background="#16213e", foreground="#ffffff", font=("Segoe UI", 10), padding=10)
        style.map("Sidebar.TButton", background=[("active", "#23365f")])

    def create_layout(self):
        self.sidebar = ttk.Frame(self, style="Sidebar.TFrame", width=190)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        tk.Label(
            self.sidebar,
            text="SHOP\nCONTROL",
            bg="#16213e",
            fg="#ffffff",
            font=("Segoe UI", 17, "bold"),
            justify="left",
        ).pack(anchor="w", padx=18, pady=(20, 28))

        ttk.Button(
            self.sidebar,
            text="Обзор",
            style="Sidebar.TButton",
            command=self.show_overview,
        ).pack(fill=tk.X, padx=12, pady=4)

        self.page = ttk.Frame(self, style="Page.TFrame")
        self.page.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def clear_page(self):
        for widget in self.page.winfo_children():
            widget.destroy()

    def show_overview(self):
        self.clear_page()

        ttk.Label(
            self.page,
            text="Обзор продаж",
            style="Title.TLabel",
        ).pack(anchor="w", padx=24, pady=(22, 12))

        cards = ttk.Frame(self.page, style="Page.TFrame")
        cards.pack(fill=tk.X, padx=24)

        values = {
            "Строк": len(self.sales),
            "Средние продажи": average([row[TARGET] for row in self.sales]),
            "Средняя цена": average([row["avg_pc_price"] for row in self.sales]),
            "Средний бюджет": average([row["marketing_budget"] for row in self.sales]),
        }

        for index, (name, value) in enumerate(values.items()):
            card = ttk.Frame(cards, style="Card.TFrame", padding=14)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 10, 0))
            ttk.Label(card, text=name, style="CardName.TLabel").pack(anchor="w")
            ttk.Label(card, text=f"{value:.0f}", style="CardValue.TLabel").pack(anchor="w", pady=(6, 0))
            cards.columnconfigure(index, weight=1)

        figure = Figure(figsize=(8, 4.5), dpi=100, facecolor="#edf2f7")
        axes = figure.add_subplot(111)
        axes.scatter(
            [row["website_traffic"] for row in self.sales],
            [row[TARGET] for row in self.sales],
            color="#2563eb",
            alpha=0.75,
        )
        axes.set_title("Продажи и посещаемость сайта")
        axes.set_xlabel("Трафик сайта, тыс.")
        axes.set_ylabel("Продажи, шт.")
        axes.grid(True, color="#cbd5e1")
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, self.page)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=24, pady=20)


def main():
    app = ShopDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
