import csv
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


BASE_DIR = Path(__file__).resolve().parent
SALES_PATH = BASE_DIR / "data" / "PC_shop.csv"
PRODUCTS_PATH = BASE_DIR / "data" / "products.json"

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

QUALITY_METRICS = {
    "performance": ("Производительность", 0.28, "max"),
    "reliability": ("Надежность", 0.18, "max"),
    "energy_efficiency": ("Энергоэффективность", 0.14, "max"),
    "warranty": ("Гарантия", 0.12, "max"),
    "service": ("Сервис", 0.16, "max"),
    "price": ("Цена", 0.12, "min"),
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


def load_products():
    with open(PRODUCTS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def average(values):
    return sum(values) / len(values) if values else 0


def calculate_quality(products):
    max_values = {}
    min_values = {}
    result = []

    for key in QUALITY_METRICS:
        values = [product[key] for product in products]
        max_values[key] = max(values)
        min_values[key] = min(values)

    # Считаем общий показатель качества товара.
    for product in products:
        score = 0

        for key, settings in QUALITY_METRICS.items():
            _, weight, direction = settings
            value = product[key]

            if direction == "max":
                normalized = value / max_values[key]
            else:
                normalized = min_values[key] / value

            score += normalized * weight

        item = dict(product)
        item["quality"] = score
        result.append(item)

    result.sort(key=lambda item: item["quality"], reverse=True)
    return result


def normalized_profile(product, products):
    names = []
    values = []

    for key, settings in QUALITY_METRICS.items():
        name, _, direction = settings
        all_values = [item[key] for item in products]

        if direction == "max":
            value = product[key] / max(all_values)
        else:
            value = min(all_values) / product[key]

        names.append(name)
        values.append(value)

    return names, values


class ShopDashboard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.sales = load_sales()
        self.products = load_products()
        self.quality = calculate_quality(self.products)

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
        ttk.Button(
            self.sidebar,
            text="Качество",
            style="Sidebar.TButton",
            command=self.show_quality,
        ).pack(fill=tk.X, padx=12, pady=4)
        ttk.Button(
            self.sidebar,
            text="Профиль товара",
            style="Sidebar.TButton",
            command=self.show_profile,
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

    def show_quality(self):
        self.clear_page()

        ttk.Label(
            self.page,
            text="Оценка качества товаров",
            style="Title.TLabel",
        ).pack(anchor="w", padx=24, pady=(22, 12))

        body = ttk.Frame(self.page, style="Page.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 20))

        table_frame = ttk.Frame(body)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        columns = ("place", "name", "type", "quality", "price")
        table = ttk.Treeview(table_frame, columns=columns, show="headings")
        table.heading("place", text="№")
        table.heading("name", text="Товар")
        table.heading("type", text="Группа")
        table.heading("quality", text="Q")
        table.heading("price", text="Цена")

        table.column("place", width=45)
        table.column("name", width=260)
        table.column("type", width=130)
        table.column("quality", width=80)
        table.column("price", width=110)

        for index, item in enumerate(self.quality, start=1):
            table.insert(
                "",
                tk.END,
                values=(
                    index,
                    item["name"],
                    item["type"],
                    round(item["quality"], 3),
                    item["price"],
                ),
            )

        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
        table.configure(yscrollcommand=scroll.set)
        table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        figure = Figure(figsize=(4.8, 4.4), dpi=100, facecolor="#edf2f7")
        axes = figure.add_subplot(111)

        best_items = self.quality[:10]
        labels = [item["name"].split(" ", 1)[-1] for item in reversed(best_items)]
        scores = [item["quality"] for item in reversed(best_items)]
        axes.barh(labels, scores, color="#0f766e")
        axes.set_title("Лучшие позиции")
        axes.set_xlim(0, 1)
        axes.grid(True, axis="x", color="#cbd5e1")
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, body)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def show_profile(self):
        self.clear_page()

        ttk.Label(
            self.page,
            text="Профиль товара",
            style="Title.TLabel",
        ).pack(anchor="w", padx=24, pady=(22, 12))

        controls = ttk.Frame(self.page, style="Page.TFrame")
        controls.pack(fill=tk.X, padx=24)

        product_names = [product["name"] for product in self.products]
        selected = tk.StringVar(value=product_names[0])

        box = ttk.Combobox(
            controls,
            textvariable=selected,
            values=product_names,
            state="readonly",
            width=44,
        )
        box.pack(side=tk.LEFT)

        chart_frame = ttk.Frame(self.page, style="Page.TFrame")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=18)

        def draw_profile():
            for widget in chart_frame.winfo_children():
                widget.destroy()

            product = next(
                item for item in self.products
                if item["name"] == selected.get()
            )

            names, values = normalized_profile(product, self.products)
            values.append(values[0])

            angles = []
            for index in range(len(names)):
                angles.append(2 * 3.14159 * index / len(names))
            angles.append(angles[0])

            figure = Figure(figsize=(7, 4.8), dpi=100, facecolor="#edf2f7")
            axes = figure.add_subplot(111, polar=True)
            axes.plot(angles, values, color="#2563eb", linewidth=2)
            axes.fill(angles, values, color="#2563eb", alpha=0.22)
            axes.set_xticks(angles[:-1])
            axes.set_xticklabels(names, fontsize=8)
            axes.set_ylim(0, 1)
            axes.set_title(product["name"])
            axes.grid(True)
            figure.tight_layout()

            canvas = FigureCanvasTkAgg(figure, chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Button(
            controls,
            text="Показать",
            command=draw_profile,
        ).pack(side=tk.LEFT, padx=10)

        draw_profile()


def main():
    app = ShopDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
