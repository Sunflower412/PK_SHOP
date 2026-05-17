import csv
import json
from itertools import combinations
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


BASE_DIR = Path(__file__).resolve().parent
SALES_PATH = BASE_DIR / "data" / "PC_shop.csv"
PRODUCTS_PATH = BASE_DIR / "data" / "products.json"
INFO_PATH = BASE_DIR / "data" / "PC_shop_info.txt"

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


def load_sales_from_path(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = [
            column for column in FEATURES + [TARGET]
            if column not in (reader.fieldnames or [])
        ]
        if missing_columns:
            raise ValueError("Нет колонок: " + ", ".join(missing_columns))
        for row in reader:
            rows.append({
                column: float(row[column])
                for column in FEATURES + [TARGET]
            })

    return rows


def load_sales():
    return load_sales_from_path(SALES_PATH)


def load_info():
    if INFO_PATH.exists():
        return INFO_PATH.read_text(encoding="utf-8")
    return ""


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
        item["profit"] = (
            item["price"] - item["stock_cost"]
        ) * item["expected_sales"]
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


def train_model(sales):
    x_values = np.array(
        [[row[column] for column in FEATURES] for row in sales],
        dtype=float,
    )
    y_values = np.array([row[TARGET] for row in sales], dtype=float)

    rng = np.random.default_rng(42)
    order = rng.permutation(len(y_values))
    test_count = max(5, round(len(y_values) * 0.2))
    test_index = order[:test_count]
    train_index = order[test_count:]

    x_train = x_values[train_index]
    y_train = y_values[train_index]
    x_test = x_values[test_index]
    y_test = y_values[test_index]

    means = x_train.mean(axis=0)
    stds = x_train.std(axis=0)
    stds[stds == 0] = 1

    scaled = (x_train - means) / stds
    design = np.column_stack([np.ones(len(scaled)), scaled])

    # Небольшая регуляризация делает модель устойчивее.
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.pinv(design.T @ design + 0.8 * penalty) @ design.T @ y_train

    predicted = predict_array(x_test, means, stds, beta)
    errors = y_test - predicted

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0

    return {
        "means": means,
        "stds": stds,
        "beta": beta,
        "actual": [float(value) for value in y_test],
        "predicted": [float(value) for value in predicted],
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
    }


def predict_array(x_values, means, stds, beta):
    scaled = (x_values - means) / stds
    design = np.column_stack([np.ones(len(scaled)), scaled])
    return design @ beta


def predict_sales(model, values):
    x_values = np.array(
        [[values[column] for column in FEATURES]],
        dtype=float,
    )
    prediction = predict_array(
        x_values,
        model["means"],
        model["stds"],
        model["beta"],
    )[0]
    return max(float(prediction), 0)


def correlation_matrix(sales):
    values = np.array(
        [[row[column] for column in FEATURES + [TARGET]] for row in sales],
        dtype=float,
    )
    return np.corrcoef(values, rowvar=False)


def find_best_set(products, budget, required_sales, selected_type):
    candidates = [
        item for item in calculate_quality(products)
        if item["type"] == selected_type
    ]

    best_set = None
    best_score = -1

    for count in range(1, len(candidates) + 1):
        for group in combinations(candidates, count):
            total_cost = sum(item["stock_cost"] for item in group)
            total_sales = sum(item["expected_sales"] for item in group)

            if total_cost > budget:
                continue
            if total_sales < required_sales:
                continue

            score = sum(
                item["quality"] * item["expected_sales"]
                + item["profit"] / 100000
                for item in group
            )

            if score > best_score:
                best_score = score
                best_set = group

    if not best_set:
        return None

    return {
        "products": best_set,
        "total_cost": sum(item["stock_cost"] for item in best_set),
        "total_sales": sum(item["expected_sales"] for item in best_set),
        "total_profit": sum(item["profit"] for item in best_set),
        "average_quality": average([item["quality"] for item in best_set]),
    }


def format_money(value):
    return f"{value:,.0f}".replace(",", " ")


class ShopDashboard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.sales = load_sales()
        self.products = load_products()
        self.quality = calculate_quality(self.products)
        self.model = train_model(self.sales)

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
            text="Данные",
            style="Sidebar.TButton",
            command=self.show_data,
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
        ttk.Button(
            self.sidebar,
            text="Прогноз",
            style="Sidebar.TButton",
            command=self.show_forecast,
        ).pack(fill=tk.X, padx=12, pady=4)
        ttk.Button(
            self.sidebar,
            text="Подбор",
            style="Sidebar.TButton",
            command=self.show_optimizer,
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

    def show_data(self):
        self.clear_page()

        ttk.Label(
            self.page,
            text="Исходные данные",
            style="Title.TLabel",
        ).pack(anchor="w", padx=24, pady=(22, 12))

        toolbar = ttk.Frame(self.page, style="Page.TFrame")
        toolbar.pack(fill=tk.X, padx=24)

        status = tk.StringVar(value=f"{len(self.sales)} строк загружено")

        def choose_file():
            path = filedialog.askopenfilename(
                filetypes=[("CSV", "*.csv"), ("Все файлы", "*.*")]
            )
            if not path:
                return

            try:
                self.sales = load_sales_from_path(Path(path))
                self.model = train_model(self.sales)
            except Exception as error:
                messagebox.showerror("Ошибка", str(error))
                return

            status.set(f"{len(self.sales)} строк загружено")
            self.show_data()

        ttk.Button(toolbar, text="Загрузить CSV", command=choose_file).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=status).pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(self.page, style="Page.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=14)

        table_frame = ttk.Frame(body)
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        columns = FEATURES + [TARGET]
        table = ttk.Treeview(table_frame, columns=columns, show="headings")
        for column in columns:
            table.heading(column, text=LABELS[column])
            table.column(column, width=115, anchor=tk.CENTER)

        for row in self.sales:
            table.insert(
                "",
                tk.END,
                values=tuple(int(row[column]) for column in columns),
            )

        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=table.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=table.xview)
        table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        info = tk.Text(
            body,
            width=34,
            wrap=tk.WORD,
            bg="#ffffff",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
        )
        info.insert(tk.END, load_info())
        info.configure(state=tk.DISABLED)
        info.pack(side=tk.LEFT, fill=tk.BOTH)

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

    def show_forecast(self):
        self.clear_page()

        ttk.Label(
            self.page,
            text="Прогноз продаж",
            style="Title.TLabel",
        ).pack(anchor="w", padx=24, pady=(22, 12))

        body = ttk.Frame(self.page, style="Page.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 20))

        panel = ttk.Frame(body, style="Card.TFrame", padding=16)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))

        defaults = {
            column: average([row[column] for row in self.sales])
            for column in FEATURES
        }
        defaults["is_new_product_launch"] = 1

        entries = {}

        for column in FEATURES:
            ttk.Label(panel, text=LABELS[column], style="CardName.TLabel").pack(anchor="w", pady=(4, 2))
            entry = ttk.Entry(panel, width=18)
            entry.insert(0, round(defaults[column], 2))
            entry.pack(fill=tk.X)
            entries[column] = entry

        feature_var = tk.StringVar(value="marketing_budget")

        ttk.Label(panel, text="График по признаку", style="CardName.TLabel").pack(anchor="w", pady=(12, 2))
        ttk.Combobox(
            panel,
            textvariable=feature_var,
            values=FEATURES,
            state="readonly",
            width=18,
        ).pack(fill=tk.X)

        result_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=result_var, style="CardValue.TLabel").pack(anchor="w", pady=(14, 4))

        chart_frame = ttk.Frame(body, style="Page.TFrame")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def read_values():
            values = {}

            for column in FEATURES:
                values[column] = float(entries[column].get().replace(",", "."))

            values["is_new_product_launch"] = (
                1 if values["is_new_product_launch"] >= 0.5 else 0
            )
            return values

        def draw():
            for widget in chart_frame.winfo_children():
                widget.destroy()

            try:
                values = read_values()
            except ValueError:
                messagebox.showerror("Ошибка", "Введите числовые значения.")
                return

            prediction = predict_sales(self.model, values)
            result_var.set(f"{prediction:.0f} шт.")

            feature = feature_var.get()

            if feature == "is_new_product_launch":
                x_values = [0, 1]
            else:
                column_values = [row[feature] for row in self.sales]
                x_values = np.linspace(min(column_values), max(column_values), 32)

            y_values = []
            for value in x_values:
                row = dict(values)
                row[feature] = float(value)
                if feature == "is_new_product_launch":
                    row[feature] = 1 if value >= 0.5 else 0
                y_values.append(predict_sales(self.model, row))

            figure = Figure(figsize=(8.5, 5.3), dpi=100, facecolor="#edf2f7")
            ax1 = figure.add_subplot(2, 2, 1)
            ax2 = figure.add_subplot(2, 2, 2)
            ax3 = figure.add_subplot(2, 1, 2)

            numbers = list(range(1, len(self.model["actual"]) + 1))
            ax1.plot(numbers, self.model["actual"], marker="o", label="Факт")
            ax1.plot(numbers, self.model["predicted"], marker="s", label="Прогноз")
            ax1.set_title("Проверка модели")
            ax1.grid(True, color="#cbd5e1")
            ax1.legend(fontsize=8)

            ax2.plot(x_values, y_values, color="#0f766e", marker="o")
            ax2.set_title(f"Влияние: {LABELS[feature]}")
            ax2.grid(True, color="#cbd5e1")

            matrix = correlation_matrix(self.sales)
            image = ax3.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
            names = [LABELS[column] for column in FEATURES + [TARGET]]
            ax3.set_title("Матрица корреляций")
            ax3.set_xticks(range(len(names)))
            ax3.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax3.set_yticks(range(len(names)))
            ax3.set_yticklabels(names, fontsize=8)

            for row_index in range(len(names)):
                for column_index in range(len(names)):
                    ax3.text(
                        column_index,
                        row_index,
                        f"{matrix[row_index, column_index]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

            figure.colorbar(image, ax=ax3, fraction=0.03, pad=0.03)
            figure.tight_layout()

            canvas = FigureCanvasTkAgg(figure, chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Button(panel, text="Рассчитать", command=draw).pack(fill=tk.X, pady=(10, 6))
        ttk.Label(
            panel,
            text=f"MAE: {self.model['mae']:.1f}\nRMSE: {self.model['rmse']:.1f}\nR2: {self.model['r2']:.3f}",
            style="CardName.TLabel",
        ).pack(anchor="w")

        draw()

    def show_optimizer(self):
        self.clear_page()

        ttk.Label(
            self.page,
            text="Подбор ассортимента",
            style="Title.TLabel",
        ).pack(anchor="w", padx=24, pady=(22, 12))

        body = ttk.Frame(self.page, style="Page.TFrame")
        body.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0, 20))

        panel = ttk.Frame(body, style="Card.TFrame", padding=16)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))

        product_types = sorted({product["type"] for product in self.products})
        selected_type = tk.StringVar(value=product_types[0])

        ttk.Label(panel, text="Группа товаров", style="CardName.TLabel").pack(anchor="w")
        ttk.Combobox(
            panel,
            textvariable=selected_type,
            values=product_types,
            state="readonly",
            width=22,
        ).pack(fill=tk.X, pady=(2, 8))

        ttk.Label(panel, text="Бюджет закупки", style="CardName.TLabel").pack(anchor="w")
        budget_entry = ttk.Entry(panel, width=18)
        budget_entry.insert(0, "500000")
        budget_entry.pack(fill=tk.X, pady=(2, 8))

        ttk.Label(panel, text="Минимум продаж", style="CardName.TLabel").pack(anchor="w")
        sales_entry = ttk.Entry(panel, width=18)
        sales_entry.insert(0, "120")
        sales_entry.pack(fill=tk.X, pady=(2, 8))

        summary = tk.StringVar(value="")
        ttk.Label(
            panel,
            textvariable=summary,
            style="CardName.TLabel",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(12, 0))

        result_area = ttk.Frame(body, style="Page.TFrame")
        result_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def draw_result():
            for widget in result_area.winfo_children():
                widget.destroy()

            try:
                budget = float(budget_entry.get().replace(",", "."))
                required_sales = float(sales_entry.get().replace(",", "."))
            except ValueError:
                messagebox.showerror("Ошибка", "Введите числовые значения.")
                return

            result = find_best_set(
                self.products,
                budget,
                required_sales,
                selected_type.get(),
            )

            if result is None:
                summary.set("Набор не найден.\nИзмените ограничения.")
                return

            summary.set(
                f"Закупка: {format_money(result['total_cost'])} руб.\n"
                f"Продажи: {result['total_sales']} шт.\n"
                f"Маржа: {format_money(result['total_profit'])} руб.\n"
                f"Средний Q: {result['average_quality']:.3f}"
            )

            columns = ("name", "quality", "sales", "cost", "profit")
            table = ttk.Treeview(result_area, columns=columns, show="headings", height=8)
            table.heading("name", text="Товар")
            table.heading("quality", text="Q")
            table.heading("sales", text="Продажи")
            table.heading("cost", text="Закупка")
            table.heading("profit", text="Маржа")

            table.column("name", width=280)
            table.column("quality", width=80, anchor=tk.CENTER)
            table.column("sales", width=90, anchor=tk.CENTER)
            table.column("cost", width=110, anchor=tk.CENTER)
            table.column("profit", width=110, anchor=tk.CENTER)

            for item in result["products"]:
                table.insert(
                    "",
                    tk.END,
                    values=(
                        item["name"],
                        round(item["quality"], 3),
                        item["expected_sales"],
                        item["stock_cost"],
                        int(item["profit"]),
                    ),
                )

            table.pack(fill=tk.X)

            figure = Figure(figsize=(7.4, 3.6), dpi=100, facecolor="#edf2f7")
            axes = figure.add_subplot(111)
            names = [item["name"].split(" ", 1)[-1] for item in result["products"]]
            values = [item["quality"] for item in result["products"]]
            axes.barh(list(reversed(names)), list(reversed(values)), color="#2563eb")
            axes.set_title("Качество выбранных товаров")
            axes.set_xlim(0, 1)
            axes.grid(True, axis="x", color="#cbd5e1")
            figure.tight_layout()

            canvas = FigureCanvasTkAgg(figure, result_area)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(16, 0))

        ttk.Button(panel, text="Подобрать", command=draw_result).pack(fill=tk.X, pady=(10, 0))
        draw_result()


def check_project():
    sales = load_sales()
    products = load_products()
    quality = calculate_quality(products)
    model = train_model(sales)

    print(f"строк продаж: {len(sales)}")
    print(f"товаров: {len(products)}")
    print(f"лучший товар: {quality[0]['name']}")
    print(f"R2: {model['r2']:.3f}")


def main():
    if "--check" in sys.argv:
        check_project()
        return

    app = ShopDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
