import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import re

# --- НАСТРОЙКИ ---
DEFAULT_INPUT = "polyclinic_dataset.xlsx"
DEFAULT_OUTPUT = "polyclinic_anonymized.xlsx"


class AnonymizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Лаб №2: Обезличивание и K-anonymity")
        self.root.geometry("1100x750")

        self.df = None
        self.original_df = None
        self.target_k = 10

        # Стили
        style = ttk.Style()
        style.configure("Bold.TLabel", font=("Arial", 10, "bold"))

        # ================== 1. ЗАГРУЗКА ==================
        top_frame = tk.LabelFrame(root, text="1. Загрузка данных", padx=10, pady=5)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(top_frame, text="Входной файл:").grid(row=0, column=0, sticky="w")
        self.entry_input = tk.Entry(top_frame, width=40)
        self.entry_input.insert(0, DEFAULT_INPUT)
        self.entry_input.grid(row=0, column=1, padx=5)
        tk.Button(top_frame, text="Обзор...", command=self.browse_input).grid(row=0, column=2)

        tk.Button(top_frame, text="ЗАГРУЗИТЬ ДАТАСЕТ", command=self.load_dataset, bg="#dddddd",
                  font=("Arial", 9, "bold")).grid(row=0, column=3, padx=20)

        tk.Label(top_frame, text="Файл вывода:").grid(row=1, column=0, sticky="w")
        self.entry_output = tk.Entry(top_frame, width=40)
        self.entry_output.insert(0, DEFAULT_OUTPUT)
        self.entry_output.grid(row=1, column=1, padx=5)

        self.lbl_target_k = tk.Label(top_frame, text="Целевое K: -", fg="blue", font=("Arial", 10, "bold"))
        self.lbl_target_k.grid(row=1, column=3, padx=20)

        # ================== 2. ОСНОВНАЯ ЗОНА ==================
        center_frame = tk.Frame(root)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Левая колонка
        left_frame = tk.LabelFrame(center_frame, text="2. Выбор полей (QID)", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.qid_vars = {}
        self.potential_columns = [
            "ФИО", "Паспорт", "СНИЛС", "Симптомы", "Специальность врача",
            "Дата посещения", "Анализы", "Дата получения анализов",
            "Стоимость (руб)", "Карта оплаты"
        ]

        for col in self.potential_columns:
            var = tk.BooleanVar()
            var.set(False)
            chk = tk.Checkbutton(left_frame, text=col, variable=var, anchor="w")
            chk.pack(fill=tk.X)
            self.qid_vars[col] = var

        # Правая колонка
        right_frame = tk.Frame(center_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(right_frame, text="Лог выполнения:", font=("Arial", 9, "bold")).pack(anchor="w")
        self.log_text = tk.Text(right_frame, height=12, font=("Consolas", 10), bg="white", fg="black")
        self.log_text.pack(fill=tk.X, pady=(0, 10))

        tk.Label(right_frame, text="Предпросмотр (первые 50 строк):", font=("Arial", 9, "bold")).pack(anchor="w")

        tree_frame = tk.Frame(right_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_frame)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=vsb.set)

        # ================== 3. ДЕЙСТВИЯ (GRID LAYOUT) ==================
        bottom_frame = tk.LabelFrame(root, text="3. Действия", padx=10, pady=10)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)

        # Настройка сетки для ровных кнопок
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.columnconfigure(2, weight=1)
        bottom_frame.columnconfigure(3, weight=1)

        tk.Button(bottom_frame, text="Рассчитать текущее K", command=self.calculate_k_gui, width=22).grid(row=0,
                                                                                                          column=0,
                                                                                                          padx=5,
                                                                                                          pady=5)
        tk.Button(bottom_frame, text="Оценить полезность (KLD)", command=self.calculate_utility, width=22).grid(row=0,
                                                                                                                column=1,
                                                                                                                padx=5,
                                                                                                                pady=5)

        tk.Button(bottom_frame, text="ОБЕЗЛИЧИТЬ", command=self.run_anonymization, width=22, bg="#e1f5fe",
                  font=("Arial", 9, "bold")).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(bottom_frame, text="СОХРАНИТЬ В ФАЙЛ", command=self.save_dataset, width=22, bg="#c8e6c9",
                  font=("Arial", 9, "bold")).grid(row=0, column=3, padx=5, pady=5)

    # --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def browse_input(self):
        f = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if f:
            self.entry_input.delete(0, tk.END)
            self.entry_input.insert(0, f)

    def load_dataset(self):
        path = self.entry_input.get()
        if not os.path.exists(path):
            messagebox.showerror("Ошибка", "Файл не найден!")
            return
        try:
            self.df = pd.read_excel(path)
            self.original_df = self.df.copy()
            rows = len(self.df)

            if rows < 51000:
                self.target_k = 10
            elif rows < 105000:
                self.target_k = 7
            else:
                self.target_k = 5

            self.lbl_target_k.config(text=f"Строк: {rows} | Целевое K >= {self.target_k}")
            self.clear_log()
            self.log(f"Файл загружен. Строк: {rows}.")
            self.log(f"Требуемое K-anonymity: {self.target_k}")
            self.update_preview()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть файл:\n{e}")

    def update_preview(self):
        if self.df is None: return
        self.tree.delete(*self.tree.get_children())
        cols = list(self.df.columns)
        self.tree["columns"] = cols
        self.tree["show"] = "headings"
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=100)
        for i, row in self.df.head(50).iterrows():
            self.tree.insert("", "end", values=list(row))

    def get_selected_qids(self):
        return [col for col, var in self.qid_vars.items() if var.get()]

    # --- РАСЧЕТ К ---
    def calculate_k_gui(self):
        if self.df is None: return
        qids = self.get_selected_qids()
        if not qids:
            messagebox.showwarning("Внимание", "Выберите хотя бы одну колонку!")
            return

        self.log("-" * 30)
        self.log(f"Расчет K по полям: {qids}")

        groups = self.df.groupby(qids).size().reset_index(name='count')
        min_k = groups['count'].min()
        self.log(f"Min K = {min_k}")

        bad_groups = groups[groups['count'] < self.target_k].sort_values(by='count')
        if len(bad_groups) > 0:
            self.log(f"Групп с K < {self.target_k}: {len(bad_groups)}")
            total = len(self.df)
            for i, row in bad_groups.head(3).iterrows():
                vals = [str(row[c]) for c in qids]
                cnt = row['count']
                pct = (cnt / total) * 100
                self.log(f"  {vals} -> {cnt} шт ({pct:.4f}%)")
        else:
            self.log("K-anonymity соблюдено.")

    # --- ЛОГИКА ОБЕЗЛИЧИВАНИЯ ---
    def run_anonymization(self):
        if self.original_df is None:
            messagebox.showerror("Ошибка", "Сначала загрузите датасет!")
            return

        self.clear_log()
        self.log("=== ЗАПУСК ОБЕЗЛИЧИВАНИЯ ===")
        self.df = self.original_df.copy()
        selected_cols = self.get_selected_qids()

        # Логика трансформаций
        transformations = []

        # 1. Прямые идентификаторы (Маскеризация)
        if "ФИО" in selected_cols and "ФИО" in self.df.columns:
            # Иванов Иван Иванович -> Иванов И.И.
            self.df["ФИО"] = self.df["ФИО"].apply(
                lambda x: f"{x.split()[0]} {x.split()[1][0]}.{x.split()[2][0]}." if isinstance(x, str) and len(
                    x.split()) >= 3 else x
            )
            transformations.append("ФИО: Сокращение до инициалов")

        if "Паспорт" in selected_cols and "Паспорт" in self.df.columns:
            # 1234 567890 -> 12** ****** (Оставляем регион примерно, но скрываем личность)
            self.df["Паспорт"] = self.df["Паспорт"].apply(lambda x: str(x)[:2] + "** ******")
            transformations.append("Паспорт: Маскеризация (12** ...)")

        if "СНИЛС" in selected_cols and "СНИЛС" in self.df.columns:
            # XXX-XXX-XX 99 (Оставляем только контрольное число в конце)
            self.df["СНИЛС"] = self.df["СНИЛС"].apply(lambda x: "XXX-XXX-XXX " + str(x)[-2:])
            transformations.append("СНИЛС: Маскеризация (кроме последних 2 цифр)")

        if "Карта оплаты" in selected_cols and "Карта оплаты" in self.df.columns:
            # 1234 ... (Сбербанк, Visa) -> **** (Сбербанк, Visa)
            def mask_card_keep_bank(val):
                s = str(val)
                # Ищем скобки с банком
                match = re.search(r'(\(.*\))', s)
                bank_info = match.group(1) if match else ""
                return f"**** {bank_info}"

            self.df["Карта оплаты"] = self.df["Карта оплаты"].apply(mask_card_keep_bank)
            transformations.append("Карта: Скрытие номера, сохранение Банка")

        # 2. Категориальные (Обобщение / Сортировка)
        if "Специальность врача" in selected_cols and "Специальность врача" in self.df.columns:
            # Терапевт-05 -> Терапевт
            self.df["Специальность врача"] = self.df["Специальность врача"].apply(lambda x: str(x).split('-')[0])
            transformations.append("Врач: Удаление ID (Обобщение)")

        def clean_and_sort_list(val):
            # "Кашель, Насморк" -> "Кашель, Насморк" (сортировка по алфавиту)
            if not isinstance(val, str): return val
            items = [item.strip() for item in val.split(',')]
            items.sort()
            return ", ".join(items)

        if "Симптомы" in selected_cols and "Симптомы" in self.df.columns:
            self.df["Симптомы"] = self.df["Симптомы"].apply(clean_and_sort_list)
            transformations.append("Симптомы: Сортировка списка")

        if "Анализы" in selected_cols and "Анализы" in self.df.columns:
            self.df["Анализы"] = self.df["Анализы"].apply(clean_and_sort_list)
            transformations.append("Анализы: Сортировка списка")

        # 3. Даты и Числа (Обобщение)
        date_cols = ["Дата посещения", "Дата получения анализов"]
        for d_col in date_cols:
            if d_col in selected_cols and d_col in self.df.columns:
                try:
                    self.df[d_col] = pd.to_datetime(self.df[d_col]).dt.to_period('M').astype(str)
                except:
                    pass
        if any(c in selected_cols for c in date_cols):
            transformations.append("Даты: Обобщение до месяца")

        if "Стоимость (руб)" in selected_cols and "Стоимость (руб)" in self.df.columns:
            self.df["Стоимость (руб)"] = pd.to_numeric(self.df["Стоимость (руб)"], errors='coerce')
            bins = [0, 1000, 3000, 5000, 10000, 100000]
            labels = ["0-1000", "1000-3000", "3000-5000", "5000-10000", "10000+"]
            self.df["Стоимость (руб)"] = pd.cut(self.df["Стоимость (руб)"], bins=bins, labels=labels).astype(str)
            transformations.append("Стоимость: Интервалы")

        # Вывод примененных методов
        for t in transformations:
            self.log(f"> {t}")

        # 4. ПОДАВЛЕНИЕ (Suppression)
        # Группируем ПО ВСЕМ выбранным полям, включая прямые ID (как ты и просил)
        if selected_cols:
            groups = self.df.groupby(selected_cols).size().reset_index(name='count')
            good_groups = groups[groups['count'] >= self.target_k][selected_cols]

            old_len = len(self.df)
            if old_len > 0:
                self.df = self.df.merge(good_groups, on=selected_cols, how='inner')
                new_len = len(self.df)
                deleted = old_len - new_len
                pct_del = (deleted / old_len) * 100

                self.log(f"\nПодавление (удаление групп < {self.target_k}):")
                self.log(f"  Удалено: {deleted} строк ({pct_del:.2f}%)")
                self.log(f"  Осталось: {new_len} строк")
            else:
                self.log("Ошибка: таблица пуста.")
        else:
            self.log("Нет выбранных полей для группировки.")

        self.update_preview()

    def save_dataset(self):
        if self.df is None:
            messagebox.showwarning("Внимание", "Нет данных.")
            return
        out_path = self.entry_output.get()
        try:
            self.df.to_excel(out_path, index=False)
            self.log(f"\nСохранено в: {out_path}")
            messagebox.showinfo("Успех", f"Сохранено в:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    def calculate_utility(self):
        col = "Стоимость (руб)"
        if self.original_df is None or col not in self.original_df.columns:
            return

        self.log("-" * 30)
        self.log("Оценка полезности (KLD) для Стоимости:")
        bins = [0, 1000, 3000, 5000, 10000, 100000]
        labels = ["0-1000", "1000-3000", "3000-5000", "5000-10000", "10000+"]

        try:
            numeric_data = pd.to_numeric(self.original_df[col], errors='coerce').dropna()
            orig_cats = pd.cut(numeric_data, bins=bins, labels=labels)
            P = orig_cats.value_counts(normalize=True).sort_index() + 1e-10

            current_col = self.df[col]
            # Если еще числа - превращаем в категории, если уже строки - считаем так
            if pd.api.types.is_numeric_dtype(current_col):
                Q_cats = pd.cut(current_col, bins=bins, labels=labels)
                Q = Q_cats.value_counts(normalize=True).sort_index()
            else:
                Q = current_col.value_counts(normalize=True).sort_index()

            all_idx = P.index.union(Q.index)
            P = P.reindex(all_idx, fill_value=1e-10)
            Q = Q.reindex(all_idx, fill_value=1e-10)

            kl_div = stats.entropy(P, Q)
            # Fix для -0.0000
            kl_div = max(0.0, kl_div)

            self.log(f"  KLD = {kl_div:.5f}")

        except Exception as e:
            self.log(f"Ошибка расчета: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AnonymizerApp(root)
    root.mainloop()