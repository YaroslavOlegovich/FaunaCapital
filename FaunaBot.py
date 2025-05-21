import tkinter as tk
from tkinter import messagebox, ttk
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

def validate_and_format_date(date_str):
    try:
        # Попробуем сначала как ISO (YYYY-MM-DD)
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        try:
            # Попробуем как DD.MM.YYYY
            return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")
        except ValueError:
            return None

def fetch_and_analyze():
    ticker = ticker_entry.get().strip().upper()
    start_date_raw = start_entry.get().strip()
    end_date_raw = end_entry.get().strip()
    model_type = model_choice.get()

    start_date = validate_and_format_date(start_date_raw)
    end_date = validate_and_format_date(end_date_raw)

    if not start_date or not end_date:
        messagebox.showerror("Ошибка", "Неверный формат даты. Используйте YYYY-MM-DD или DD.MM.YYYY.")
        return

    # Проверка, что дата начала не после даты окончания
    if datetime.strptime(start_date, "%Y-%m-%d") >= datetime.strptime(end_date, "%Y-%m-%d"):
        messagebox.showerror("Ошибка", "Дата начала должна быть раньше даты окончания.")
        return

    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError("Нет данных для выбранного тикера или диапазона дат.")
        
        required_data_for_indicators = 34 
        if len(df) < required_data_for_indicators: 
            messagebox.showerror("Недостаточно данных", f"Требуется минимум {required_data_for_indicators} дней данных для расчета индикаторов. Сейчас доступно только {len(df)} дней. Увеличьте диапазон дат.")
            return

    except Exception as e:
        messagebox.showerror("Ошибка загрузки данных", f"Ошибка: {e}\nПроверьте тикер и даты. Возможно, дата начала позже даты окончания или тикер недействителен.")
        return

    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int) 

    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # --- Ручной расчет RSI ---
    delta = df['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()

    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    df['RSI'] = 100 - (100 / (1 + rs))
    # --- Конец ручного расчета RSI ---

    # --- Ручной расчет MACD ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Diff'] = df['MACD_Line'] - df['Signal_Line']
    # --- Конец ручного расчета MACD ---

    df.dropna(inplace=True)

    min_data_for_model = 50
    if len(df) < min_data_for_model: 
        messagebox.showerror("Ошибка данных", f"Недостаточно данных после расчета всех индикаторов. Требуется минимум {min_data_for_model} дней данных для обучения модели. Попробуйте увеличить диапазон дат.")
        return

    features = ['Close', 'MA5', 'MA10', 'MA20', 'Volatility', 'RSI', 'MACD_Diff'] 
    
    missing_features = [
        f for f in features 
        if f not in df.columns or (f in df.columns and df[f].squeeze().isnull().all())
    ]
    if missing_features:
        messagebox.showerror("Ошибка признаков", f"Не удалось создать следующие признаки или они содержат только NaN: {', '.join(missing_features)}. Возможно, недостаточно данных для их расчета.")
        return

    X = df[features]
    y = df['Target']

    test_size_val = 0.2
    if len(X) < 2:
        messagebox.showerror("Ошибка данных", "Недостаточно данных для обучения модели (минимум 2 точки). Увеличьте диапазон дат.")
        return
    elif len(X) < 5: 
        test_size_val = 1 / len(X) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, shuffle=False)

    if model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        messagebox.showerror("Ошибка", "Неизвестный тип модели.")
        return

    if len(X_train) == 0:
        messagebox.showerror("Ошибка обучения", "Недостаточно данных для обучения модели. Попробуйте увеличить диапазон дат.")
        return

    model.fit(X_train, y_train)

    accuracy = 0
    if len(X_test) > 0:
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

    next_day_prediction = None
    if not X.empty:
        latest_data_point = X.iloc[[-1]] 
        next_day_prediction = model.predict(latest_data_point)[0]

    prognosis_text = "Не удалось спрогнозировать"
    if next_day_prediction is not None:
        prognosis_text = "Рост  📈 " if next_day_prediction == 1 else "Падение  📉 "

    messagebox.showinfo("Результат анализа",
                        f"Точность модели ({model_type}): {accuracy * 100:.2f}%\n"
                        f"Прогноз на следующий торговый день: {prognosis_text}")

    # Построение графиков
    # Здесь цвета графиков лучше оставить такими, чтобы они были различимы
    # Если ты хочешь их тоже изменить на черно-бело-фиолетовые, дай знать!
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Анализ {ticker}", fontsize=16, color='black') # Можно изменить цвет заголовка графика

    # График цены и скользящих средних
    ax1.plot(df.index, df['Close'], label='Цена закрытия', color='blue')
    ax1.plot(df.index, df['MA5'], label='MA 5', linestyle='--', color='orange')
    ax1.plot(df.index, df['MA10'], label='MA 10', linestyle='--', color='red')
    ax1.plot(df.index, df['MA20'], label='MA 20', linestyle='--', color='purple')
    ax1.set_ylabel("Цена")
    ax1.legend()
    ax1.grid(True)

    # График RSI
    ax2.plot(df.index, df['RSI'], label='RSI (14)', color='green')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7, label='Перекупленность (70)')
    ax2.axhline(30, color='blue', linestyle='--', alpha=0.7, label='Перепроданность (30)')
    ax2.set_ylabel("RSI")
    ax2.legend()
    ax2.grid(True)

    # График MACD
    ax3.plot(df.index, df['MACD_Line'], label='MACD Line', color='blue')
    ax3.plot(df.index, df['Signal_Line'], label='Signal Line', color='red', linestyle='--')
    ax3.bar(df.index, df['MACD_Diff'], label='MACD Histogram', color='gray', alpha=0.5)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax3.set_ylabel("MACD")
    ax3.set_xlabel("Дата")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
    plt.show()


# Интерфейс
root = tk.Tk()
root.title("FaunaBOT")
root.geometry("450x380") 

# --- Цвета для стиля ---
bg_color = "#1a1a1a"       # Очень темный серый / почти черный
fg_color = "#e0e0e0"       # Светло-серый / почти белый
button_bg_color = "#8a2be2" # Фиолетовый (BlueViolet)
button_fg_color = "#ffffff" # Белый
entry_bg_color = "#333333" # Темно-серый для полей ввода
entry_fg_color = "#ffffff" # Белый текст в полях ввода
select_bg_color = "#5a2be2" # Немного другой фиолетовый для выделения/фона комбобокса

root.config(bg=bg_color)

tk.Label(root, text="Тикер (например, BTC-USD, AAPL):", bg=bg_color, fg=fg_color).pack(pady=5)
ticker_entry = tk.Entry(root, width=30, bg=entry_bg_color, fg=entry_fg_color, insertbackground=fg_color) # insertbackground меняет цвет курсора
ticker_entry.pack(pady=2)
ticker_entry.insert(0, "BTC-USD")

tk.Label(root, text="Дата начала (YYYY-MM-DD):", bg=bg_color, fg=fg_color).pack(pady=5)
start_entry = tk.Entry(root, width=30, bg=entry_bg_color, fg=entry_fg_color, insertbackground=fg_color)
start_entry.pack(pady=2)
start_entry.insert(0, (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")) 

tk.Label(root, text="Дата окончания (YYYY-MM-DD):", bg=bg_color, fg=fg_color).pack(pady=5)
end_entry = tk.Entry(root, width=30, bg=entry_bg_color, fg=entry_fg_color, insertbackground=fg_color)
end_entry.pack(pady=2)
end_entry.insert(0, datetime.now().strftime("%Y-%m-%d")) 

tk.Label(root, text="Выбор модели:", bg=bg_color, fg=fg_color).pack(pady=5)

# Создаем стиль для Combobox
style = ttk.Style()
# Настраиваем тему, если нужно (например, 'clam', 'alt', 'default', 'classic')
style.theme_use('clam') 
# Настраиваем фон и цвет текста для Combobox
style.configure("TCombobox", 
                fieldbackground=entry_bg_color, 
                background=bg_color, # Фон выпадающего списка
                foreground=entry_fg_color,
                selectbackground=select_bg_color, # Цвет фона при выделении
                selectforeground=button_fg_color,
                arrowcolor=fg_color) # Цвет стрелки
# Настраиваем цвет текста в выпадающем списке при его открытии
style.map('TCombobox',
          fieldbackground=[('readonly', entry_bg_color)],
          background=[('readonly', bg_color)],
          foreground=[('readonly', entry_fg_color)])

model_choice = ttk.Combobox(root, values=["Random Forest", "Logistic Regression"], width=27, style="TCombobox")
model_choice.set("Random Forest") 
model_choice.pack(pady=2)

tk.Button(root, text="Запустить анализ и прогноз", command=fetch_and_analyze, 
          bg=button_bg_color, fg=button_fg_color, padx=10, pady=5, 
          activebackground=select_bg_color, activeforeground=button_fg_color, # Цвета при нажатии
          relief="flat").pack(pady=20) # Убираем стандартную рамку для более плоского вида

root.mainloop()