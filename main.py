#!/usr/bin/env python3
"""Luxhomme review assistant: CSV -> topic/sentiment analysis -> SQLite -> Telegram."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
DB_PATH = ROOT / "review_history.sqlite3"
TOPICS = [
    "Качество и надёжность", "Доставка и упаковка", "Удобство использования",
    "Производительность", "Шум", "Автономность", "Очистка и уход",
    "Поддержка и возврат", "Внешний вид и материалы", "Цена и ценность", "Другое"
]
PROMPT_VERSION = "1.0"

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"date", "sku", "rating", "text"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV должен содержать колонки: date, sku, rating, text")
        for line_no, row in enumerate(reader, start=2):
            try:
                d = date.fromisoformat(row["date"].strip())
                rating = int(row["rating"])
                sku = row["sku"].strip()
                review_text = row["text"].strip()
                if not sku or not review_text:
                    raise ValueError("пустой SKU или текст")
                if not 1 <= rating <= 5:
                    raise ValueError("рейтинг должен быть от 1 до 5")
                rows.append({"date": d.isoformat(), "sku": sku, "rating": rating, "text": review_text})
            except Exception as exc:
                raise ValueError(f"Ошибка в строке {line_no}: {exc}") from exc
    if not rows:
        raise ValueError("В CSV нет отзывов")
    return rows

def heuristic_label(review: dict) -> dict:
    """Transparent fallback. Treat as baseline, not as an LLM result."""
    t = review["text"].lower()
    rules = [
        ("Доставка и упаковка", ["достав", "короб", "упаков", "помят"]),
        ("Шум", ["шум", "громк"]),
        ("Автономность", ["батар", "заряд", "аккумулятор"]),
        ("Поддержка и возврат", ["поддерж", "сервис", "возврат", "не отвечает", "молчит"]),
        ("Удобство использования", ["удоб", "управлен", "инструкц", "приложен", "связь"]),
        ("Производительность", ["быстро", "мощност", "нагрев", "готовит", "убира", "застревает", "проходит"]),
        ("Очистка и уход", ["моется", "мыть", "отмы", "чист", "очист"]),
        ("Внешний вид и материалы", ["пластик", "царап", "корпус", "шнур", "крышк"]),
        ("Качество и надёжность", ["слом", "перестал", "качеств", "сборк", "работает"]),
        ("Цена и ценность", ["цен", "дорог", "дешев"]),
    ]
    topic = next((name for name, terms in rules if any(term in t for term in terms)), "Другое")
    positive = any(x in t for x in ["отлич", "вкусн", "аромат", "хорош", "удоб", "быстро", "качеств", "понрав", "легко", "насыщен"])
    negative = any(x in t for x in ["но ", "плохо", "шум", "застре", "слом", "царап", "задерж", "опозда", "неудоб", "маленьк", "коротк", "пахнет", "перестал", "сложно", "не отвечает", "могла бы", "мог бы", "туго", "теряет"])
    if review["rating"] <= 2 or (negative and not positive):
        sentiment = "negative"
    elif review["rating"] >= 4 and not negative:
        sentiment = "positive"
    else:
        sentiment = "mixed"
    return {"topic": topic, "sentiment": sentiment, "reason": "Эвристическая разметка; проверить вручную"}

def gemini_labels(reviews: list[dict]) -> list[dict]:
    """Classify reviews through the Gemini Interactions REST API with bounded retries."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Не задан GEMINI_API_KEY")

    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    endpoint = "https://generativelanguage.googleapis.com/v1beta/interactions"
    prompt = (
        "Ты аналитик клиентских отзывов на русском языке. Для каждого отзыва выбери ровно одну основную тему из списка: "
        + "; ".join(TOPICS)
        + ". Тональность должна быть одной из positive, negative, mixed. Оценивай содержание, а не только рейтинг. "
        "Если в отзыве есть и похвала, и жалоба, выбирай mixed. Не придумывай факты. "
        "Верни только валидный JSON-массив объектов с полями id (целое число из входа), topic, sentiment, reason "
        "(краткое объяснение до 15 слов), без Markdown и без дополнительных пояснений.\nОтзывы:\n"
        + json.dumps([{"id": i, **r} for i, r in enumerate(reviews)], ensure_ascii=False)
    )
    payload = {"model": model, "input": prompt}
    req_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    data = None
    for attempt in range(4):
        req = urllib.request.Request(
            endpoint,
            data=req_data,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            if exc.code in {429, 500, 502, 503, 504} and attempt < 3:
                import time
                delay = 2 ** (attempt + 1)
                print(f"Gemini временно вернул HTTP {exc.code}; повтор через {delay} сек. ({attempt + 1}/3)", file=sys.stderr)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini Interactions API вернул HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, ConnectionError) as exc:
            if attempt < 3:
                import time
                delay = 2 ** (attempt + 1)
                print(f"Тайм-аут или ошибка соединения Gemini; повтор через {delay} сек. ({attempt + 1}/3)", file=sys.stderr)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini не ответил вовремя: {exc}") from exc
        except urllib.error.URLError as exc:
            if attempt < 3:
                import time
                delay = 2 ** (attempt + 1)
                print(f"Ошибка сети Gemini; повтор через {delay} сек. ({attempt + 1}/3)", file=sys.stderr)
                time.sleep(delay)
                continue
            raise RuntimeError(f"Не удалось подключиться к Gemini API: {exc}") from exc

    try:
        # Interactions API returns model output in steps[].content[].text.
        text_parts = []
        for step in data.get("steps", []):
            if step.get("type") == "model_output":
                for block in step.get("content", []):
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        text_parts.append(block["text"])
        raw = "\n".join(text_parts).strip()
        if not raw:
            raise ValueError("в ответе нет текстового результата модели")
        # Be tolerant if the model wraps JSON in a Markdown fence despite the prompt.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("ожидался JSON-массив")
        by_id = {int(item["id"]): item for item in parsed}
        if set(by_id) != set(range(len(reviews))):
            raise ValueError("набор id в ответе не совпадает с набором входных отзывов")
        result = []
        for i in range(len(reviews)):
            item = by_id[i]
            if item.get("topic") not in TOPICS or item.get("sentiment") not in {"positive", "negative", "mixed"}:
                raise ValueError(f"недопустимая разметка у отзыва {i}")
            result.append({
                "topic": item["topic"],
                "sentiment": item["sentiment"],
                "reason": str(item.get("reason", ""))[:240],
            })
        return result
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ответ Gemini не соответствует ожидаемой JSON-схеме: {exc}") from exc

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_date TEXT NOT NULL, sku TEXT NOT NULL, rating INTEGER NOT NULL,
            review_text TEXT NOT NULL, topic TEXT NOT NULL, sentiment TEXT NOT NULL,
            reason TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL,
            PRIMARY KEY (review_date, sku, rating, review_text)
        )
    """)
    conn.commit()

def save_reviews(rows: list[dict], labels: list[dict], model: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)
        for r, label in zip(rows, labels):
            conn.execute("""INSERT OR REPLACE INTO reviews
                (review_date, sku, rating, review_text, topic, sentiment, reason, model, prompt_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (r["date"], r["sku"], r["rating"], r["text"], label["topic"], label["sentiment"], label["reason"], model, PROMPT_VERSION))
        conn.commit()

def get_history(start: date, end: date) -> list[dict]:
    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""SELECT review_date AS date, sku, rating, review_text AS text, topic, sentiment, reason
            FROM reviews WHERE review_date >= ? AND review_date <= ? ORDER BY review_date""",
            (start.isoformat(), end.isoformat())).fetchall()
        return [dict(r) for r in rows]

def build_report(rows: list[dict], labels: list[dict], prior_rows: list[dict] | None = None) -> str:
    enriched = [{**r, **lab} for r, lab in zip(rows, labels)]
    topic_counts = Counter(r["topic"] for r in enriched)
    sentiments = Counter(r["sentiment"] for r in enriched)
    avg = sum(r["rating"] for r in enriched) / len(enriched)
    lines = [
        "🛍 <b>Luxhomme — сводка отзывов</b>",
        f"Период: {min(r['date'] for r in rows)} — {max(r['date'] for r in rows)}",
        f"Всего отзывов: <b>{len(rows)}</b> · Средний рейтинг: <b>{avg:.2f}/5</b>",
        f"Тональность: 🟢 {sentiments['positive']} положительных · 🔴 {sentiments['negative']} отрицательных · 🟡 {sentiments['mixed']} смешанных",
        "", "<b>Темы</b>"
    ]
    for topic, count in topic_counts.most_common():
        lines.append(f"• {html_escape(topic)} — {count} ({count / len(rows):.0%})")
    negative = [r for r in enriched if r["sentiment"] == "negative"]
    lines.extend(["", "<b>Жалобы с примерами</b>"])
    if not negative:
        lines.append("Отрицательных отзывов не найдено.")
    for r in sorted(negative, key=lambda x: x["rating"])[:4]:
        lines.append(f"• [{html_escape(r['sku'])}, {r['rating']}/5] «{html_escape(r['text'])}» — {html_escape(r['topic'])}")
    if prior_rows:
        prev_counts = Counter(r["topic"] for r in prior_rows)
        lines.extend(["", "<b>Изменения к предыдущей неделе</b>"])
        all_topics = set(topic_counts) | set(prev_counts)
        diffs = sorted(((topic, topic_counts[topic] - prev_counts[topic]) for topic in all_topics), key=lambda x: abs(x[1]), reverse=True)
        for topic, delta in diffs[:4]:
            sign = "+" if delta > 0 else ""
            lines.append(f"• {html_escape(topic)}: {sign}{delta} отзывов ({prev_counts[topic]} → {topic_counts[topic]})")
    lines.extend(["", "<i>AI-разметка может ошибаться; перед бизнес-решениями проверьте примеры вручную.</i>"])
    return "\n".join(lines)

def html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Для отправки нужны TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    # Telegram text messages are limited; split conservatively.
    chunks = [message[i:i+3800] for i in range(0, len(message), 3800)]
    for chunk in chunks:
        payload = urllib.parse.urlencode({"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"})
        req = urllib.request.Request(url, data=payload.encode(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())
            if not result.get("ok"):
                raise RuntimeError(f"Telegram API: {result}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Telegram API HTTP {exc.code}: {detail}") from exc

def main() -> int:
    parser = argparse.ArgumentParser(description="Разбор CSV отзывов и отправка сводки в Telegram.")
    parser.add_argument("--input", default="data/reviews.csv", help="Путь к CSV")
    parser.add_argument("--previous", default="data/previous_week.csv", help="CSV предыдущей недели для сравнения")
    parser.add_argument("--send-telegram", action="store_true", help="Отправить сводку в Telegram")
    parser.add_argument("--baseline", action="store_true", help="Использовать локальную эвристическую разметку вместо Gemini")
    args = parser.parse_args()
    try:
        rows = read_csv(Path(args.input))
        if args.baseline:
            labels = [heuristic_label(r) for r in rows]
            model_name = "heuristic-baseline"
        else:
            labels = gemini_labels(rows)
            model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        save_reviews(rows, labels, model_name)
        prior_rows = []
        if args.previous and Path(args.previous).exists():
            prev = read_csv(Path(args.previous))
            if args.baseline:
                prev_labels = [heuristic_label(r) for r in prev]
            else:
                prev_labels = gemini_labels(prev)
            prior_rows = [{**r, **lab} for r, lab in zip(prev, prev_labels)]
        report = build_report(rows, labels, prior_rows)
        # Console output is plain text; strip simple Telegram HTML tags.
        print(re.sub(r"<[^>]+>", "", report).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
        if args.send_telegram:
            send_telegram(report)
            print("\nСводка отправлена в Telegram.")
        else:
            print("\nРежим предпросмотра: Telegram не вызывался. Добавьте --send-telegram для отправки.")
        return 0
    except Exception as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
