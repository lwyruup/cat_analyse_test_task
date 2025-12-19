import pandas as pd
import requests
import json
import re
import time
from datetime import datetime
from dotenv import load_dotenv
import os

#API
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL_SYNC", "llama-3.1-8b-instant")
SLEEP_BETWEEN_CALLS = 2.5  # пауза между запросами
MAX_RETRIES = 3  # попытки при ошибках



def classify_with_improved_prompt(company_info: str) -> dict:
    """Улучшенный LLM классификатор с примерами"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Ты детектируешь использование CAT-систем (Computer-Assisted Translation) в компаниях.

CAT-системы: Trados, MemoQ, Smartcat, Memsource, XTM, Phrase TMS, Wordfast, Déjà Vu, Transit, OmegaT.

ПРАВИЛА:
1. has_cat = true если:
   - Упоминается конкретная CAT-система
   - Компания - переводческое/локализационное агентство (это отраслевой стандарт)
   - Есть слова "CAT-инструменты", "translation memory", "TM", "TMS"
2. has_cat = false если:
   - Нет признаков переводческой деятельности
   - Не упоминаются CAT-технологии
   - Индивидуальный переводчик (но может использовать OmegaT)
3. confidence должен отражать уверенность:
   - 0.9-1.0: явное упоминание CAT
   - 0.6-0.8: переводческая компания без явного упоминания
   - 0.3-0.5: возможное использование
   - 0.0-0.2: нет признаков

ПРИМЕРЫ:
1. "SmartCAT Technologies" → has_cat=true, confidence=1.0 (они делают CAT)
2. "ЛингваПро Переводы" → has_cat=true, confidence=0.8 (переводческое агентство)
3. "Яндекс.Такси" → has_cat=false, confidence=0.1 (не переводческая)
4. "Индивидуальный переводчик" → has_cat=false, confidence=0.3 (может быть OmegaT)

Компания:
{company_info}

Ответь ТОЛЬКО JSON:
{{
  "has_cat": true/false,
  "confidence": 0.0-1.0,
  "evidence": "краткое обоснование",
  "product": "название CAT или пусто"
}}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Ты эксперт по переводческим технологиям."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=40)

            # Обработка rate limit
            if r.status_code == 429:
                retry_after = int(r.headers.get('Retry-After', 30))
                print(f"⏳ Rate limit, жду {retry_after} секунд...")
                time.sleep(retry_after)
                continue

            r.raise_for_status()
            data = r.json()

            text = data["choices"][0]["message"]["content"]

            # Ищем JSON в ответе
            match = re.search(r'\{.*\}', text, re.DOTALL)

            if match:
                result = json.loads(match.group())

                # Валидация полей
                if "has_cat" not in result:
                    result["has_cat"] = False
                if "confidence" not in result:
                    result["confidence"] = 0.1 if not result.get("has_cat", False) else 0.5
                if "evidence" not in result:
                    result["evidence"] = "No evidence provided"
                if "product" not in result:
                    result["product"] = ""

                if not result["has_cat"]:
                    result["confidence"] = min(result["confidence"], 0.3)

                return result
            else:
                return {
                    "has_cat": False,
                    "confidence": 0.1,
                    "evidence": f"Ошибка парсинга: {text[:100]}",
                    "product": ""
                }

        except requests.exceptions.RequestException as e:
            print(f"Попытка {attempt}/{MAX_RETRIES} — ошибка сети: {e}")
            time.sleep(5)
        except json.JSONDecodeError as e:
            print(f"Попытка {attempt}/{MAX_RETRIES} — ошибка JSON: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"Попытка {attempt}/{MAX_RETRIES} — ошибка: {e}")
            time.sleep(5)

    # Если все попытки исчерпаны
    return {
        "has_cat": False,
        "confidence": 0.1,
        "evidence": "Все попытки запроса исчерпаны",
        "product": ""
    }


def main():
    print("🚀 LLM-КЛАССИФИКАЦИЯ CAT-СИСТЕМ")
    print("=" * 50)

    # Загружаем данные
    try:
        df = pd.read_csv("/Users/samvelsimavoryan/PycharmProjects/cat_test_task/src/data/candidate_companies_simple.csv")
        print(f"Загружено компаний: {len(df)}")
    except FileNotFoundError:
        print("Файл не найден!")
        return

    # Фильтруем компании для анализа (как в оригинальном коде)
    companies = df[
        df['nace_code'].astype(str).str.startswith(('62', '63.11')) |
        df['name'].str.contains('перевод|translation|лингв|локализ|localization|транслей',
                                case=False, na=False)
        ].copy()

    # Если слишком много, берем первые 150
    if len(companies) > 150:
        companies = companies.head(2000)
        print(f"Ограничение: берем первые 150 компаний")

    print(f"Компаний для анализа: {len(companies)}")

    results = []
    errors = 0

    # Обрабатываем каждую компанию
    for i, row in enumerate(companies.itertuples(), 1):
        print(f"{i:3d}. {row.name[:45]:45s}", end="", flush=True)

        company_info = f"""
Название: {row.name}
Выручка: {getattr(row, 'revenue', 'не указано')} ₽
ОКВЭД: {getattr(row, 'nace_code', 'не указан')}
Описание: {getattr(row, 'description', 'нет описания')[:300]}
"""

        # Классификация
        result = classify_with_improved_prompt(company_info)

        # Если CAT-компания, сохраняем
        if result.get("has_cat"):
            results.append({
                "inn": str(getattr(row, 'inn', '')),
                "name": row.name,
                "revenue": int(getattr(row, 'revenue', 0)),
                "cat_product": result.get("product", ""),
                "confidence": result.get("confidence", 0.0),
                "evidence": result.get("evidence", ""),
                "source": "Groq + Llama-3 (improved)",
                "date": datetime.now().strftime("%Y-%m-%d")
            })

            status = f"CAT (conf: {result.get('confidence', 0.0):.2f})"
        else:
            status = f"нет (conf: {result.get('confidence', 0.0):.2f})"

        print(f" | {status}")

        # Пауза между запросами (кроме последнего)
        if i < len(companies):
            time.sleep(SLEEP_BETWEEN_CALLS)

    # Сохраняем результаты
    if results:
        output_df = pd.DataFrame(results)
        output_df.to_csv("companies_cat_llm.csv", index=False, encoding="utf-8-sig")

        print(f"\n{'=' * 50}")
        print(f"Найдено CAT-компаний: {len(results)}")
        print(f"Файл: companies_cat_llm.csv")

    else:
        print("\nCAT-компании не найдены")

    print("\n" + "=" * 50)
    print("Классификация завершена!")


if __name__ == "__main__":
    main()