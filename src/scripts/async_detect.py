import pandas as pd
import asyncio
import aiohttp
import json
import re
import time
from datetime import datetime
import os
import ssl
import certifi
from typing import Dict, List, Optional
from dotenv import load_dotenv


def create_ssl_context():
    """Создание SSL контекста с сертификатами"""
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        return ssl_context
    except Exception as e:
        print(f"Ошибка создания SSL контекста: {e}")
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        return ssl_context


# Загрузка конфигурации
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL_SYNC", "llama-3.1-8b-instant")
MAX_CONCURRENT_REQUESTS = 2
REQUEST_TIMEOUT = 30
BATCH_SIZE = 300
SAVE_EVERY = 100  # Сохранять каждые 100 компаний


class AsyncCatDetector:
    def __init__(self, api_key: str, max_concurrent: int = 2):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.results = []
        self.processed_count = 0
        self.cat_count = 0
        self.errors = 0
        self.start_time = None
        self.rate_limit_wait = 0  # Таймер для rate limit

    async def __aenter__(self):
        ssl_context = create_ssl_context()
        connector = aiohttp.TCPConnector(
            ssl=ssl_context,
            limit=0,
            use_dns_cache=True
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            trust_env=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def handle_rate_limit(self, response_headers: dict, company_name: str):
        """Обработка rate limit с экспоненциальной задержкой"""
        retry_after = int(response_headers.get('Retry-After', 30))

        # Увеличиваем задержку с каждой ошибкой
        self.rate_limit_wait = max(retry_after, self.rate_limit_wait * 1.5)
        self.rate_limit_wait = min(self.rate_limit_wait, 300)  # Макс 5 минут

        print(f"\nRate limit для {company_name[:30]}. Жду {self.rate_limit_wait:.1f} секунд...")
        await asyncio.sleep(self.rate_limit_wait)

        return self.rate_limit_wait

    async def classify_company(self, company_data: Dict, max_retries: int = 3) -> Dict:
        """Классификация компании с ограниченным количеством попыток"""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        company_info = f"Название: {company_data['name']}"

        prompt = f"""
Ты определяешь, использует ли компания ПРОФЕССИОНАЛЬНЫЕ CAT-системы.

CAT-системы (профессиональные): 
- Trados, MemoQ, Smartcat, Memsource, XTM, Phrase TMS
- Wordfast, Déjà Vu, Transit

НЕ СЧИТАТЬ CAT:
- Google Translate, Яндекс.Переводчик, DeepL - это машинный перевод
- OmegaT - бесплатный, для фрилансеров
- Онлайн-словари, мобильные приложения

ПРАВИЛА:
1. has_cat = true  если:
   - Указана профессиональная CAT-система из списка выше
   - Есть фразы: "работаем в Trados", "используем MemoQ", "внедрили Smartcat"

2.  has_cat = false если:
   - Нет явного упоминания CAT
   - Упоминаются только онлайн-переводчики
   - Компания маленькая (индивидуальный переводчик)
   - Описание общее без технических деталей

Компания:
{company_info}

Верни JSON:
{{
  "has_cat": true/false,
  "confidence": число от 0.0 до 1.0,
  "evidence": "конкретная причина",
  "product": "название системы или пусто"
}}
"""

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Ты определяешь использование CAT-систем в компаниях."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }

        for attempt in range(1, max_retries + 1):
            async with self.semaphore:
                # Пауза между запросами
                if attempt > 1:
                    wait_time = 5 * (2 ** (attempt - 1))  # Экспоненциальная задержка
                    wait_time = min(wait_time, 60)
                    print(f"Повторная попытка {attempt} для {company_data['name'][:30]}, жду {wait_time} сек")
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(2.0)  # Базовая задержка

                try:
                    async with self.session.post(url, headers=headers, json=payload) as response:
                        # Обработка rate limit
                        if response.status == 429:
                            await self.handle_rate_limit(response.headers, company_data['name'])
                            continue  # Продолжаем цикл попыток

                        if response.status != 200:
                            print(f"HTTP {response.status} для {company_data['name'][:30]}, попытка {attempt}")
                            if attempt < max_retries:
                                continue
                            return {
                                "has_cat": False,
                                "confidence": 0.1,
                                "evidence": f"HTTP error {response.status}",
                                "product": ""
                            }

                        data = await response.json()
                        text = data["choices"][0]["message"]["content"]

                        # Парсинг JSON
                        match = re.search(r'\{.*\}', text, re.DOTALL)

                        if match:
                            try:
                                result = json.loads(match.group())

                                if "has_cat" not in result:
                                    result["has_cat"] = False

                                if "confidence" in result:
                                    try:
                                        conf = float(result["confidence"])
                                        conf = max(0.0, min(1.0, conf))
                                        if not result["has_cat"]:
                                            conf = min(conf, 0.3)
                                        result["confidence"] = round(conf, 2)
                                    except:
                                        result["confidence"] = 0.1
                                else:
                                    result["confidence"] = 0.1

                                result["evidence"] = result.get("evidence", "Нет данных")
                                result["product"] = result.get("product", "")

                                # Сброс таймера rate limit при успешном запросе
                                if self.rate_limit_wait > 0:
                                    self.rate_limit_wait = max(0, self.rate_limit_wait - 10)

                                return result

                            except json.JSONDecodeError as e:
                                print(f"Ошибка JSON для {company_data['name'][:30]}: {e}")
                                if attempt < max_retries:
                                    continue
                                return {
                                    "has_cat": False,
                                    "confidence": 0.1,
                                    "evidence": "Ошибка парсинга JSON",
                                    "product": ""
                                }
                        else:
                            if attempt < max_retries:
                                continue
                            return {
                                "has_cat": False,
                                "confidence": 0.1,
                                "evidence": "Нет JSON в ответе",
                                "product": ""
                            }

                except asyncio.TimeoutError:
                    print(f"Таймаут для {company_data['name'][:30]}, попытка {attempt}")
                    if attempt < max_retries:
                        continue
                    return {
                        "has_cat": False,
                        "confidence": 0.1,
                        "evidence": "Таймаут",
                        "product": ""
                    }
                except Exception as e:
                    print(f"Ошибка для {company_data['name'][:30]}: {e}, попытка {attempt}")
                    if attempt < max_retries:
                        continue
                    return {
                        "has_cat": False,
                        "confidence": 0.1,
                        "evidence": f"Ошибка: {str(e)[:50]}",
                        "product": ""
                    }

        # Если все попытки исчерпаны
        return {
            "has_cat": False,
            "confidence": 0.1,
            "evidence": "Все попытки запроса исчерпаны",
            "product": ""
        }

    async def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Обработка батча компаний"""
        tasks = []
        for company_data in batch:
            task = asyncio.create_task(self.classify_company(company_data))
            tasks.append((task, company_data))

        batch_results = []

        for task, company_data in tasks:
            try:
                result = await task

                full_result = {
                    **company_data,
                    **result,
                    "source": "Async Groq + Llama-3",
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                batch_results.append(full_result)
                self.processed_count += 1

                if result.get("has_cat", False):
                    self.cat_count += 1
                    status = "CAT"
                else:
                    status = "нет"

                progress = self.processed_count
                elapsed = time.time() - self.start_time
                speed = progress / elapsed * 60 if elapsed > 0 else 0

                print(f"[{progress:4d}] {company_data['name'][:40]:40s} | "
                      f"{status} (conf: {result.get('confidence', 0.0):.2f}) | "
                      f"{speed:.1f} ком/мин | Rate wait: {self.rate_limit_wait:.1f}s")

            except Exception as e:
                print(f"Ошибка обработки {company_data['name'][:30]}: {e}")
                self.errors += 1

        return batch_results

    async def process_companies(self, companies: List[Dict], limit: Optional[int] = None):
        """Обработка компаний"""
        self.start_time = time.time()

        if limit and len(companies) > limit:
            companies = companies[:limit]

        print(f"\nОбрабатываю {len(companies)} компаний")
        print(f"Максимум одновременных запросов: {self.max_concurrent}")
        print("=" * 70)

        total_batches = (len(companies) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num in range(0, len(companies), BATCH_SIZE):
            batch = companies[batch_num:batch_num + BATCH_SIZE]
            current_batch = batch_num // BATCH_SIZE + 1

            print(f"\nБатч {current_batch}/{total_batches} ({len(batch)} компаний)")

            batch_results = await self.process_batch(batch)
            self.results.extend(batch_results)

            # Сохраняем промежуточные результаты
            if self.processed_count % SAVE_EVERY == 0 or current_batch == total_batches:
                await self.save_results()

            # Пауза между батчами (кроме последнего)
            if current_batch < total_batches:
                pause = 10  # Увеличиваем паузу между батчами
                print(f"Пауза между батчами: {pause} секунд...")
                await asyncio.sleep(pause)

        return self.results

    async def save_results(self):
        """Сохранение результатов"""
        if not self.results:
            return

        try:
            # Сохраняем все результаты
            df = pd.DataFrame(self.results)
            df.to_csv("cat_results_async.csv", index=False, encoding='utf-8-sig')

            # Сохраняем только CAT-компании
            cat_df = df[df['has_cat'] == True]
            if len(cat_df) > 0:
                cat_df.to_csv("cat_companies.csv", index=False, encoding='utf-8-sig')

            print(f"\nСохранено: {len(self.results)} результатов, {len(cat_df)} CAT-компаний")

        except Exception as e:
            print(f"Ошибка сохранения: {e}")


def load_companies(file_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Загрузка компаний"""
    print(f"Загружаю {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"Загружено: {len(df)} компаний")

        # Фильтрация
        mask = (
                df['nace_code'].astype(str).str.startswith(('62', '63.11')) |
                df['name'].str.contains('перевод|translation|лингв|локализ|localization|транслей',
                                        case=False, na=False)
        )

        companies = df[mask].copy()

        if limit and len(companies) > limit:
            companies = companies.head(limit)

        print(f"Отобрано для анализа: {len(companies)}")

        # Подготовка данных
        companies_list = []
        for row in companies.itertuples():
            companies_list.append({
                "inn": str(getattr(row, 'inn', '')),
                "name": row.name,
                "revenue": int(getattr(row, 'revenue', 0)),
                "nace_code": getattr(row, 'nace_code', ''),
                "description": getattr(row, 'description', '')[:500] if getattr(row, 'description', '') else ""
            })

        return companies_list

    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
        return []
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return []


async def main():
    print("CAT-ДЕТЕКТОР (Асинхронная версия)")
    print("=" * 70)

    if not GROQ_API_KEY:
        print("Ошибка: GROQ_API_KEY не найден в переменных окружения")
        return

    # Загружаем компании
    companies = load_companies(
        "/Users/samvelsimavoryan/PycharmProjects/cat_test_task/src/data/candidate_companies_simple.csv",
        limit=None
    )

    if not companies:
        print("Нет данных для обработки")
        return

    # Создаем детектор
    async with AsyncCatDetector(GROQ_API_KEY, max_concurrent=MAX_CONCURRENT_REQUESTS) as detector:
        # Обработка компаний
        await detector.process_companies(companies)

        # Статистика
        elapsed = time.time() - detector.start_time
        print(f"\n" + "=" * 70)
        print(f"ОБРАБОТКА ЗАВЕРШЕНА")
        print(f"Всего обработано: {detector.processed_count} компаний")
        print(f"Найдено CAT-компаний: {detector.cat_count}")
        print(f"Ошибок: {detector.errors}")
        print(f"Результаты сохранены в cat_results_async.csv")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nОбработка прервана пользователем.")
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")