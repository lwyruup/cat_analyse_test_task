import pandas as pd
import numpy as np


def analyze_msp_data():
    """Анализируем реестр МСП"""
    print("АНАЛИЗ РЕЕСТРА МСП 2025")
    print("=" * 60)

    # Загружаем данные
    df = pd.read_csv('/Users/samvelsimavoryan/PycharmProjects/cat_test_task/src/data/data_new.csv')

    print(f"Всего записей: {df.shape[0]:,}")
    print(f"Колонок: {df.shape[1]}")

    # 1. Проверяем заполненность выручки
    revenue_stats = df['revenue'].notna().sum()
    print(f"\nЗАПОЛНЕННОСТЬ ДАННЫХ:")
    print(f"   С выручкой: {revenue_stats:,} ({revenue_stats / len(df) * 100:.1f}%)")
    print(f"   Без выручки: {len(df) - revenue_stats:,}")

    # 2. Анализ выручки у тех, у кого она есть
    if revenue_stats > 0:
        revenue_data = df[df['revenue'].notna()]['revenue']
        print(f"\n СТАТИСТИКА ВЫРУЧКИ:")
        print(f"   Максимум: {revenue_data.max():,.0f} руб.")

        print(f"   Среднее: {revenue_data.mean():,.0f} руб.")

        # Компании с выручкой > 100 млн
        big_companies = df[df['revenue'] >= 100_000_000]
        print(f"   >100 млн ₽: {len(big_companies):,} компаний")

    # 3. Поиск IT и переводческих компаний
    print(f"\n👨КОМПАНИИ ПО ОТРАСЛЯМ:")

    # IT компании (коды ОКВЭД)
    it_codes = ['62', '62.01', '62.02', '62.03', '62.09', '63', '63.1', '63.11']
    it_companies = df[df['main_nace_code'].astype(str).str.startswith(tuple(it_codes))]
    print(f"   IT компании: {len(it_companies):,}")

    # Переводческие компании по названию
    translation_keywords = ['перевод', 'translation', 'лингвист', 'локализ', 'localiz']

    # Создаем маску для поиска в названиях
    mask = df['org_name'].astype(str).str.lower().apply(
        lambda x: any(keyword in x for keyword in translation_keywords)
    )
    translation_companies = df[mask]
    print(f"Переводческие компании: {len(translation_companies):,}")

    # 4. Ищем конкретные компании для теста
    print(f"\nТЕСТОВЫЕ КОМПАНИИ:")

    # Яндекс (должен быть, если выручка > 2 млрд ₽, то не в МСП)
    yandex_mask = df['org_name'].astype(str).str.contains('ЯНДЕКС|Яндекс|Yandex', case=False, na=False)
    yandex_companies = df[yandex_mask]
    print(f"   Яндекс: найдено {len(yandex_companies)} записей")

    if not yandex_companies.empty:
        for _, row in yandex_companies.head().iterrows():
            print(f"      - {row['org_name']}: выручка {row['revenue'] if pd.notna(row['revenue']) else 'нет данных'}")

    # 5. Проверяем, какие компании подходят под наши критерии
    print(f"\nКОМПАНИИ ДЛЯ НАШЕГО ЗАДАНИЯ:")

    # Фильтры:
    # 1) Выручка >= 100 млн ₽
    # 2) IT или переводческие компании
    # 3) Есть выручка

    filtered = df[
        (df['revenue'] >= 100_000_000) &
        (
                df['main_nace_code'].astype(str).str.startswith(tuple(it_codes)) |
                df['org_name'].astype(str).str.lower().str.contains('|'.join(translation_keywords))
        )
        ]

    print(f"   Подходят под критерии: {len(filtered):,}")

    if not filtered.empty:
        print(f"\n   Примеры подходящих компаний:")
        for _, row in filtered.head(5).iterrows():
            print(f"{row['org_name'][:50]}...")
            print(f" ИНН: {row['tax_number']}, Выручка: {row['revenue']:,.0f} ₽, Отрасль: {row['main_nace_code']}")
            print()

    return df


# Сохраняем список подходящих компаний
def save_candidate_companies(df):
    """Сохраняем список компаний-кандидатов"""

    # коды ОКВЭД
    it_nace_codes = ['62', '62.01', '62.02', '62.03', '58', '63', '63.1']

    # Переводческие компании
    translation_keywords = [
        'перевод', 'translation', 'лингвист', 'лингво', 'локализ',
        'localization', 'переводческ', 'translat', 'interpret'
    ]

    # Применяем фильтры
    candidates = df[
        (df['revenue'] >= 100_000_000) &  # Выручка >= 100 млн
        df['revenue'].notna() &  # Выручка указана
        (
                df['main_nace_code'].astype(str).str.startswith(tuple(it_nace_codes)) |
                df['org_name'].astype(str).str.lower().str.contains('|'.join(translation_keywords))
        )
        ]

    print(f"\nПолучено {len(candidates)} компаний-кандидатов...")
    candidates.to_csv('candidate_companies.csv', index=False)

    # Также сохраняем упрощенную версию
    simplified = candidates[[
        'tax_number', 'org_name', 'revenue', 'employees_count',
        'main_nace_code', 'region', 'start_date'
    ]].copy()

    simplified.columns = ['inn', 'name', 'revenue', 'employees',
                          'nace_code', 'region', 'registration_date']

    simplified.to_csv('candidate_companies_simple.csv', index=False)

    print(f"Сохранено:")
    print(f" candidate_companies.csv ({len(candidates)} записей)")
    print(f" candidate_companies_simple.csv (упрощенная версия)")

    return candidates


if __name__ == "__main__":
    df = analyze_msp_data()
    candidates = save_candidate_companies(df)

    # Дополнительная статистика
    print(f"\nСТАТИСТИКА КАНДИДАТОВ:")
    print(f"   Всего кандидатов: {len(candidates):,}")
    print(f"   Средняя выручка: {candidates['revenue'].mean():,.0f} ₽")
    print(f"   Медианная выручка: {candidates['revenue'].median():,.0f} ₽")

    if len(candidates) > 0:
        print(f"   Максимальная выручка: {candidates['revenue'].max():,.0f} ₽")
        print(f"   Код  ОКВЭД чаще всего: {candidates['main_nace_code'].mode().iloc[0]}")