import asyncio
import sys

from src.data.analyze_data import analyze_msp_data, save_candidate_companies
from src.scripts.sync_detect import main as sync_detect_main
from src.scripts.async_detect import main as async_detect_main


def print_menu():
    print("\nCAT ANALYSIS PIPELINE")
    print("=" * 50)
    print("1. Проанализировать реестр МСП + сохранить кандидатов")
    print("2. CAT-детекция (синхронно)")
    print("3. CAT-детекция (асинхронно)")
    print("Выход")
    print("=" * 50)


def run_data_analysis():
    print("\nШАГ 1: Анализ данных МСП")
    df = analyze_msp_data()
    save_candidate_companies(df)
    print("Кандидаты сохранены")


def run_sync_detection():
    print("\n ШАГ 2: CAT-детекция (sync)")
    sync_detect_main()


def run_async_detection():
    print("\nШАГ 3: CAT-детекция (async)")
    asyncio.run(async_detect_main())


def main():
    while True:
        print_menu()
        choice = input("Выбери действие: ").strip()

        if choice == "1":
            run_data_analysis()

        elif choice == "2":
            run_sync_detection()

        elif choice == "3":
            run_async_detection()

        elif choice == "0":
            print("Выход")
            sys.exit(0)

        else:
            print("Неверный выбор")


if __name__ == "__main__":
    main()