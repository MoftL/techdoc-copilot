import sys

from config import CONFIG
from retrieval.answer import generate_answer
from retrieval.pipelines import PIPELINES


def main():
    keys = list(PIPELINES)
    while True:
        print("\n" + "=" * 50)
        print("        TECHDOC COPILOT")
        print("=" * 50)
        for number, key in enumerate(keys, 1):
            print(f"  {number}. {PIPELINES[key]['name']}: {PIPELINES[key]['summary']}")
        print("  0. Exit")

        choice = input(f"\nSelect a pipeline [0-{len(keys)}]: ").strip()
        if choice == "0":
            print("Exiting Copilot.")
            sys.exit(0)
        if not choice.isdigit() or not 1 <= int(choice) <= len(keys):
            print(f"Invalid selection. Enter a number between 0 and {len(keys)}.")
            continue

        question = input("\nEnter your technical question: ").strip()[:CONFIG["max_question_chars"]]
        if not question:
            continue

        sources = PIPELINES[keys[int(choice) - 1]]["retrieve"](question)
        print("\n" + generate_answer(question, sources))
        print("\nSOURCES:")
        for number, source in enumerate(sources, 1):
            print(f"  [{number}] {source['section'] or source['title']} | {source['url']}")


if __name__ == "__main__":
    main()
