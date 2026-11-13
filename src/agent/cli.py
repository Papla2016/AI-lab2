"""Linux CLI: each command can run in a separate process with the same SQLite DB."""
import argparse
import json
import sys
from uuid import uuid4

from .runtime import open_agent


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="ЛР2 — LangGraph помощник ALT Linux")
    commands = root.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="новый вопрос или явная команда инструмента")
    ask.add_argument("question")
    for name in ("resume", "status", "continue"):
        commands.add_parser(name)
    for name, command in commands.choices.items():
        command.add_argument("--thread-id", required=name != "ask")
        command.add_argument("--trace", action="store_true", help="узлы и решения в stderr")
    resume = commands.choices["resume"]
    resume.add_argument("--decision", choices=["yes", "no"], required=True)
    resume.add_argument("--action-id", help="проверить идентификатор из запроса подтверждения")
    return root


def main() -> None:
    args = parser().parse_args()
    thread_id = args.thread_id or uuid4().hex

    def observer(event: dict) -> None:
        if args.trace:
            print(json.dumps(event, ensure_ascii=False), file=sys.stderr)

    try:
        with open_agent(observer=observer) as agent:
            if args.command == "status":
                snapshot = agent.state(thread_id)
                pending = [x.value for task in snapshot.tasks for x in task.interrupts]
                output = dict(thread_id=thread_id, next=list(snapshot.next),
                              history_messages=len(snapshot.values.get("messages", [])),
                              interrupt=pending[0] if pending else None,
                              result=snapshot.values.get("final"))
            elif args.command == "ask":
                output = agent.ask(args.question, thread_id).model_dump()
            elif args.command == "resume":
                output = agent.resume(thread_id, args.decision == "yes", args.action_id).model_dump()
            else:
                output = agent.continue_run(thread_id).model_dump()
            print(json.dumps(output, ensure_ascii=False, indent=2))
    except Exception as exc:
        # Never expose provider bodies, paths, prompts or credentials.
        code = "invalid_request" if isinstance(exc, ValueError) else "execution_error"
        print(json.dumps({"error": code, "thread_id": thread_id,
                          "hint": "Проверьте аргументы, status, настройки .env и доступность зависимостей."},
                         ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
