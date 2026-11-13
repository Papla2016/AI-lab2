"""Read models and explicit commands for the existing persistent GraphAgent.

No Streamlit dependency, dialogue cache, or separate generation chain lives here.
Every operation closes its SQLite connection before returning to the renderer.
"""
from collections import deque
from dataclasses import dataclass
import json
from typing import Callable

from langchain_core.messages import AIMessage

from .runtime import GraphAgent, Outcome, open_agent
from .schemas import AgentResult, AgentSettings


@dataclass
class Dialogue:
    thread_id: str
    messages: list
    values: dict
    pending: dict | None
    next_nodes: tuple[str, ...]
    runs: list[str]
    selected_run: str | None
    execution: dict
    events: list[dict]
    trace_truncated: bool
    diagram: str
    mermaid: str


def result_message(message) -> AgentResult | None:
    if not isinstance(message, AIMessage) or not isinstance(message.content, str):
        return None
    try:
        return AgentResult.model_validate_json(message.content)
    except ValueError:
        return None


def trace_events(path, thread_id: str, run_id: str | None, limit: int = 500) -> tuple[list[dict], bool]:
    """Read only this run's actual audit records, tolerating an incomplete tail."""
    events: deque = deque(maxlen=limit)
    count = 0
    if not run_id or not path.exists():
        return [], False
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if (isinstance(event, dict) and event.get("thread_id") == thread_id
                    and event.get("run_id") == run_id):
                events.append(event)
                count += 1
    return list(events), count > limit


def graph_diagram(graph) -> str:
    """DOT from the compiled graph; no hand-maintained list of transitions."""
    drawing = graph.get_graph()
    lines = ['digraph agent {', 'rankdir=TB;', 'node [shape=box, style=rounded];']
    for key in drawing.nodes:
        lines.append(f'{json.dumps(key)} [label={json.dumps(key)}];')
    for edge in drawing.edges:
        style = ' [style=dashed]' if edge.conditional else ''
        lines.append(f'{json.dumps(edge.source)} -> {json.dumps(edge.target)}{style};')
    return "\n".join([*lines, "}"])


class AgentWebService:
    def __init__(self, settings: AgentSettings, *, opener: Callable = open_agent) -> None:
        self.settings, self.opener = settings, opener

    def threads(self) -> list[str]:
        with self.opener(self.settings) as agent:
            return agent.threads()

    def inspect(self, thread_id: str, run_id: str | None = None) -> Dialogue:
        GraphAgent.config(thread_id)
        with self.opener(self.settings) as agent:
            snapshot = agent.state(thread_id)
            values = snapshot.values
            messages = values.get("messages", [])
            runs = []
            for message in messages:
                result = result_message(message)
                if result and result.run_id not in runs:
                    runs.append(result.run_id)
            current = values.get("run_id")
            if current and current not in runs:
                runs.append(current)
            selected = run_id if run_id in runs else current
            execution = values
            if selected and selected != current:
                # The most recent checkpoint of that run includes all critique
                # iterations. Messages remain those of the current dialogue.
                execution = next((item.values for item in agent.graph.get_state_history(agent.config(thread_id))
                                  if item.values.get("run_id") == selected), {})
            pending = [item.value for task in snapshot.tasks for item in task.interrupts]
            events, truncated = trace_events(self.settings.agent_log_path, thread_id, selected)
            return Dialogue(thread_id, messages, values, pending[0] if pending else None,
                            snapshot.next, runs, selected, execution, events, truncated,
                            graph_diagram(agent.graph), agent.graph.get_graph().draw_mermaid())

    def ask(self, thread_id: str, question: str, observer=None) -> Outcome:
        with self.opener(self.settings, observer=observer) as agent:
            return agent.ask(question, thread_id)

    def resume(self, thread_id: str, approved: bool, action_id: str, observer=None) -> Outcome:
        with self.opener(self.settings, observer=observer) as agent:
            snapshot = agent.state(thread_id)
            pending = [item.value for task in snapshot.tasks for item in task.interrupts]
            # A stale button must never approve OR reject a different action.
            if not pending or pending[0].get("action_id") != action_id:
                raise ValueError("Подтверждение устарело. Обновите состояние диалога.")
            return agent.resume(thread_id, approved, action_id)

    def continue_run(self, thread_id: str, observer=None) -> Outcome:
        with self.opener(self.settings, observer=observer) as agent:
            return agent.continue_run(thread_id)
