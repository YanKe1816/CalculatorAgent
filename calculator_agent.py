"""Calculator Agent implemented using an OpenAI Agent SDK style structure."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from typing import Any, Protocol


class Tool(Protocol):
    """Protocol that represents a callable tool for the agent."""

    name: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - interface
        ...


@dataclass
class CalculatorTool:
    """A simple tool that evaluates arithmetic expressions safely."""

    name: str = "calculate"

    def __call__(self, expression: str) -> str:
        try:
            result = evaluate_expression(expression)
        except ValueError:
            return "输入有误，请输入正确的算式"
        return str(result)


def evaluate_expression(expression: str) -> float:
    """Safely evaluate a mathematical expression using the AST module."""

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ValueError("invalid expression") from exc

    if not _is_expression_safe(tree):
        raise ValueError("unsafe expression")

    compiled = compile(tree, filename="<expr>", mode="eval")
    return eval(compiled, {"__builtins__": {}}, ALLOWED_FUNCTIONS.copy())


ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Load,
    ast.Call,
    ast.Name,
)

ALLOWED_FUNCTIONS = {"abs": abs, "round": round}


def _is_expression_safe(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ALLOWED_NODES):
            return False
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name) or child.func.id not in ALLOWED_FUNCTIONS:
                return False
        if isinstance(child, ast.Name) and child.id not in ALLOWED_FUNCTIONS:
            return False
        if isinstance(child, ast.Constant) and not isinstance(child.value, (int, float)):
            return False
    return True


class CalculatorAgent:
    """Agent that routes prompts to the calculator tool."""

    def __init__(self, tool: Tool | None = None) -> None:
        self.tool: Tool = tool or CalculatorTool()

    def run(self, expression: str) -> str:
        if not expression or not expression.strip():
            return "输入有误，请输入正确的算式"
        result = self.tool(expression)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculator Agent CLI")
    parser.add_argument("expression", type=str, help="Expression to evaluate")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    agent = CalculatorAgent()
    output = agent.run(args.expression)
    print(output)


if __name__ == "__main__":
    main()
