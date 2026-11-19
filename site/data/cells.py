"""Table cells for the pages that results.js draws: the generators format each cell's text here,
mark the bold (best) and muted ones, and keep the run each value comes from."""
from __future__ import annotations


def text(content: str, muted: bool = False) -> dict:
    return {"text": content, "muted": True} if muted else {"text": content}


def number(value: float, digits: int, run_id: str | None = None) -> dict:
    cell = {"text": f"{value:.{digits}f}", "value": value}
    if run_id:
        cell["run_id"] = run_id
    return cell


def bold_lowest(cells: list[dict]) -> None:
    """Bold the lowest of these cells as shown, so values that tie at the shown precision are all bold."""
    numbers = [cell for cell in cells if "value" in cell]
    lowest = min(float(cell["text"]) for cell in numbers)
    for cell in numbers:
        if float(cell["text"]) == lowest:
            cell["best"] = True
