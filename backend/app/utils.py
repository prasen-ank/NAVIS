import os
import uuid
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List

CHROOT = os.path.join(os.getcwd(), "static")
os.makedirs(CHROOT, exist_ok=True)


def is_chart_request(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["chart", "plot", "diagram", "graph"])


def _try_parse_table_from_text(text: str):
    """Attempt to parse CSV-like text into a DataFrame"""
    try:
        df = pd.read_csv(pd.compat.StringIO(text))
        return df
    except Exception:
        # fallback: try lines split by commas
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            return None
        first = lines[0]
        if "," in first:
            try:
                df = pd.read_csv(pd.compat.StringIO("\n".join(lines)))
                return df
            except Exception:
                return None
        return None


def generate_chart_from_docs(docs: List[str], namespace: str = "default") -> str:
    """Try to find tabular data in docs and plot a simple chart. Returns saved image path or None."""
    for d in docs:
        df = _try_parse_table_from_text(d)
        if df is not None and df.shape[1] >= 2:
            # pick first numeric pair
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) >= 1:
                x = df.index
                y = df[numeric_cols[0]]
                plt.figure(figsize=(6, 4))
                plt.plot(x, y, marker="o")
                plt.title(f"Chart: {numeric_cols[0]}")
                plt.xlabel("row")
                plt.ylabel(numeric_cols[0])
                fname = f"chart_{namespace}_{uuid.uuid4().hex[:8]}.png"
                out = os.path.join("static", fname)
                plt.tight_layout()
                plt.savefig(out)
                plt.close()
                return out
    return None
