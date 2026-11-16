"""Convert a '# %%'-delimited Python script into a Jupyter notebook.
Usage: python tools/py_to_ipynb.py input.py output.ipynb"""
import json
import sys

src = open(sys.argv[1], encoding="utf-8").read().splitlines()
cells, cur, kind = [], [], None


def flush():
    if kind is None:
        return
    lines = cur[:]
    while lines and not lines[-1].strip():
        lines.pop()
    if kind == "markdown":
        lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
    text = [l + "\n" for l in lines]
    if text:
        text[-1] = text[-1].rstrip("\n")
    cell = {"cell_type": kind, "metadata": {}, "source": text}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    cells.append(cell)


for line in src:
    if line.startswith("# %%"):
        flush()
        kind, cur = ("markdown" if "[markdown]" in line else "code"), []
    elif kind is not None:
        cur.append(line)
flush()
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
print(f"{len(cells)} cells -> {sys.argv[2]}")
