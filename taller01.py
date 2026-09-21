#!/usr/bin/env python3
"""Experimento Taller 01 — MMIA 6013.

Modo de ejecucion 
  python taller01.py parte0
  python taller01.py parte1 --backend openai
  python taller01.py parte2a --backend openai
  python taller01.py parte2b --backend openai
  python taller01.py parte2b --backend ollama
  python taller01.py parte3 --backend openai
  python taller01.py parte4 --backend openai

"""
from __future__ import annotations

import argparse, csv, json, math, os, re, time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "salidas"; OUT.mkdir(exist_ok=True)
RESULTS = ROOT / "resultados.csv"
FIELDS = ["timestamp_utc", "parte", "modelo", "backend", "caso_id", "parametros", "corrida",
          "prompt", "salida", "respuesta_extraida", "esperada", "acierto", "input_tokens",
          "output_tokens", "reasoning_tokens", "visible_output_tokens", "latencia_s", "costo_usd", "error"]

MODELOS = {
    "economico": {"id": "gpt-4o-mini", "in": .15, "out": .60, "verified": "2026-08-26"},
    "razonamiento": {"id": "gpt-5.6-luna", "in": .20, "out": 1.20, "verified": "2026-09-18"},
    "local": {"id": "qwen3:1.7b", "in": 0, "out": 0, "verified": "—"},
}

def cases(name: str = "casos.json") -> list[dict[str, Any]]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))

def save(row: dict[str, Any]) -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})

def number_from(text: str) -> int | float | None:
    try:
        obj = json.loads(text)
        val = obj.get("answer", obj.get("respuesta"))
        return float(val) if "." in str(val) else int(val)
    except Exception:
        found = re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?", text)
        if not found: return None
        n = found[-1]
        return float(n) if "." in n else int(n)

def prompt_for(question: str, variant: str = "structured") -> str:
    base = "Resuelve el problema. La respuesta debe ser un único número."
    if variant == "fewshot":
        base += "\nEjemplo: 3 + 4 = 7. Respuesta: 7\nEjemplo: 20 - 8 = 12. Respuesta: 12"
    elif variant == "cot":
        base += " Razona paso a paso de forma breve y termina con 'Respuesta: <número>'."
    elif variant == "structured":
        base += ' Devuelve exclusivamente JSON válido con esta forma: {"answer": numero}.'
    return f"{base}\n\nProblema: {question}"

def openai_call(model_key: str, prompt: str, temperature: float | None = None,
                top_p: float | None = None, effort: str | None = None, top_k: int | None = None,
                structured: bool = False) -> dict[str, Any]:
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI()
    model = MODELOS[model_key]
    kwargs: dict[str, Any] = {"model": model["id"], "input": prompt}
    if effort: kwargs["reasoning"] = {"effort": effort}
    if temperature is not None: kwargs["temperature"] = temperature
    if top_p is not None: kwargs["top_p"] = top_p
    if top_k is not None: kwargs["top_k"] = top_k
    if structured:
        kwargs["text"] = {"format": {"type": "json_schema", "name": "respuesta_numerica", "strict": True,
                                      "schema": {"type": "object", "properties": {"answer": {"type": "number"}},
                                                 "required": ["answer"], "additionalProperties": False}}}
    started = time.perf_counter(); r = client.responses.create(**kwargs); latency = time.perf_counter() - started
    usage = getattr(r, "usage", None)
    inp, out = getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0
    details = getattr(usage, "output_tokens_details", None)
    reason = getattr(details, "reasoning_tokens", 0) if details else 0
    return {"text": r.output_text, "input": inp, "output": out, "reason": reason or 0, "latency": latency,
            "cost": inp * model["in"] / 1e6 + out * model["out"] / 1e6}

def ollama_call(prompt: str, temperature: float | None = None, top_p: float | None = None,
                top_k: int | None = None, thinking: bool = False) -> dict[str, Any]:
    opts = {k: v for k, v in {"temperature": temperature, "top_p": top_p, "top_k": top_k}.items() if v is not None}
    started = time.perf_counter()
    r = requests.post("http://localhost:11434/api/chat", json={"model": MODELOS["local"]["id"], "stream": False,
                      "think": thinking, "messages": [{"role": "user", "content": prompt}], "options": opts}, timeout=300)
    r.raise_for_status(); raw = r.json(); latency = time.perf_counter() - started
    text = raw.get("message", {}).get("content", "")
    think = raw.get("message", {}).get("thinking", "")
    return {"text": text, "input": raw.get("prompt_eval_count", 0), "output": raw.get("eval_count", 0),
            "reason": len(think.split()), "latency": latency, "cost": 0}

def run_one(part: str, case: dict[str, Any], backend: str, model_key: str, params: dict[str, Any], run: int,
            variant: str = "structured") -> dict[str, Any]:
    p = prompt_for(case["pregunta"], variant)
    try:
        data = (openai_call(model_key, p, params.get("temperature"), params.get("top_p"), params.get("effort"), params.get("top_k"), variant == "structured")
                if backend == "openai" else ollama_call(p, params.get("temperature"), params.get("top_p"),
                                                          params.get("top_k"), params.get("thinking", False)))
        answer = number_from(data["text"]); ok = answer == case["respuesta"]
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "parte": part, "modelo": MODELOS[model_key]["id"],
               "backend": backend, "caso_id": case["id"], "parametros": json.dumps(params), "corrida": run,
               "prompt": p, "salida": data["text"], "respuesta_extraida": answer, "esperada": case["respuesta"],
               "acierto": ok, "input_tokens": data["input"], "output_tokens": data["output"],
               "reasoning_tokens": data["reason"], "visible_output_tokens": max(0, data["output"] - data["reason"]),
               "latencia_s": round(data["latency"], 3), "costo_usd": data["cost"], "error": ""}
    except Exception as e:
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "parte": part, "modelo": MODELOS[model_key]["id"],
               "backend": backend, "caso_id": case["id"], "parametros": json.dumps(params), "corrida": run,
               "prompt": p, "salida": "", "respuesta_extraida": "", "esperada": case["respuesta"], "acierto": False,
               "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "visible_output_tokens": 0,
               "latencia_s": 0, "costo_usd": 0, "error": f"{type(e).__name__}: {e}"}
    save(row); return row

def summary(part: str) -> None:
    df = pd.read_csv(RESULTS); df_part = df[df.parte == part]
    if df_part.empty: return
    errores = df_part[df_part.error.notna() & (df_part.error != "")]
    if not errores.empty:
        print(f"\n[Aviso] Se registraron {len(errores)} llamadas con error en la Parte {part}:")
        for err in errores["error"].unique():
            print(f"  - {err}")
    df_ok = df_part[df_part.error.isna() | (df_part.error == "")]
    if df_ok.empty: return
    group = df_ok.groupby(["modelo", "parametros"], dropna=False).agg(exactitud=("acierto", "mean"),
        latencia_s=("latencia_s", "mean"), input_tokens=("input_tokens", "mean"), output_tokens=("output_tokens", "mean"),
        reasoning_tokens=("reasoning_tokens", "mean"), costo_usd=("costo_usd", "sum")).reset_index()
    group.to_csv(OUT / f"resumen_{part}.csv", index=False)
    print("\n" + group.to_string(index=False))

def part0() -> None:
    """Parte 0: GPT-2 local, figuras de distribución y generaciones crudas."""
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, use_safetensors=False, dtype=torch.float32)
    model.eval()
    prefixes = {"seguro": "The capital of France is", "incierto": "In the future, artificial intelligence will"}
    rows = []
    for label, prefix in prefixes.items():
        inputs = tok(prefix, return_tensors="pt")
        with torch.no_grad(): logits = model(**inputs).logits[0, -1]
        for t in [.1, .7, 1., 1.5, 2.]:
            probs = torch.softmax(logits / t, dim=0); vals, ids = torch.topk(probs, 15)
            entropy = float(-(probs * torch.log2(probs.clamp_min(1e-12))).sum())
            nucleus = int(torch.searchsorted(torch.cumsum(torch.sort(probs, descending=True).values, 0), .9).item() + 1)
            rows.append({"prefijo": label, "T": t, "entropia_bits": entropy, "nucleo_90": nucleus})
            plt.figure(figsize=(10, 4)); plt.bar(range(15), vals.tolist()); plt.xticks(range(15), [tok.decode([i]) for i in ids], rotation=55, ha="right")
            plt.ylabel("Probabilidad"); plt.title(f"GPT-2: {label}, T={t}"); plt.tight_layout(); plt.savefig(OUT / f"p0_{label}_T{t}.png", dpi=160); plt.close()
        for mode, kwargs in [("greedy", {"do_sample": False}), ("topk1", {"do_sample": True, "top_k": 1})]:
            for run in range(5):
                torch.manual_seed(123); output = model.generate(**inputs, max_new_tokens=40, **kwargs)
                text = tok.decode(output[0], skip_special_tokens=True)
                save({"timestamp_utc": datetime.now(timezone.utc).isoformat(), "parte": "0b", "modelo": name, "backend": "local",
                      "caso_id": label, "parametros": json.dumps(kwargs), "corrida": run, "prompt": prefix, "salida": text,
                      "costo_usd": 0})
    instruction = prompt_for(cases()[0]["pregunta"])
    out = model.generate(**tok(instruction, return_tensors="pt"), max_new_tokens=100, do_sample=False)
    (OUT / "p0c_modelo_base.txt").write_text(tok.decode(out[0], skip_special_tokens=True), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "p0_tabla_distribucion.csv", index=False)
    print("Parte 0 terminada. Figuras y tabla en salidas/.")

def part1(backend: str) -> None:
    configs = [("openai", "razonamiento", {"effort": "low"}), ("openai", "economico", {"temperature": .2}),
               ("ollama", "local", {"temperature": .2})]
    for b, m, params in configs:
        if b != backend and backend != "all": continue
        for c in cases(): run_one("1", c, b, m, params, 1)
    summary("1")

def part2a(backend: str) -> None:
    tests = ([{"temperature": .2}, {"temperature": 1.2}, {"top_p": .5}, {"top_p": 1.0}, {"top_k": 5}]
             if backend == "ollama" else [{"temperature": .2}, {"temperature": 1.2}, {"top_p": .5}, {"top_p": 1.0}, {"top_k": 5}])
    key = "local" if backend == "ollama" else "economico"
    for params in tests:
        for n in range(3): run_one("2a", cases()[0], backend, key, params, n)
    summary("2a")

def part2b(backend: str) -> None:
    key = "local" if backend == "ollama" else "economico"
    for temp in [0, .3, .7, 1., 1.5]:
        for top_p in [.5, .9, 1.]:
            for c in cases():
                for n in range(5): run_one("2b", c, backend, key, {"temperature": temp, "top_p": top_p}, n)
    if backend == "ollama":
        for k in [1, 5, 40]:
            for c in cases():
                for n in range(5): run_one("2b_topk", c, backend, key, {"temperature": .7, "top_k": k}, n)
    summary("2b")

def part3(backend: str) -> None:
    key = "local" if backend == "ollama" else "economico"
    for variant in ["zero", "fewshot", "cot", "structured"]:
        for c in cases(): run_one("3_" + variant, c, backend, key, {"temperature": .2}, 1, variant)
        summary("3_" + variant)

def part4(backend: str) -> None:
    key = "local" if backend == "ollama" else "razonamiento"
    levels = (["low", "medium", "high"] if backend == "openai" else [False, True])
    for level in levels:
        params = {"effort": level} if backend == "openai" else {"thinking": level}
        for c in cases(): run_one("4a", c, backend, key, params, 1)
    for level in [levels[0], levels[-1]]:
        params = {"effort": level} if backend == "openai" else {"thinking": level}
        for c in cases("contaminados.json"): run_one("4b", c, backend, key, params, 1)
    summary("4a"); summary("4b")
    df = pd.read_csv(RESULTS); d = df[df.parte == "4a"].groupby("parametros").agg(exactitud=("acierto", "mean"), reasoning=("reasoning_tokens", "mean"), costo=("costo_usd", "sum")).reset_index()
    for x, y, file in [("reasoning", "exactitud", "p4_exactitud_vs_razonamiento.png"), ("costo", "exactitud", "p4_costo_vs_exactitud.png")]:
        plt.figure(); plt.plot(d[x], d[y], "o-"); plt.xlabel(x); plt.ylabel(y); plt.tight_layout(); plt.savefig(OUT / file, dpi=160); plt.close()

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("parte", choices=["parte0", "parte1", "parte2a", "parte2b", "parte3", "parte4"])
    p.add_argument("--backend", choices=["openai", "ollama", "all"], default="openai"); a = p.parse_args()
    {"parte0": lambda: part0(), "parte1": lambda: part1(a.backend), "parte2a": lambda: part2a(a.backend),
     "parte2b": lambda: part2b(a.backend), "parte3": lambda: part3(a.backend), "parte4": lambda: part4(a.backend)}[a.parte]()

if __name__ == "__main__": main()
