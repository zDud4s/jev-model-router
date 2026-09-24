"""The live routing view served at /routing: one self-contained page, no assets.

It polls /routing/traces for requests that moved and /healthz for the catalog
and the subscriptions, and draws each request end to end: what arrived, which
tiers could take it, what Jev read in it, every option weighed, what answered.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Routing · llm-router</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #16181d; --muted: #646b78; --line: #e3e6eb;
  --soft: #eef1f5; --accent: #3b5bdb; --ok: #2b8a3e; --warn: #c77700; --bad: #c92a2a;
  --claude: #d9480f; --codex: #1c7ed6; --local: #7048e8; --other: #868e96;
  --mono: ui-monospace, SFMono-Regular, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --panel: #171a21; --ink: #e6e8ec; --muted: #9aa1ad; --line: #2a2f39;
    --soft: #1f232c; --accent: #748ffc; --ok: #51cf66; --warn: #fcc419; --bad: #ff6b6b;
    --claude: #ff8a4c; --codex: #4dabf7; --local: #9775fa; --other: #adb5bd;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--sans); }
header { display: flex; align-items: center; gap: 16px; padding: 12px 20px; border-bottom: 1px solid var(--line);
  background: var(--panel); position: sticky; top: 0; z-index: 2; flex-wrap: wrap; }
header h1 { font-size: 15px; margin: 0; font-weight: 650; letter-spacing: .01em; }
.live { display: inline-flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 25%, transparent); }
.dot.off { background: var(--bad); box-shadow: none; }
.stat { color: var(--muted); font-size: 12px; }
.stat b { color: var(--ink); font-weight: 600; }
.meters { display: flex; gap: 14px; margin-left: auto; }
.meter { font-size: 12px; color: var(--muted); min-width: 150px; }
.meter .bar { height: 6px; background: var(--soft); border-radius: 3px; overflow: hidden; margin-top: 3px; }
.meter .bar i { display: block; height: 100%; background: var(--accent); }
main { display: grid; grid-template-columns: 340px 1fr; gap: 16px; padding: 16px 20px; align-items: start; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } .meters { margin-left: 0; } }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; }
.compose { padding: 12px; margin-bottom: 12px; }
.compose textarea { width: 100%; min-height: 84px; resize: vertical; border: 1px solid var(--line); border-radius: 8px;
  background: var(--bg); color: var(--ink); padding: 8px 10px; font: 13px/1.4 var(--sans); }
.compose details { margin-top: 6px; font-size: 12px; color: var(--muted); }
.compose details textarea { min-height: 54px; font-family: var(--mono); font-size: 12px; }
.row { display: flex; gap: 8px; margin-top: 8px; align-items: center; flex-wrap: wrap; }
button { border: 1px solid var(--line); background: var(--soft); color: var(--ink); border-radius: 7px; padding: 6px 12px;
  font: 600 13px var(--sans); cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button:disabled { opacity: .5; cursor: wait; }
.hint { font-size: 12px; color: var(--muted); }
.list { max-height: calc(100vh - 290px); overflow: auto; }
.item { padding: 10px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }
.item:hover { background: var(--soft); }
.item.sel { background: color-mix(in srgb, var(--accent) 10%, var(--panel)); box-shadow: inset 3px 0 0 var(--accent); }
.item .p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.item .m { display: flex; gap: 8px; align-items: center; margin-top: 4px; font-size: 12px; color: var(--muted); }
.chip { display: inline-flex; align-items: center; gap: 5px; padding: 1px 8px; border-radius: 999px; font-size: 12px;
  background: var(--soft); color: var(--ink); white-space: nowrap; max-width: 100%; overflow: hidden; text-overflow: ellipsis; }
.chip .sw { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.status-ok { color: var(--ok); } .status-error { color: var(--bad); } .status-running { color: var(--warn); }
.empty { padding: 40px; text-align: center; color: var(--muted); }
.detail { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.card { padding: 14px 16px; }
.card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 10px; font-weight: 650; }
.prompt { white-space: pre-wrap; max-height: 120px; overflow: auto; font-size: 13px; }
.pipeline { display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; margin-top: 12px; }
@media (max-width: 700px) { .pipeline { grid-template-columns: repeat(2, 1fr); } }
.step { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; background: var(--bg); min-width: 0; }
.step .n { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.step .v { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.step .t { font-size: 11px; color: var(--muted); font-family: var(--mono); }
.step.done { border-color: color-mix(in srgb, var(--ok) 45%, var(--line)); }
.step.fail { border-color: var(--bad); }
.step.wait { opacity: .45; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 1100px) { .two { grid-template-columns: 1fr; } }
.need { display: grid; grid-template-columns: 110px 1fr 42px; gap: 8px; align-items: center; margin: 5px 0; font-size: 13px; }
.need .track { position: relative; height: 10px; background: var(--soft); border-radius: 5px; }
.need .fill { position: absolute; inset: 0 auto 0 0; border-radius: 5px; background: var(--accent); }
.need .fill.low { background: color-mix(in srgb, var(--muted) 45%, transparent); }
.need .floor { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--ink); opacity: .35; }
.need .val { font-family: var(--mono); font-size: 12px; text-align: right; }
pre { margin: 0; font: 12px/1.45 var(--mono); white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow: auto;
  background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }
svg text { fill: var(--muted); font: 11px var(--sans); }
.chart-wrap { position: relative; }
.tip { position: absolute; pointer-events: none; background: var(--ink); color: var(--bg); font-size: 12px; padding: 6px 8px;
  border-radius: 6px; white-space: nowrap; transform: translate(-50%, -115%); display: none; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--muted); margin-top: 6px; }
.legend span { display: inline-flex; gap: 5px; align-items: center; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
td, th { padding: 5px 6px; border-bottom: 1px solid var(--line); text-align: left; }
th { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
td.num { font-family: var(--mono); font-size: 12px; text-align: right; }
tr.pick td { font-weight: 650; background: color-mix(in srgb, var(--accent) 9%, transparent); }
.kv { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 10px; }
.kv > div { background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
.kv .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
.kv .v { font-weight: 600; font-family: var(--mono); font-size: 13px; overflow: hidden; text-overflow: ellipsis; }
.why { font-size: 13px; margin-bottom: 8px; }
.rej { font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Routing</h1>
  <span class="live"><span class="dot" id="dot"></span><span id="live">live</span></span>
  <span class="stat" id="stat"></span>
  <div class="meters" id="meters"></div>
</header>
<main>
  <aside>
    <div class="panel compose">
      <textarea id="msg" placeholder="Write a task and route it"></textarea>
      <details><summary>Packet (optional JSON: kind, size, risk, failed_tiers...)</summary>
        <textarea id="packet" placeholder='{"kind": "debug", "size": "medium"}'></textarea>
      </details>
      <div class="row">
        <button class="primary" id="dry">Route only</button>
        <button id="send">Send</button>
        <label class="hint"><input type="checkbox" id="follow" checked> follow newest</label>
      </div>
      <div class="hint" style="margin-top:6px">Route only asks Jev and runs no model. Send runs the chosen model and spends its quota.</div>
      <div class="hint status-error" id="err"></div>
    </div>
    <div class="panel list" id="list"><div class="empty">No requests yet.</div></div>
  </aside>
  <section class="detail" id="detail"><div class="panel empty">Select a request, or route one.</div></section>
</main>
<script>
"use strict";
const $ = (s) => document.querySelector(s);
const traces = new Map();
let seq = 0, selected = null, health = null;

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") n.className = v; else if (k === "style") n.style.cssText = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
}
const svgEl = (tag, attrs) => { const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v); return n; };
const family = (t) => t && t.startsWith("claude") ? "claude" : t && t.startsWith("codex") ? "codex"
  : t && t.startsWith("local") ? "local" : "other";
const color = (t) => `var(--${family(t)})`;
const money = (v) => v == null ? "spent" : v === 0 ? "$0" : v < 0.01 ? `$${v.toPrecision(2)}` : `$${v.toFixed(3)}`;
const pct = (v) => v == null ? "–" : `${Math.round(v * 100)}%`;
const stageOf = (t, name) => t.stages.find((s) => s.name === name);
const tierChip = (t) => el("span", { class: "chip", title: t }, el("span", { class: "sw", style: `background:${color(t)}` }), t);

async function poll() {
  try {
    const r = await fetch(`/routing/traces?after=${seq}`);
    const data = await r.json();
    seq = data.seq;
    for (const t of data.traces) traces.set(t.id, t);
    if (data.traces.length) {
      if ($("#follow").checked || !selected) {
        const newest = [...traces.values()].reduce((a, b) => (!a || b.started_at > a.started_at ? b : a), null);
        if (newest) selected = newest.id;
      }
      renderList(); renderDetail();
    }
    $("#dot").classList.remove("off"); $("#live").textContent = "live";
  } catch (e) { $("#dot").classList.add("off"); $("#live").textContent = "disconnected"; }
  setTimeout(poll, 1000);
}

async function pollHealth() {
  try { health = await (await fetch("/healthz")).json(); renderHeader(); } catch (e) {}
  setTimeout(pollHealth, 5000);
}

function renderHeader() {
  const c = health.catalog || {};
  const served = (health.tiers || []).length;
  const bad = Object.keys(c.unavailable || {}).length;
  $("#stat").replaceChildren(...[
    el("span", {}, el("b", {}, served), " tiers"),
    c.discovered_tiers != null ? el("span", {}, " · ", el("b", {}, c.discovered_tiers), " discovered") : null,
    bad ? el("span", { class: "status-error", title: Object.entries(c.unavailable).map(([k, v]) => `${k}: ${v}`).join("\n") }, ` · ${bad} unavailable`) : null,
    health.router ? el("span", {}, " · router ", el("b", {}, health.router)) : null,
  ].filter(Boolean));
  const subs = (health.routing || {}).subscriptions || {};
  $("#meters").replaceChildren(...Object.entries(subs).map(([name, s]) => {
    return el("div", { class: "meter" },
      el("div", {}, el("b", { style: `color:var(--${name === "claude" || name === "codex" ? name : "accent"})` }, name), " ",
        s.locked ? el("span", { class: "status-error" }, "locked (429)") : `${money(s.used_usd)} at list price this window`));
  }));
}

function renderList() {
  const items = [...traces.values()].sort((a, b) => b.started_at - a.started_at);
  $("#list").replaceChildren(...items.map((t) => {
    const route = stageOf(t, "route");
    const time = new Date(t.started_at * 1000).toLocaleTimeString();
    return el("div", { class: "item" + (t.id === selected ? " sel" : ""), onclick: () => { selected = t.id; $("#follow").checked = false; renderList(); renderDetail(); } },
      el("div", { class: "p" }, (t.preview || "(no user message)").split("\n")[0]),
      el("div", { class: "m" },
        el("span", { class: `status-${t.status}` }, t.status === "running" ? "●" : t.status === "ok" ? "✓" : "✕"),
        el("span", {}, time),
        t.kind === "dry-run" ? el("span", { class: "chip" }, "route only") : null,
        route ? tierChip(route.tier) : null,
        t.total_ms != null ? el("span", {}, `${(t.total_ms / 1000).toFixed(1)}s`) : null));
  }));
}

function step(name, value, t, state) {
  return el("div", { class: `step ${state}` }, el("div", { class: "n" }, name), el("div", { class: "v", title: value || "" }, value || "—"),
    el("div", { class: "t" }, t != null ? `+${t} ms` : ""));
}

function renderDetail() {
  const t = traces.get(selected);
  if (!t) return;
  const elig = stageOf(t, "eligibility"), route = stageOf(t, "route"), call = stageOf(t, "call"),
    answer = stageOf(t, "answer"), done = stageOf(t, "done");
  const d = (route && route.detail) || {};
  const failed = t.status === "error";
  const pipeline = el("div", { class: "pipeline" },
    step("Received", t.requested_model === "auto" ? "auto" : t.requested_model, 0, "done"),
    step("Eligible", elig ? `${elig.eligible.length} of ${elig.eligible.length + elig.rejected.length}` : null, elig && elig.t_ms, elig ? (elig.eligible.length ? "done" : "fail") : "wait"),
    step("Jev", d.jev_ms != null ? `${d.jev_ms} ms` : (route ? (route.reason === "requested" ? "skipped: named" : "—") : null), route && route.t_ms, route ? "done" : "wait"),
    step("Decision", route ? route.tier : null, route && route.t_ms, route ? (d.rule === "fallback" ? "fail" : "done") : "wait"),
    step("Model", t.kind === "dry-run" ? "not run" : call ? `${call.model}${call.effort ? " @" + call.effort : ""}` : null, call && call.t_ms, t.kind === "dry-run" ? "wait" : call ? "done" : "wait"),
    step("Done", t.kind === "dry-run" ? "routed" : done ? `HTTP ${done.status}` : t.status, done ? done.t_ms : t.total_ms, failed ? "fail" : (done || t.kind === "dry-run") && t.status === "ok" ? "done" : "wait"));

  const head = el("div", { class: "panel card" },
    el("h2", {}, t.kind === "dry-run" ? "Routed (no model run)" : "Request"),
    el("div", { class: "prompt" }, t.preview || "(no user message)"), pipeline);

  const cards = [head];
  if (route) cards.push(decisionCard(route, d));
  if (d.needs && Object.keys(d.needs).length) cards.push(el("div", { class: "two" }, needsCard(d), packetCard(d)));
  if (d.options && d.options.length) cards.push(optionsCard(d));
  if (done || answer) cards.push(answerCard(answer, done));
  if (elig && elig.rejected.length) cards.push(el("details", { class: "panel card" },
    el("summary", { class: "hint" }, `${elig.rejected.length} tier(s) not eligible`),
    el("div", { class: "rej" }, ...elig.rejected.map((r) => el("div", {}, el("b", {}, r.tier), ` — ${r.reason}: ${r.detail}`)))));
  $("#detail").replaceChildren(...cards);
}

function decisionCard(route, d) {
  const pickOpt = (d.options || []).find((o) => o.tier === route.tier);
  const why = d.rule === "fallback" ? `Fallback: ${d.why}`
    : route.reason === "requested" ? "The client named this tier; Jev was not asked."
    : d.rule === "least expected cost" ? `Least expected cost: price + (1 − success) × the cost of failing. A failure here costs `
      + `${d.stakes}× a redo on ${d.redo_tier} (stakes from ${d.stakes_from}), `
      + (d.verifiable ? "and the client's tests would catch it." : "and nothing is known to catch it.")
    : d.rule && d.rule.startsWith("none") ? `No tier reached the target of ${pct(d.target)}; the most likely one was chosen.`
    : `The cheapest of ${(d.options || []).filter((o) => o.covered).length} tier(s) estimated to succeed at ${pct(d.target)} or more.`;
  return el("div", { class: "panel card" }, el("h2", {}, "Decision"),
    el("div", { class: "why" }, why),
    el("div", { class: "kv" },
      el("div", { style: "grid-column: span 2" }, el("div", { class: "k" }, "Tier"), el("div", { class: "v" }, tierChip(route.tier))),
      el("div", {}, el("div", { class: "k" }, "Est. success"), el("div", { class: "v" }, pct(route.score))),
      el("div", {}, el("div", { class: "k" }, "Est. cost"), el("div", { class: "v" }, pickOpt ? money(pickOpt.cost) : "–")),
      pickOpt && pickOpt.expected != null ? el("div", {}, el("div", { class: "k" }, "Expected cost"), el("div", { class: "v" }, money(pickOpt.expected))) : null,
      el("div", {}, el("div", { class: "k" }, "Routing time"), el("div", { class: "v" }, `${route.router_ms} ms`)),
      d.prompt_tokens != null ? el("div", {}, el("div", { class: "k" }, "Prompt"), el("div", { class: "v" }, `~${d.prompt_tokens} tok`)) : null),
    d.skipped_failed && d.skipped_failed.length ? el("div", { class: "hint" }, `Not offered again (already failed): ${d.skipped_failed.join(", ")}`
      + (d.failed_bar != null ? `; nothing rated below ${pct(d.failed_bar)} is offered` : "")) : null);
}

function needsCard(d) {
  const rows = Object.entries(d.needs).sort((a, b) => b[1] - a[1]);
  return el("div", { class: "panel card" }, el("h2", {}, "What Jev read in the task"),
    ...rows.map(([k, v]) => el("div", { class: "need" }, el("span", {}, k.replace("_", " ")),
      el("div", { class: "track", title: `p = ${v.toFixed(2)}; below ${d.floor} counts as not needed` },
        el("div", { class: "fill" + (v <= d.floor ? " low" : ""), style: `width:${v * 100}%` }),
        el("div", { class: "floor", style: `left:${d.floor * 100}%` })),
      el("span", { class: "val" }, v.toFixed(2)))),
    el("div", { class: "hint", style: "margin-top:8px" }, `The dark mark is Jev's "no" band (${d.floor}): at or below it a requirement counts as absent.`));
}

function packetCard(d) {
  return el("div", { class: "panel card" }, el("h2", {}, "Packet sent to Jev"), el("pre", {}, d.packet || "(none)"));
}

function optionsCard(d) {
  const opts = d.options.filter((o) => o.cost != null);
  // Under the expected-cost rule there is no fixed bar: every option is a
  // price and a risk, and the table ranks them by the two together.
  const byExpected = d.rule === "least expected cost";
  const W = 760, H = 300, L = 44, R = 12, T = 12, B = 34;
  const costs = opts.map((o) => Math.max(o.cost, 1e-6));
  const lo = Math.log10(Math.min(...costs)) - 0.15, hi = Math.log10(Math.max(...costs)) + 0.15;
  const x = (c) => L + (Math.log10(Math.max(c, 1e-6)) - lo) / (hi - lo || 1) * (W - L - R);
  const y = (p) => T + (1 - p) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Estimated success against cost for every tier" });
  for (const p of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y(p), y2: y(p), stroke: "var(--line)" }));
    const tx = svgEl("text", { x: L - 6, y: y(p) + 4, "text-anchor": "end" }); tx.textContent = pct(p); svg.append(tx);
  }
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) {
    svg.append(svgEl("line", { x1: x(10 ** e), x2: x(10 ** e), y1: T, y2: H - B, stroke: "var(--line)" }));
    const tx = svgEl("text", { x: x(10 ** e), y: H - B + 16, "text-anchor": "middle" }); tx.textContent = money(10 ** e); svg.append(tx);
  }
  const axis = svgEl("text", { x: (W + L) / 2, y: H - 4, "text-anchor": "middle" }); axis.textContent = "estimated cost per call (log) →"; svg.append(axis);
  if (!byExpected) {
    svg.append(svgEl("line", { x1: L, x2: W - R, y1: y(d.target), y2: y(d.target), stroke: "var(--ok)", "stroke-dasharray": "5 4" }));
    const tl = svgEl("text", { x: L + 6, y: y(d.target) - 5, "text-anchor": "start", style: "fill:var(--ok)" }); tl.textContent = `target ${pct(d.target)}`; svg.append(tl);
  }
  const tip = el("div", { class: "tip" });
  const wrap = el("div", { class: "chart-wrap" }, svg, tip);
  const pick = opts.find((o) => o.tier === d.pick);
  for (const o of [...opts].sort((a, b) => (a.tier === d.pick) - (b.tier === d.pick))) {
    const isPick = o.tier === d.pick;
    const c = svgEl("circle", { cx: x(o.cost), cy: y(o.success), r: isPick ? 8 : 5, fill: byExpected || o.covered ? color(o.tier) : "var(--panel)",
      stroke: color(o.tier), "stroke-width": isPick ? 3 : 1.5, style: "cursor:pointer" });
    c.addEventListener("mouseenter", () => { tip.style.display = "block";
      tip.textContent = `${o.tier} · ${pct(o.success)} · ${money(o.cost)}` + (o.expected != null ? ` · expected ${money(o.expected)}` : "");
      const box = svg.getBoundingClientRect(), s = box.width / W;
      tip.style.left = `${x(o.cost) * s}px`; tip.style.top = `${y(o.success) * s}px`; });
    c.addEventListener("mouseleave", () => { tip.style.display = "none"; });
    svg.append(c);
  }
  if (pick) {
    const right = x(pick.cost) > W * 0.62;
    const lbl = svgEl("text", { x: x(pick.cost) + (right ? -14 : 14), y: y(pick.success) + 22,
      "text-anchor": right ? "end" : "start", style: "fill:var(--ink);font-weight:600" });
    lbl.textContent = pick.tier; svg.append(lbl);
  }
  const legend = el("div", { class: "legend" },
    ...["claude", "codex", "local"].map((f) => el("span", {}, el("span", { class: "chip", style: `background:var(--${f});width:10px;height:10px;padding:0` }), f)),
    el("span", {}, byExpected ? "large ring = least expected cost" : "filled = reaches the target · hollow = does not · large ring = chosen"));
  // The table: the chosen tier and its neighbours in cost, cheapest first.
  const sorted = [...opts].sort((a, b) => a.cost - b.cost);
  const at = sorted.findIndex((o) => o.tier === d.pick);
  const shown = sorted.slice(Math.max(0, at - 6), at + 4);
  const table = el("table", {}, el("tr", {}, el("th", {}, "tier"), el("th", {}, "effort"), el("th", {}, "est. success"), el("th", {}, "est. cost"),
      byExpected ? el("th", {}, "expected") : null, el("th", {}, "")),
    ...shown.map((o) => el("tr", { class: o.tier === d.pick ? "pick" : "" }, el("td", {}, tierChip(o.tier)), el("td", {}, o.effort || "—"),
      el("td", { class: "num" }, pct(o.success)), el("td", { class: "num" }, money(o.cost)),
      byExpected ? el("td", { class: "num" }, o.expected != null ? money(o.expected) : "–") : null,
      byExpected ? el("td", { class: o.tier === d.pick ? "status-ok" : "hint" }, o.tier === d.pick ? "chosen" : "")
        : el("td", { class: o.covered ? "status-ok" : "hint" }, o.tier === d.pick ? "chosen" : o.covered ? "covers" : "below target"))));
  return el("div", { class: "panel card" }, el("h2", {}, `Every option weighed (${opts.length})`), wrap, legend,
    el("div", { class: "hint", style: "margin-top:10px" }, "Around the choice, by cost:"), table);
}

function answerCard(answer, done) {
  const u = (done && done.usage) || {};
  return el("div", { class: "panel card" }, el("h2", {}, "Answer"),
    done ? el("div", { class: "kv" },
      el("div", {}, el("div", { class: "k" }, "Status"), el("div", { class: "v " + (done.status < 400 ? "status-ok" : "status-error") }, done.status)),
      el("div", {}, el("div", { class: "k" }, "Served by"), el("div", { class: "v" }, tierChip(done.final_tier || ""))),
      el("div", {}, el("div", { class: "k" }, "Tokens in / out"), el("div", { class: "v" }, `${u.prompt_tokens || 0} / ${u.completion_tokens || 0}`)),
      el("div", {}, el("div", { class: "k" }, "Latency"), el("div", { class: "v" }, `${(done.t_ms / 1000).toFixed(1)} s`)),
      done.verdict ? el("div", {}, el("div", { class: "k" }, "Review"), el("div", { class: "v" }, done.verdict)) : null) : null,
    done && done.error ? el("pre", { class: "status-error" }, done.error) : null,
    answer ? el("pre", {}, answer.text) : null);
}

function readPacket() {
  const raw = $("#packet").value.trim();
  if (!raw) return undefined;
  try { return JSON.parse(raw); } catch (e) { throw new Error("The packet is not valid JSON."); }
}

async function submit(kind) {
  const text = $("#msg").value.trim();
  $("#err").textContent = "";
  if (!text) { $("#err").textContent = "Write a task first."; return; }
  let packet;
  try { packet = readPacket(); } catch (e) { $("#err").textContent = e.message; return; }
  const body = { model: "auto", messages: [{ role: "user", content: text }] };
  if (packet) body.packet = packet;
  $("#follow").checked = true; selected = null;
  for (const b of [$("#dry"), $("#send")]) b.disabled = true;
  try {
    const r = await fetch(kind === "dry" ? "/routing/dry-run" : "/v1/chat/completions",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { const e = await r.json().catch(() => ({})); $("#err").textContent = (e.error && e.error.message) || `HTTP ${r.status}`; }
  } catch (e) { $("#err").textContent = String(e); }
  finally { for (const b of [$("#dry"), $("#send")]) b.disabled = false; }
}

$("#dry").addEventListener("click", () => submit("dry"));
$("#send").addEventListener("click", () => submit("send"));
$("#msg").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit("dry"); });
poll(); pollHealth();
</script>
</body>
</html>
"""
