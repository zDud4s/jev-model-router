"""The live routing view served at /routing: one self-contained page, no assets.

It polls /routing/traces for requests that moved and /healthz for the catalog
and the subscriptions, and draws each request end to end: what arrived, which
tiers could take it, what Jev read in it, every option weighed, what answered.

The Tiers tab reads /routing/tiers (built at startup) and draws every carded
tier's levels: where each came from (benchmark, blended, profile), how far the
evidence moved it from the profile, and which tiers dominate it.

The Config tab reads /routing/config and edits the file the proxy was started
from: a form for the common fields and the tiers, a YAML view for the rest.
Save validates with the config parser and writes only what it accepts; Review
shows the diff first. A saved file is served from the next start.
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Routing · jev-model-router</title>
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
button:disabled { opacity: .5; cursor: not-allowed; }
button.busy { cursor: progress; }
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
.tabs { display: inline-flex; background: var(--soft); border: 1px solid var(--line); border-radius: 8px; padding: 2px; }
.tabs button { border: 0; background: transparent; padding: 4px 12px; font-weight: 600; color: var(--muted); border-radius: 6px; }
.tabs button.on { background: var(--panel); color: var(--ink); box-shadow: 0 1px 2px color-mix(in srgb, var(--ink) 15%, transparent); }
[hidden] { display: none !important; }
.tiers { padding: 16px 20px; display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.tiers > * { min-width: 0; }
.tt th.num { text-align: right; }
.strip { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 12px; color: var(--muted); align-items: baseline; }
.strip b { color: var(--ink); font-family: var(--mono); font-weight: 600; }
.strip .grp { display: inline-flex; gap: 6px; align-items: baseline; flex-wrap: wrap; min-width: 0; overflow-wrap: anywhere; }
.filters { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.filters input[type=search], .filters select { border: 1px solid var(--line); border-radius: 7px; background: var(--bg); color: var(--ink);
  padding: 5px 9px; font: 13px var(--sans); }
.filters input[type=search] { min-width: 0; flex: 1 1 200px; max-width: 360px; }
.table-wrap { overflow-x: auto; max-width: 100%; }
.tt { width: auto; min-width: 100%; margin-top: 0; }
.tt th { position: sticky; top: 0; background: var(--panel); cursor: pointer; user-select: none; white-space: nowrap; }
.tt th[aria-sort="ascending"]::after { content: " \2191"; }
.tt th[aria-sort="descending"]::after { content: " \2193"; }
.tt td { white-space: nowrap; vertical-align: top; background: var(--panel); }
.tt td.name, .tt th.name { position: sticky; left: 0; z-index: 1; }
.tt tr.dom td > * { opacity: .45; }
.tt tr.dom:hover td > * { opacity: .85; }
.tt tr:hover td { background: color-mix(in srgb, var(--soft) 70%, var(--panel)); }
.lvl { display: inline-block; min-width: 44px; padding: 1px 6px; border-radius: 5px; font: 12px var(--mono); text-align: right; border: 1px solid var(--line); }
.lvl.benchmark { background: color-mix(in srgb, var(--accent) 24%, var(--panel)); border-color: var(--accent); color: var(--ink); }
.lvl.blended { background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 24%, var(--panel)) 50%, var(--panel) 50%);
  border: 1px dashed var(--accent); color: var(--ink); }
.lvl.profile { color: var(--muted); background: transparent; }
.delta { font: 11px var(--mono); margin-left: 4px; color: var(--muted); display: inline-block; min-width: 40px; }
.delta.up { color: var(--ok); }
.delta.down { color: var(--bad); }
.capm { font-size: 10px; color: var(--warn); margin-left: 2px; }
.sub { font-size: 11px; color: var(--muted); }
.tt details summary { cursor: pointer; color: var(--warn); font-size: 12px; }
.tt details div { font-size: 12px; color: var(--muted); }
.key { display: inline-flex; gap: 6px; align-items: center; }
.cfg { padding: 16px 20px 90px; display: flex; flex-direction: column; gap: 16px; min-width: 0; max-width: 980px; margin: 0 auto; width: 100%; }
.sec h2 { font-size: 15px; text-transform: none; letter-spacing: 0; color: var(--ink); margin: 0; }
.sec .intro { color: var(--muted); font-size: 13px; margin: 4px 0 6px; max-width: 70ch; }
.sec-head { display: flex; gap: 12px; align-items: flex-start; justify-content: space-between; }
.frow { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 300px); gap: 6px 24px; align-items: center; padding: 10px 0;
  border-top: 1px solid var(--line); }
.frow .flabel label { font-weight: 600; font-size: 13px; }
.frow .help { color: var(--muted); font-size: 12px; margin-top: 2px; }
.frow .help code, .ykey { font: 11px var(--mono); color: var(--muted); opacity: .8; }
.frow input[type=text], .frow input[type=number], .frow select, .cfg textarea, .cfg input.name {
  width: 100%; border: 1px solid var(--line); border-radius: 7px; background: var(--bg); color: var(--ink);
  padding: 6px 9px; font: 13px var(--sans); }
.frow .pct { display: flex; gap: 10px; align-items: center; }
.frow .pct input[type=range] { flex: 1; accent-color: var(--accent); }
.frow .pct output { font: 600 13px var(--mono); min-width: 42px; text-align: right; }
.frow .dirty { border-color: var(--warn) !important; box-shadow: 0 0 0 2px color-mix(in srgb, var(--warn) 22%, transparent); border-radius: 7px; }
.switch { position: relative; display: inline-block; width: 38px; height: 22px; flex: none; }
.switch input { opacity: 0; width: 0; height: 0; }
.switch span { position: absolute; inset: 0; background: var(--line); border-radius: 11px; cursor: pointer; transition: background .15s; }
.switch span::before { content: ""; position: absolute; width: 16px; height: 16px; left: 3px; top: 3px; background: var(--panel); border-radius: 50%; transition: transform .15s; }
.switch input:checked + span { background: var(--accent); }
.switch input:checked + span::before { transform: translateX(16px); }
.switch input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
.switch.dirty span { box-shadow: 0 0 0 2px var(--warn); }
.checks { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 13px; padding: 2px 4px; }
details.adv { margin-top: 6px; border-top: 1px solid var(--line); }
details.adv > summary { cursor: pointer; padding: 10px 0 4px; color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
details.adv .frow:first-of-type { border-top: 0; }
.items { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }
.item-card { border: 1px solid var(--line); border-radius: 8px; background: var(--bg); min-width: 0; }
.item-card > summary { padding: 9px 12px; cursor: pointer; display: flex; gap: 10px; align-items: center; list-style: none; min-width: 0; }
.item-card > summary::-webkit-details-marker { display: none; }
.item-card > summary::before { content: "\25B8"; color: var(--muted); font-size: 11px; }
.item-card[open] > summary::before { content: "\25BE"; }
.item-card > summary .sub { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
.item-card > .body { padding: 0 12px 8px; }
.item-card.gone { opacity: .5; }
.item-card.gone > summary .chip { text-decoration: line-through; }
.item-card.added { border-color: var(--ok); }
.addrow { display: flex; gap: 8px; margin-top: 10px; align-items: center; flex-wrap: wrap; }
.picks { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 2px 16px; max-height: 360px; overflow: auto;
  border: 1px solid var(--line); border-radius: 8px; padding: 6px 8px; background: var(--panel); }
.mpick { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 2px 8px; align-items: baseline; padding: 4px 2px; font-size: 13px; cursor: pointer; }
.mpick .pid { font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
.mpick > .sub:not(.num) { grid-column: 2; }
.mpick .num { grid-row: 1; grid-column: 3; font-family: var(--mono); }
.mpick.off .pid { color: var(--muted); text-decoration: line-through; }
.offer { display: flex; gap: 12px; align-items: center; justify-content: space-between; width: 100%;
  border: 1px dashed var(--line); border-radius: 8px; padding: 8px 12px; }
@media (max-width: 640px) { .frow { grid-template-columns: 1fr; } }
.cfg textarea.yaml { min-height: 60vh; font: 12px/1.5 var(--mono); tab-size: 2; resize: vertical; }
.savebar { position: fixed; left: 0; right: 0; bottom: 0; background: var(--panel); border-top: 1px solid var(--line);
  padding: 10px 20px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; z-index: 3; }
.diff { max-height: 45vh; }
.diff .add { color: var(--ok); } .diff .del { color: var(--bad); } .diff .at { color: var(--accent); }
.banner { padding: 10px 14px; border-radius: 8px; border: 1px solid var(--line); background: var(--soft); font-size: 13px; }
.banner.warn { border-color: var(--warn); } .banner.bad { border-color: var(--bad); } .banner.good { border-color: var(--ok); }
</style>
</head>
<body>
<header>
  <h1>Routing</h1>
  <nav class="tabs" role="tablist" aria-label="View">
    <button role="tab" data-view="requests" class="on" aria-selected="true">Requests</button>
    <button role="tab" data-view="tiers" aria-selected="false">Tiers</button>
    <button role="tab" data-view="config" aria-selected="false">Config</button>
  </nav>
  <span class="live"><span class="dot" id="dot"></span><span id="live">live</span></span>
  <span class="stat" id="stat"></span>
  <div class="meters" id="meters"></div>
</header>
<main id="view-requests">
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
<section class="tiers" id="view-tiers" hidden>
  <div class="panel card"><h2>Cards and their evidence</h2><div class="strip" id="tstrip"><span>Loading…</span></div></div>
  <div class="panel card">
    <div class="filters">
      <input type="search" id="tq" placeholder="Filter tiers, models, families" aria-label="Filter tiers">
      <select id="teffort" aria-label="Effort"><option value="*">every effort</option></select>
      <label class="hint"><input type="checkbox" id="tdom"> hide dominated</label>
      <span class="hint" id="tcount"></span>
      <button id="treload" title="The view is built at startup; this re-reads it">Reload</button>
    </div>
    <div class="legend" style="margin:10px 0 4px">
      <span class="key"><span class="lvl benchmark">2.00</span> benchmark</span>
      <span class="key"><span class="lvl blended">2.00</span> blended with the profile</span>
      <span class="key"><span class="lvl profile">2.00</span> profile</span>
      <span class="key"><span class="delta up">+0.30</span> vs the profile</span>
      <span class="key"><span class="capm">▲cap</span> held at its level cap</span>
      <span class="key" style="opacity:.5">dimmed = dominated</span>
    </div>
    <div class="table-wrap" id="twrap"></div>
    <div class="hint status-error" id="terr"></div>
  </div>
</section>
<section class="cfg" id="view-config" hidden>
  <div class="panel card">
    <div class="sec-head">
      <div>
        <h2 style="margin:0;font-size:15px;text-transform:none;letter-spacing:0;color:var(--ink)">Settings</h2>
        <div class="hint" id="cpath"></div>
      </div>
      <div class="filters">
        <nav class="tabs" aria-label="Editor">
          <button data-mode="form" class="on">Settings</button>
          <button data-mode="yaml">Edit file</button>
        </nav>
        <button id="creload" title="Read the file again, dropping unsaved edits">Reload</button>
      </div>
    </div>
    <div id="cstate" style="margin-top:10px"></div>
  </div>
  <div id="cform" class="cfg" style="padding:0"></div>
  <div class="panel card" id="cyaml" hidden>
    <textarea class="yaml" id="ctext" spellcheck="false" aria-label="Config file"></textarea>
    <div class="hint" style="margin-top:6px">The whole file, comments included. Review shows what changes and whether the router would start with it.</div>
  </div>
  <div class="panel card" id="creview" hidden>
    <h2>Review</h2>
    <div id="cverdict"></div>
    <pre class="diff" id="cdiff" style="margin-top:10px"></pre>
  </div>
</section>
<div class="savebar" id="csave" hidden>
  <span id="ccount" class="hint"></span>
  <span class="hint status-error" id="cerr"></span>
  <span style="margin-left:auto"></span>
  <button id="cdiscard">Discard</button>
  <button id="ccheck">Review changes</button>
  <button class="primary" id="cwrite">Save</button>
</div>
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
  if ($("#cfound")) $("#cfound").textContent = foundText();
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
    step("Jev", d.jev_ms != null ? `${d.jev_ms} ms` : route && route.jev_ms != null ? `${route.jev_ms} ms` : (route ? (route.reason === "requested" ? "skipped: named" : "—") : null), route && route.t_ms, route ? (route.error ? "fail" : "done") : "wait"),
    step("Decision", route ? (route.error || route.tier) : null, route && route.t_ms, route ? (route.error || d.rule === "fallback" ? "fail" : "done") : "wait"),
    step("Model", t.kind === "dry-run" ? "not run" : call ? `${call.model}${call.effort ? " @" + call.effort : ""}` : null, call && call.t_ms, t.kind === "dry-run" ? "wait" : call ? "done" : "wait"),
    step("Done", t.kind === "dry-run" ? "routed" : done ? `HTTP ${done.status}` : t.status, done ? done.t_ms : t.total_ms, failed ? "fail" : (done || t.kind === "dry-run") && t.status === "ok" ? "done" : "wait"));

  const head = el("div", { class: "panel card" },
    el("h2", {}, t.kind === "dry-run" ? "Routed (no model run)" : "Request"),
    el("div", { class: "prompt" }, t.preview || "(no user message)"), pipeline);

  const cards = [head];
  if (route && route.error) cards.push(errorCard(route));
  else if (route) cards.push(decisionCard(route, d));
  if (d.needs && Object.keys(d.needs).length) cards.push(el("div", { class: "two" }, needsCard(d), packetCard(d)));
  if (d.options && d.options.length) cards.push(optionsCard(d));
  if (done || answer) cards.push(answerCard(answer, done));
  if (elig && elig.rejected.length) cards.push(el("details", { class: "panel card" },
    el("summary", { class: "hint" }, `${elig.rejected.length} tier(s) not eligible`),
    el("div", { class: "rej" }, ...elig.rejected.map((r) => el("div", {}, el("b", {}, r.tier), ` — ${r.reason}: ${r.detail}`)))));
  $("#detail").replaceChildren(...cards);
}

function errorCard(route) {
  return el("div", { class: "panel card" }, el("h2", {}, "Refused"), el("pre", { class: "status-error" }, route.error));
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
    const result = await r.json().catch(() => ({}));
    if (!r.ok) { $("#err").textContent = (result.error && result.error.message) || `HTTP ${r.status}`; }
    else if (result.error) { $("#err").textContent = result.error; }
  } catch (e) { $("#err").textContent = String(e); }
  finally { for (const b of [$("#dry"), $("#send")]) b.disabled = false; }
}

// ------------------------------------------------------------------ tiers
let tierData = null, tierSort = { key: "name", dir: 1 };
const fmt = (v, d = 2) => v == null ? "–" : Number(v).toFixed(d);

async function loadTiers() {
  $("#terr").textContent = "";
  try {
    const r = await fetch("/routing/tiers");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    tierData = await r.json();
    if (tierData.error) $("#terr").textContent = `The view failed at startup: ${tierData.error}`;
    fillEfforts(); renderStrip(); renderTiers();
  } catch (e) { $("#terr").textContent = `Could not load /routing/tiers: ${e.message || e}`; }
}

function effortOrder(e) {
  const order = ((tierData && tierData.global) || {}).efforts || [];
  return e == null ? -1 : order.includes(e) ? order.indexOf(e) : order.length;
}

function fillEfforts() {
  const sel = $("#teffort"), keep = sel.value;
  const efforts = [...new Set(tierData.tiers.map((t) => t.effort == null ? "" : t.effort))]
    .sort((a, b) => effortOrder(a || null) - effortOrder(b || null) || a.localeCompare(b));
  sel.replaceChildren(el("option", { value: "*" }, "every effort"),
    ...efforts.map((e) => el("option", { value: e }, e || "no effort")));
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "*";
}

function renderStrip() {
  const g = tierData.global;
  if (!g) { $("#tstrip").replaceChildren(el("span", {}, "No capabilities router: no cards to show.")); return; }
  const item = (label, ...v) => el("span", { class: "grp" }, label, ...v);
  const parts = [
    item("cards", el("b", {}, tierData.tiers.length)),
    item("dominated", el("b", {}, tierData.tiers.filter((t) => t.dominated_by.length).length)),
    g.line ? item("line", el("b", {}, `a ${fmt(g.line.a)}`), el("b", {}, `k ${fmt(g.line.k)}`),
      el("b", { title: "profile_weight: the evidence C at which a level is half measurement, half profile" }, `profile_weight ${fmt(g.line.profile_weight)}`))
      : item("line", el("b", {}, "none"), "(no benchmark evidence: every card is its profile)"),
    item("miss_scale", el("b", {}, fmt(g.miss_scale))),
    g.rule === "expected_cost" ? item("rule", el("b", {}, "expected cost")) : item("target", el("b", {}, pct(g.target))),
  ];
  const ev = g.evidence;
  if (ev) {
    parts.push(item("imported", el("b", {}, ev.imported_at ? new Date(ev.imported_at).toLocaleString() : "never")));
    parts.push(item("evidence", el("b", {}, ev.benchmarks_count), "benchmarks ·", el("b", {}, ev.points), "points"));
    for (const s of ev.sources) {
      const bad = /fail|skip|error/i.test(s.status);
      parts.push(item(`source ${s.name}`, el("b", { class: bad ? "status-error" : "", title: `${s.origin}: ${s.status}` },
        s.status.length > 60 ? s.status.slice(0, 57) + "…" : s.status)));
    }
    for (const [k, names] of Object.entries(ev.benchmarks)) {
      if (names.length) parts.push(item(`${k}:`, el("b", { class: k === "thin" ? "" : "status-error", title: names.join("\n") }, names.join(", "))));
    }
  }
  if (g.dominance_error) parts.push(item("dominance", el("b", { class: "status-error" }, g.dominance_error)));
  $("#tstrip").replaceChildren(...parts);
}

function sortValue(t, key) {
  if (key === "name") return t.name;
  if (key === "effort") return effortOrder(t.effort);
  if (key === "price") return t.prices.output;
  if (key === "output") return t.output_tokens;
  if (key === "beaten") return t.dominated_by.length;
  const c = t.levels[key.slice(2)];
  return c ? c.level : -1;
}

function renderTiers() {
  if (!tierData) return;
  const reqs = (tierData.global || {}).requirements || [];
  const q = $("#tq").value.trim().toLowerCase(), eff = $("#teffort").value, hide = $("#tdom").checked;
  const rows = tierData.tiers.filter((t) =>
    (!q || [t.name, t.model, t.family, t.backend].some((v) => v && v.toLowerCase().includes(q)))
    && (eff === "*" || (t.effort == null ? "" : t.effort) === eff)
    && !(hide && t.dominated_by.length));
  const { key, dir } = tierSort;
  rows.sort((a, b) => { const x = sortValue(a, key), y = sortValue(b, key);
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir || a.name.localeCompare(b.name); });
  $("#tcount").textContent = `${rows.length} of ${tierData.tiers.length}`;
  const th = (k, label, cls) => el("th", { class: cls || "", scope: "col",
    "aria-sort": key === k ? (dir > 0 ? "ascending" : "descending") : "none",
    onclick: () => { tierSort = { key: k, dir: tierSort.key === k ? -tierSort.dir : (k === "name" || k === "effort" ? 1 : -1) }; renderTiers(); } }, label);
  const head = el("tr", {}, th("name", "tier", "name"), th("effort", "effort"), th("price", "$/M in · out", "num"), th("output", "out tok", "num"),
    ...reqs.map((r) => th("r:" + r, r.replace(/_/g, " "))), th("beaten", "dominated"));
  const body = rows.map((t) => {
    const beaten = t.dominated_by.length;
    return el("tr", { class: beaten ? "dom" : "", title: beaten ? `beaten by ${t.dominated_by.join(", ")}` : "" },
      el("td", { class: "name" }, tierChip(t.name),
        el("div", { class: "sub" }, [t.model, t.family ? `family ${t.family}` : null, t.card,
          t.prices_from ? `prices ${t.prices_from}` : null].filter(Boolean).join(" · "))),
      el("td", {}, t.effort || "—"),
      el("td", { class: "num" }, `${fmt(t.prices.input)} · ${fmt(t.prices.output)}`),
      el("td", { class: "num" }, t.output_tokens),
      ...reqs.map((r) => levelCell(t.levels[r])),
      el("td", {}, beaten ? el("details", {}, el("summary", {}, `beaten by ${beaten}`),
        ...t.dominated_by.map((d) => el("div", {}, d))) : el("span", { class: "hint" }, "—")));
  });
  $("#twrap").replaceChildren(rows.length ? el("table", { class: "tt" }, el("thead", {}, head), el("tbody", {}, ...body))
    : el("div", { class: "empty" }, tierData.tiers.length ? "No tier matches the filters." : "No carded tiers."));
}

function levelCell(c) {
  if (!c) return el("td", {}, "–");
  const d = c.level - c.profile_level;
  const tip = [c.source, `C = ${fmt(c.evidence)}`, `profile ${fmt(c.profile_level)}`,
    c.cap != null ? `cap ${fmt(c.cap)}${c.capped ? " (held there)" : ""}` : null].filter(Boolean).join(" · ");
  return el("td", { title: tip },
    el("span", { class: `lvl ${c.source}` }, fmt(c.level)),
    el("span", { class: "delta" + (d > 0.005 ? " up" : d < -0.005 ? " down" : "") },
      Math.abs(d) < 0.005 ? "" : `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(2)}`),
    c.capped ? el("span", { class: "capm" }, "▲cap") : null);
}

// ------------------------------------------------------------------ config
// Settings grouped by what they are for. Each edit is a change against the
// file as it was read; the server applies the changes to the text (comments
// kept) and validates the result before anything is written.
let cfg = null, cmode = "form";
const edits = new Map();            // path key -> {path, value} | {path, delete: true}
const removals = new Map();         // path key -> path of a tier or source to delete
const added = new Map();            // new tier name -> its fields
const addedSources = new Map();     // new discover source name -> its entry
let catalogTiers = null;            // every carded tier at this start, from /routing/tiers
const opened = new Set();           // advanced panels and item cards left open
const pkey = (p) => p.join("\u0000");
const getIn = (o, p) => p.reduce((a, k) => (a && typeof a === "object" ? a[k] : undefined), o);
const edited = (path, fallback) => { const c = edits.get(pkey(path)); return c ? (c.delete ? undefined : c.value) : fallback; };

const BACKEND_LABELS = { ollama: "Ollama, on this machine", openai_compatible: "An OpenAI-compatible API", jev: "Jev judge",
  claude_cli: "Claude Code CLI (subscription)", codex_cli: "Codex CLI (subscription)" };
const KIND_LABELS = { static: "Always the default model", classifier: "A trained classifier",
  capabilities: "Jev reads each task and picks a model" };
const RULE_LABELS = { target: "The cheapest model likely to succeed", expected_cost: "Weigh price against the cost of failing" };
const POLICY_LABELS = { accept: "Keep the answer", escalate: "Ask a stronger model" };
const labelled = (values, labels) => values.map((v) => [v, labels[v] || v]);

async function loadConfig() {
  $("#cerr").textContent = "";
  try {
    cfg = await (await fetch("/routing/config")).json();
  } catch (e) {
    $("#cstate").replaceChildren(el("div", { class: "banner bad" }, `Could not load the settings: ${e.message || e}`));
    return;
  }
  edits.clear(); removals.clear(); added.clear(); addedSources.clear(); offers = null;
  $("#creview").hidden = true;
  if (!catalogTiers) fetch("/routing/tiers").then((r) => r.json())
    .then((t) => { catalogTiers = t.tiers || []; if (cfg) renderForm(); }).catch(() => { catalogTiers = []; });
  $("#ctext").value = cfg.text || "";
  renderConfigState(); renderForm(); updateSaveBar();
}

function renderConfigState() {
  $("#cpath").textContent = cfg.path ? `From ${cfg.path}` : "";
  const parts = [];
  if (!cfg.path) parts.push(el("div", { class: "banner warn" }, "This proxy was not started from a config file (",
    el("code", {}, "jev-model-router -c FILE serve"), "), so there is nothing to edit here."));
  else if (cfg.error) parts.push(el("div", { class: "banner bad" }, `The file cannot be read: ${cfg.error}. Fix it under Edit file.`));
  else {
    const c = cfg.check || {};
    if (!c.ok) parts.push(el("div", { class: "banner bad" }, `The router would not start with this file: ${c.error}`));
    else if (!cfg.served) parts.push(el("div", { class: "banner warn" }, "Saved changes are not in use yet: restart ",
      el("code", {}, "jev-model-router serve"), " to apply them."));
  }
  $("#cstate").replaceChildren(...parts);
  $("#cstate").hidden = !parts.length;
  for (const b of document.querySelectorAll("#view-config .tabs button")) b.disabled = !cfg.path;
  // A file the form cannot read as data is only editable as text.
  if (cfg.path && (cfg.error || !cfg.data || typeof cfg.data !== "object")) setMode("yaml");
}

// One setting: what it is called, what it does, and the key it writes.
function field(path, kind, label, help, opt) {
  opt = opt || {};
  const key = pkey(path), orig = getIn(cfg.data, path);
  const cur = edited(path, orig);
  let input, holder;
  const mark = () => (holder || input).classList.toggle("dirty", edits.has(key));
  const commit = (value, empty) => {
    if (empty ? orig === undefined : JSON.stringify(value) === JSON.stringify(orig)) edits.delete(key);
    else edits.set(key, empty ? { path, delete: true } : { path, value });
    mark(); updateSaveBar();
    if (opt.rerender) renderForm();
  };
  if (kind === "bool") {
    input = el("input", { type: "checkbox", role: "switch" });
    input.checked = cur === undefined ? !!opt.def : !!cur;
    input.addEventListener("change", () => commit(input.checked, false));
    holder = el("label", { class: "switch" }, input, el("span", {}));
  } else if (kind === "select") {
    const choices = (opt.choices || []).map((c) => (Array.isArray(c) ? c : [c, c]));
    input = el("select", {}, el("option", { value: "" }, opt.none || "(not set)"),
      ...choices.map(([v, l]) => el("option", { value: v }, l)));
    if (cur != null && !choices.some(([v]) => v === cur)) input.append(el("option", { value: cur }, cur));
    input.value = cur == null ? "" : cur;
    input.addEventListener("change", () => commit(input.value, input.value === ""));
  } else if (kind === "percent") {
    const lo = opt.min != null ? opt.min : 0, def = opt.def != null ? opt.def : lo;
    input = el("input", { type: "range", min: lo, max: 100, step: 1 });
    const out = el("output", {});
    const show = () => { out.textContent = `${input.value}%`; };
    input.value = Math.round((cur == null ? def : cur) * 100); show();
    input.addEventListener("input", () => { show(); commit(Number(input.value) / 100, false); });
    holder = el("div", { class: "pct" }, input, out);
  } else if (kind === "multi") {
    const chosen = new Set(Array.isArray(cur) ? cur : []);
    input = el("div", { class: "checks" }, ...(opt.choices || []).map((c) => {
      const box = el("input", { type: "checkbox", value: c });
      box.checked = chosen.has(c);
      box.addEventListener("change", () => {
        const now = [...input.querySelectorAll("input:checked")].map((b) => b.value);
        commit(now, now.length === 0 && orig === undefined);
      });
      return el("label", {}, box, " ", c);
    }));
  } else if (kind === "list") {
    input = el("input", { type: "text", placeholder: opt.placeholder || "none" });
    input.value = Array.isArray(cur) ? cur.join(", ") : cur == null ? "" : String(cur);
    input.addEventListener("input", () => {
      const items = input.value.split(",").map((v) => v.trim()).filter(Boolean);
      commit(items, items.length === 0);
    });
  } else {
    input = el("input", { type: kind === "number" ? "number" : "text", step: opt.step || "any",
      placeholder: opt.placeholder || (opt.def != null ? `${opt.def} (default)` : "not set") });
    input.value = cur == null ? "" : cur;
    input.addEventListener("input", () => {
      const raw = input.value.trim();
      if (raw === "") return commit(null, true);
      commit(kind === "number" ? Number(raw) : raw, false);
    });
  }
  mark();
  input.id = "f_" + key.replace(/[^A-Za-z0-9]/g, "_");
  return el("div", { class: "frow" },
    el("div", { class: "flabel" }, el("label", { for: input.id }, label),
      el("div", { class: "help" }, help ? help + " " : "", el("code", {}, path.join(".")))),
    el("div", { class: "fctl" }, holder || input));
}

function advanced(id, ...rows) {
  rows = rows.flat().filter(Boolean);
  if (!rows.length) return null;
  const d = el("details", { class: "adv" }, el("summary", {}, "Advanced"), ...rows);
  d.open = opened.has(id);
  d.addEventListener("toggle", () => { if (d.open) opened.add(id); else opened.delete(id); });
  return d;
}

function section(title, intro, head, ...rows) {
  return el("section", { class: "panel card sec" },
    el("div", { class: "sec-head" }, el("div", {}, el("h2", {}, title), intro ? el("div", { class: "intro" }, intro) : null), head || null),
    ...rows.flat().filter(Boolean));
}

function renderForm() {
  const box = $("#cform");
  if (!cfg.path || cfg.error || !cfg.data || typeof cfg.data !== "object") { box.replaceChildren(); return; }
  const d = cfg.data, o = cfg.options;
  const inFile = Object.keys(d.tiers || {}).concat([...added.keys()]).filter((t) => !removals.has(pkey(["tiers", t])));
  const router = d.router || {}, kind = edited(["router", "kind"], router.kind) || "static";
  const backendOf = (t) => (added.get(t) || (d.tiers || {})[t] || {}).backend;
  const answering = inFile.filter((t) => backendOf(t) !== "jev");
  const caps = kind === "capabilities" && router.capabilities && typeof router.capabilities === "object" ? router.capabilities : null;
  box.replaceChildren(...[
    choosingSection(kind, caps, inFile, answering, o),
    caps ? sourcesSection(caps) : null,
    tiersSection(inFile, o, kind),
    reviewSection(inFile, o),
    serverSection(),
  ].filter(Boolean));
}

function choosingSection(kind, caps, tiers, answering, o) {
  const C = ["router", "capabilities"];
  const rows = [
    field(["router", "kind"], "select", "How a model is chosen", null,
      { choices: labelled(o.router_kinds, KIND_LABELS), none: `(default) ${KIND_LABELS.static}`, rerender: true }),
    field(["router", "default_tier"], "select", kind === "capabilities" ? "Fallback model" : "Default model",
      kind === "capabilities" ? "Not where requests go: Jev picks among every model below for each one. This answers only when Jev cannot: unreachable, every subscription locked, or nothing else fits. It must be a model written in the file."
        : kind === "classifier" ? "The cheaper model tried first; the classifier sends harder tasks to the stronger one."
        : "Serves every request that does not name a model.", { choices: answering }),
  ];
  let adv = [];
  if (kind === "classifier") {
    rows.push(field(["router", "strong_tier"], "select", "Stronger model", "Where the classifier sends tasks it expects the default to fail.", { choices: answering }),
      field(["router", "model_path"], "text", "Classifier file", "Written by `train`."));
    adv = [field(["router", "threshold"], "number", "Threshold", "Above it, a task goes to the stronger model.", { placeholder: "the classifier's own" }),
      field(["router", "explore_rate"], "number", "Explore rate", "Share of would-be escalations still answered by the default, so they get reviewed.", { def: 0 })];
  }
  if (kind === "capabilities" && caps) {
    const rule = edited([...C, "rule"], caps.rule) || "target";
    rows.push(field([...C, "rule"], "select", "What counts as the best model", null,
      { choices: labelled(o.rules, RULE_LABELS), none: `(default) ${RULE_LABELS.target}`, rerender: true }));
    if (rule === "target") rows.push(field([...C, "target"], "percent", "Required chance of success",
      "The cheapest model estimated to succeed at least this often is chosen. Higher picks stronger, dearer models.", { min: 50, def: 0.8 }));
    rows.push(field([...C, "max_effort"], "select", "Highest reasoning effort",
      "Efforts above it are never offered. Higher efforts are slower and use more quota.", { choices: o.efforts, none: "(no limit)" }));
    adv = [field([...C, "jev_tier"], "select", "Judge that reads each task",
      "Reads every request and says how much it needs each requirement (reasoning, several files, debugging...); the model is chosen from that. One cheap call per request. Must be a model with backend jev.", { choices: tiers }),
      rule === "target" ? null : field([...C, "target"], "percent", "Target for the redo estimate", "Picks the model a failed task would be redone on.", { min: 50, def: 0.8 }),
      field([...C, "floor"], "number", "Floor", "A requirement the judge rates at or below this counts as absent.", { def: 0.2 }),
      field([...C, "miss_scale"], "number", "Miss scale", "Written by `calibrate`. Above 1 makes every model look less likely to succeed.", { def: 1.0 })];
  } else if (kind === "capabilities") {
    rows.push(el("div", { class: "banner warn", style: "margin-top:8px" },
      "This router needs a capabilities block (judge, requirements, sources). Add it under Edit file; config.example.yaml has one."));
  }
  return section("Choosing a model", "Which model answers a request that does not name one.", null, rows,
    advanced("choose", adv, caps ? el("div", { class: "hint", style: "padding:8px 0" },
      "Requirements, model profiles and benchmarks are under Edit file.") : null));
}

const foundText = () => health && health.catalog && health.catalog.discovered_tiers != null
  ? ` ${health.catalog.discovered_tiers} found at this start.` : "";
let offers = null;                  // providers the file could add, from /routing/config/sources
const pickerFilter = new Map();     // source -> filter text
const seenSources = new Set();      // sources whose card has been shown once

function sourcesSection(caps) {
  const discover = caps.discover || {};
  const cards = Object.entries(discover).map(([name, s]) => {
    const p = ["router", "capabilities", "discover", name];
    // Open at first sight: the model list is what this section is for.
    if (!seenSources.has(name)) { seenSources.add(name); opened.add(pkey(p)); }
    const gone = removals.has(pkey(p));
    s = s && typeof s === "object" ? s : {};
    return itemCard(p, name, BACKEND_LABELS[s.backend] || s.backend || "", gone, () => [
      modelPicker(name, p),
      s.backend === "openai_compatible" ? field([...p, "require_evidence"], "bool", "Only models with evidence",
        "Skip models no profile or benchmark speaks for. With it off, every model the source lists is offered.") : null,
      advanced("src:" + name,
        field([...p, "exclude"], "list", "Leave out models matching", "Comma-separated patterns, e.g. *-mini, old-*. Unticking a model above adds it here.", { placeholder: "nothing left out" }),
        field([...p, "include"], "list", "Only models matching", "Empty: every model the source serves.", { placeholder: "every model" }),
        field([...p, "base_url"], "text", "Where it runs", "A URL, or the CLI executable."),
        field([...p, "context_window"], "number", "Context window", "Tokens, for every model from this source.", { step: 1 }),
        field([...p, "timeout_s"], "number", "Timeout (s)", null, { def: 600 })),
    ]);
  });
  for (const [name, entry] of addedSources) {
    const remove = el("button", { style: "margin-left:auto", onclick: (e) => { e.preventDefault(); addedSources.delete(name); renderForm(); updateSaveBar(); } }, "Remove");
    cards.push(el("details", { class: "item-card added" }, el("summary", {}, tierChip(name),
      el("span", { class: "sub" }, `${BACKEND_LABELS[entry.backend] || entry.backend} · new: its models appear after the next start`), remove)));
  }
  return section("Models Jev can choose from",
    el("span", {}, "Every model these sources serve is an option, at every effort it takes. Untick a model to keep it out of routing.",
      el("span", { id: "cfound" }, foundText())), null,
    el("div", { class: "items" }, ...cards),
    addSourceRow());
}

// The models a source served at this start, from /routing/tiers, and the ones
// its `exclude` list names exactly (those were not discovered, so not listed there).
function modelPicker(source, p) {
  const excludePath = [...p, "exclude"];
  const exclude = edited(excludePath, getIn(cfg.data, excludePath)) || [];
  const exact = new Set(exclude.filter((g) => !/[*?[\]]/.test(g)));
  const models = new Map();
  for (const t of catalogTiers || []) {
    if (!t.name.startsWith(source + ":")) continue;
    const m = models.get(t.model) || { id: t.model, efforts: [], beaten: 0, price: t.prices.output };
    m.efforts.push(t.effort); if (t.dominated_by.length) m.beaten++;
    models.set(t.model, m);
  }
  for (const id of exact) if (!models.has(id)) models.set(id, { id, efforts: [], beaten: 0, price: null, out: true });
  if (!models.size) return el("div", { class: "hint", style: "padding:8px 0" },
    catalogTiers ? "No model from this source at this start." : "Loading the models…");
  const set = (next) => {
    const keep = exclude.filter((g) => /[*?[\]]/.test(g));
    const value = [...keep, ...[...next].sort()];
    const orig = getIn(cfg.data, excludePath);
    const key = pkey(excludePath);
    if (JSON.stringify(value) === JSON.stringify(orig || []) ) edits.delete(key);
    else edits.set(key, value.length || orig !== undefined ? { path: excludePath, value } : { path: excludePath, delete: true });
    renderForm(); updateSaveBar();
  };
  const q = (pickerFilter.get(source) || "").toLowerCase();
  const rows = [...models.values()].sort((a, b) => a.id.localeCompare(b.id)).filter((m) => !q || m.id.toLowerCase().includes(q));
  const on = [...models.values()].filter((m) => !exact.has(m.id)).length;
  const search = el("input", { type: "search", class: "name", placeholder: "Filter models", "aria-label": `Filter ${source} models`, style: "max-width:240px" });
  search.value = pickerFilter.get(source) || "";
  search.addEventListener("input", () => {
    pickerFilter.set(source, search.value);
    const at = search.selectionStart;
    renderForm();
    const again = document.querySelector(`input[aria-label="Filter ${source} models"]`);
    if (again) { again.focus(); again.setSelectionRange(at, at); }
  });
  const shown = rows.map((m) => m.id);
  const list = el("div", { class: "picks" }, ...rows.map((m) => {
    const box = el("input", { type: "checkbox" });
    box.checked = !exact.has(m.id);
    box.addEventListener("change", () => { const next = new Set(exact); if (box.checked) next.delete(m.id); else next.add(m.id); set(next); });
    const order = cfg.options.efforts;
    const efforts = m.efforts.filter(Boolean).sort((a, b) => order.indexOf(a) - order.indexOf(b));
    const note = m.out ? "left out" : m.beaten && m.beaten === m.efforts.length ? "never chosen: another model is as good for less"
      : efforts.length ? `efforts ${efforts.join(", ")}` : "";
    return el("label", { class: "mpick" + (box.checked ? "" : " off") }, box,
      el("span", { class: "pid" }, m.id), el("span", { class: "sub" }, note),
      m.price != null ? el("span", { class: "sub num" }, `$${fmt(m.price)}/M out`) : null);
  }));
  return el("div", { class: "picker" },
    el("div", { class: "addrow", style: "margin:4px 0 8px" },
      el("b", {}, `${on} of ${models.size} in routing`), el("span", { style: "margin-left:auto" }), search,
      el("button", { onclick: () => { const next = new Set(exact); shown.forEach((id) => next.delete(id)); set(next); } }, "Tick shown"),
      el("button", { onclick: () => { const next = new Set(exact); shown.forEach((id) => next.add(id)); set(next); } }, "Untick shown")),
    list,
    el("div", { class: "hint", style: "margin-top:6px" }, "Takes effect at the next start. A model left out stays out when the source lists it again."));
}

function addSourceRow() {
  const box = el("div", { class: "addrow" });
  const show = () => {
    const free = (offers || []).filter((o) => !addedSources.has(o.name));
    if (!offers) { box.replaceChildren(el("button", { onclick: load }, "Add a source")); return; }
    if (!free.length) { box.replaceChildren(el("span", { class: "hint" }, "Every source init knows is already here. Others are added under Edit file.")); return; }
    box.replaceChildren(...free.map((o) => el("div", { class: "offer" },
      el("div", {}, el("b", {}, o.label), el("div", { class: "hint" }, o.ready ? (o.note || "ready") : o.note)),
      el("button", { onclick: () => { addedSources.set(o.name, o.entry); renderForm(); updateSaveBar(); } }, "Add"))));
  };
  const load = async () => {
    box.replaceChildren(el("span", { class: "hint" }, "Looking for sources…"));
    try {
      const r = await fetch("/routing/config/sources");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      offers = (await r.json()).sources;
    } catch (e) { offers = null; $("#cerr").textContent = `Could not list sources: ${e.message || e}`; }
    show();
  };
  show();
  return box;
}

function tiersSection(tiers, o, kind) {
  const d = cfg.data;
  const cards = tiers.map((name) => added.has(name) ? newTierCard(name, o)
    : itemCard(["tiers", name], name, tierSummary((d.tiers || {})[name]), false, () => tierFields(name, o)));
  for (const [k, p] of removals) if (p[0] === "tiers") cards.push(itemCard(p, p[1], tierSummary((d.tiers || {})[p[1]]), true, () => []));
  const nameIn = el("input", { class: "name", placeholder: "name", "aria-label": "New model name", style: "max-width:220px" });
  const add = el("button", { onclick: () => {
    const name = nameIn.value.trim();
    $("#cerr").textContent = "";
    if (!name) { $("#cerr").textContent = "Name the model first."; return; }
    if ((d.tiers || {})[name] || added.has(name)) { $("#cerr").textContent = `${name} is already in the file.`; return; }
    added.set(name, { backend: "openai_compatible", model: "" });
    renderForm(); updateSaveBar();
  } }, "Add");
  return section(kind === "capabilities" ? "Models written in the file" : "Models",
    kind === "capabilities" ? "Beside the discovered ones: the judge, and anything discovery cannot find." : "The models requests can be sent to.", null,
    el("div", { class: "items" }, ...cards),
    el("div", { class: "addrow" }, nameIn, add));
}

const tierSummary = (t) => t && typeof t === "object"
  ? [BACKEND_LABELS[t.backend] || t.backend, t.model, t.effort ? `effort ${t.effort}` : null].filter(Boolean).join(" · ") : "";

function itemCard(path, name, summaryText, gone, build) {
  const key = pkey(path);
  const toggle = el("button", { style: "margin-left:auto", onclick: (e) => {
    e.preventDefault();
    if (gone) removals.delete(key); else removals.set(key, path);
    renderForm(); updateSaveBar();
  } }, gone ? "Keep" : "Remove");
  const card = el("details", { class: "item-card" + (gone ? " gone" : "") },
    el("summary", {}, tierChip(name), el("span", { class: "sub" }, summaryText), toggle));
  if (gone) return card;
  // Fields are built when the card opens, so a long file stays quick.
  const fill = () => {
    if (card.open) opened.add(key); else opened.delete(key);
    if (!card.open || card.querySelector(".body")) return;
    card.append(el("div", { class: "body" }, ...build().filter(Boolean)));
  };
  card.addEventListener("toggle", fill);
  if (opened.has(key) || [...edits.keys()].some((k) => k.startsWith(key + "\u0000"))) { card.open = true; fill(); }
  return card;
}

function tierFields(name, o) {
  const p = ["tiers", name], t = (cfg.data.tiers || {})[name] || {};
  const backend = edited([...p, "backend"], t.backend);
  return [
    field([...p, "backend"], "select", "Runs on", null, { choices: labelled(o.backends, BACKEND_LABELS), rerender: true }),
    field([...p, "model"], "text", "Model id", "As the provider names it."),
    o.subscription_backends.includes(backend) ? field([...p, "effort"], "select", "Reasoning effort", null, { choices: o.efforts }) : null,
    field([...p, "base_url"], "text", "Address", "The API's URL, or the CLI executable.", { placeholder: "the default for this kind" }),
    backend === "openai_compatible" || backend === "jev"
      ? field([...p, "api_key_env"], "text", "Key variable", "The environment variable holding the API key; the file never holds the key.") : null,
    field([...p, "context_window"], "number", "Context window", "Tokens the model can read.", { def: 8192, step: 1 }),
    field([...p, "supports_tools"], "bool", "Can use tools", "Requests that send tools only go to models that can."),
    backend === "jev" ? field([...p, "jev", "threshold"], "percent", "Pass mark", "A review passes at or above this probability.", { def: 0.5 }) : null,
    advanced("tier:" + name,
      field([...p, "prices", "input"], "number", "Input price", "USD per million tokens.", { def: 0 }),
      field([...p, "prices", "output"], "number", "Output price", "USD per million tokens.", { def: 0 }),
      field([...p, "prices", "cache_read"], "number", "Cached input price", "USD per million tokens.", { placeholder: "the input price" }),
      field([...p, "prices", "cache_write"], "number", "Cache write price", "USD per million tokens.", { def: 0 }),
      field([...p, "timeout_s"], "number", "Timeout (s)", null, { def: 600 })),
  ];
}

function newTierCard(name, o) {
  const t = added.get(name);
  const put = (k, v) => { if (v === "" || v == null) delete t[k]; else t[k] = v; updateSaveBar(); };
  const row = (label, help, control) => el("div", { class: "frow" },
    el("div", { class: "flabel" }, el("label", {}, label), help ? el("div", { class: "help" }, help) : null), el("div", {}, control));
  const text = (k, num, ph) => {
    const i = el("input", { type: num ? "number" : "text", placeholder: ph || "not set" });
    i.value = t[k] == null ? "" : t[k];
    i.addEventListener("input", () => put(k, i.value.trim() === "" ? null : num ? Number(i.value) : i.value.trim()));
    return i;
  };
  const backend = el("select", {}, ...o.backends.map((b) => el("option", { value: b }, BACKEND_LABELS[b] || b)));
  backend.value = t.backend;
  backend.addEventListener("change", () => {
    put("backend", backend.value);
    if (!o.subscription_backends.includes(backend.value)) delete t.effort;
    renderForm();
  });
  const tools = el("input", { type: "checkbox", role: "switch" });
  tools.checked = !!t.supports_tools;
  tools.addEventListener("change", () => put("supports_tools", tools.checked || null));
  const rows = [row("Runs on", null, backend), row("Model id", "As the provider names it.", text("model")),
    row("Address", "The API's URL, or the CLI executable.", text("base_url", false, "the default for this kind"))];
  if (o.subscription_backends.includes(t.backend)) {
    const effort = el("select", {}, el("option", { value: "" }, "(not set)"), ...o.efforts.map((e) => el("option", { value: e }, e)));
    effort.value = t.effort || "";
    effort.addEventListener("change", () => put("effort", effort.value));
    rows.push(row("Reasoning effort", null, effort));
  } else rows.push(row("Key variable", "The environment variable holding the API key.", text("api_key_env")));
  rows.push(row("Context window", "Tokens the model can read.", text("context_window", true, "8192 (default)")),
    row("Can use tools", null, el("label", { class: "switch" }, tools, el("span", {}))));
  const remove = el("button", { style: "margin-left:auto", onclick: (e) => { e.preventDefault(); added.delete(name); renderForm(); updateSaveBar(); } }, "Remove");
  const card = el("details", { class: "item-card added" },
    el("summary", {}, tierChip(name), el("span", { class: "sub" }, "new"), remove), el("div", { class: "body" }, ...rows));
  card.open = true;
  return card;
}

function reviewSection(tiers, o) {
  const V = ["verification"];
  const on = !!edited([...V, "enabled"], (cfg.data.verification || {}).enabled);
  const sw = field([...V, "enabled"], "bool", "", null, { rerender: true }).querySelector(".switch");
  sw.title = on ? "Turn reviews off" : "Turn reviews on";
  if (!on) return section("Reviewing answers",
    "Off. When on, a stronger model reviews answers and a failed one is answered again. Each review is an extra call.", sw);
  return section("Reviewing answers", "A stronger model reads answers; a failed one is answered again.", sw,
    field([...V, "verifier_tier"], "select", "Reviewer", "The model that judges answers.", { choices: tiers }),
    field([...V, "verify_tiers"], "multi", "Review answers from", "None ticked: every model but the reviewer.", { choices: tiers }),
    field([...V, "sample_rate"], "percent", "Share of answers reviewed", "Lower it once you know how often answers fail.", { def: 1 }),
    field([...V, "escalate_to"], "select", "Re-answer failed ones with", "auto: the router picks a model at least as strong.",
      { choices: ["auto", ...tiers], none: "(default) the reviewer" }),
    advanced("review",
      field([...V, "prefilter_tier"], "select", "Cheap first reviewer", "Asked first; only its pass is final.", { choices: tiers }),
      field([...V, "on_unparseable"], "select", "When the review is unreadable", null, { choices: labelled(o.fallback_policies, POLICY_LABELS), none: `(default) ${POLICY_LABELS.accept}` }),
      field([...V, "on_verifier_error"], "select", "When the reviewer fails", null, { choices: labelled(o.fallback_policies, POLICY_LABELS), none: `(default) ${POLICY_LABELS.accept}` }),
      field([...V, "retry_unfinished"], "bool", "Retry cut-off answers once", "On the same model, before paying for a stronger one."),
      field([...V, "max_verdict_tokens"], "number", "Longest review", "Tokens.", { def: 1024, step: 1 })));
}

function serverSection() {
  return section("Server and privacy", null, null,
    field(["server", "port"], "number", "Port", "The port clients connect to; their base URL ends in /v1.", { def: 8080, step: 1 }),
    field(["log", "store_prompts"], "bool", "Keep the text of prompts", "Off: the log keeps only a fingerprint of each prompt."),
    advanced("server",
      field(["server", "host"], "text", "Listen on", "127.0.0.1 keeps it to this machine.", { def: "127.0.0.1" }),
      field(["log", "path"], "text", "Request log file", null, { placeholder: "the default" }),
      field(["catalog", "check_on_start"], "bool", "Check which models are served at startup", "Needed to find models from the sources.", { def: true }),
      field(["catalog", "path"], "text", "Catalog file", null, { placeholder: "the default" }),
      field(["trace", "enabled"], "bool", "This live view", "Off also removes this page.", { def: true }),
      field(["trace", "keep"], "number", "Requests kept in the live view", null, { def: 200, step: 1 })));
}

function pendingChanges() {
  // Edits inside something being removed are moot: the delete covers them.
  const under = (path) => [...removals.keys()].some((k) => pkey(path).startsWith(k + "\u0000"));
  const out = [...edits.values()].filter((c) => !under(c.path));
  for (const path of removals.values()) out.push({ path, delete: true });
  for (const [t, v] of added) out.push({ path: ["tiers", t], value: v });
  for (const [n, v] of addedSources) out.push({ path: ["router", "capabilities", "discover", n], value: v });
  return out;
}

function updateSaveBar() {
  const n = !cfg ? 0 : cmode === "yaml" ? ($("#ctext").value !== (cfg.text || "") ? 1 : 0) : pendingChanges().length;
  $("#csave").hidden = !(n && !$("#view-config").hidden);
  $("#cwrite").disabled = !n;
  $("#ccount").textContent = cmode === "yaml" ? "The file has unsaved edits" : `${n} unsaved change${n === 1 ? "" : "s"}`;
}

function editBody() {
  return cmode === "yaml" ? { text: $("#ctext").value, base: cfg.digest } : { changes: pendingChanges(), base: cfg.digest };
}

async function configCall(method, url) {
  $("#cerr").textContent = "";
  const r = await fetch(url, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(editBody()) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data.error && data.error.message) || `HTTP ${r.status}`);
  return data;
}

function showReview(res) {
  $("#creview").hidden = false;
  $("#cverdict").replaceChildren(res.ok
    ? el("div", { class: "banner good" }, "The router accepts this. Save writes the file; restart serve to use it.")
    : el("div", { class: "banner bad" }, `Not saved: the router would not start with this. ${res.error}`));
  $("#cdiff").replaceChildren(...(res.diff || "(no change)").split("\n").map((l) => el("div", {
    class: l.startsWith("@@") ? "at" : l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : "" }, l || " ")));
  $("#creview").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Busy only while a call is out: the label says so, then the button comes back.
async function busy(button, label, work) {
  const text = button.textContent;
  button.disabled = true; button.classList.add("busy"); button.textContent = label;
  try { await work(); } catch (e) { $("#cerr").textContent = e.message || String(e); }
  finally { button.textContent = text; button.classList.remove("busy"); updateSaveBar(); }
}

const reviewConfig = () => busy($("#ccheck"), "Checking…", async () => showReview(await configCall("POST", "/routing/config/check")));

// Save checks first and writes only what the router would start with.
const saveConfig = () => busy($("#cwrite"), "Saving…", async () => {
  const res = await configCall("POST", "/routing/config/check");
  if (!res.ok) { showReview(res); return; }
  await configCall("PUT", "/routing/config");
  await loadConfig();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

function setMode(mode) {
  cmode = mode;
  for (const b of document.querySelectorAll("#view-config .tabs button")) b.classList.toggle("on", b.dataset.mode === mode);
  $("#cform").hidden = mode !== "form"; $("#cyaml").hidden = mode !== "yaml";
  $("#creview").hidden = true;
  updateSaveBar();
}

for (const b of document.querySelectorAll("#view-config .tabs button")) b.addEventListener("click", () => {
  if (b.dataset.mode === cmode || !cfg) return;
  const dirty = cmode === "yaml" ? $("#ctext").value !== (cfg.text || "") : pendingChanges().length;
  if (dirty && !confirm("Switching drops the unsaved edits. Switch?")) return;
  edits.clear(); removals.clear(); added.clear(); addedSources.clear(); $("#ctext").value = cfg.text || "";
  renderForm(); setMode(b.dataset.mode);
});
$("#ctext").addEventListener("input", updateSaveBar);
$("#creload").addEventListener("click", loadConfig);
$("#cdiscard").addEventListener("click", loadConfig);
$("#ccheck").addEventListener("click", reviewConfig);
$("#cwrite").addEventListener("click", saveConfig);
window.addEventListener("beforeunload", (e) => { if (!$("#csave").hidden) e.preventDefault(); });

function showView(name) {
  for (const b of document.querySelectorAll("header .tabs button")) {
    const on = b.dataset.view === name; b.classList.toggle("on", on); b.setAttribute("aria-selected", String(on));
  }
  $("#view-requests").hidden = name !== "requests";
  $("#view-tiers").hidden = name !== "tiers";
  $("#view-config").hidden = name !== "config";
  if (name === "tiers" && !tierData) loadTiers();
  if (name === "config" && !cfg) loadConfig();
  if (cfg) updateSaveBar();
  const hash = name === "requests" ? "" : "#" + name;
  if (location.hash !== hash) history.replaceState(null, "", location.pathname + location.search + hash);
}

for (const b of document.querySelectorAll("header .tabs button")) b.addEventListener("click", () => showView(b.dataset.view));
for (const id of ["#tq", "#teffort", "#tdom"]) $(id).addEventListener("input", renderTiers);
$("#treload").addEventListener("click", loadTiers);
showView(["#tiers", "#config"].includes(location.hash) ? location.hash.slice(1) : "requests");

$("#dry").addEventListener("click", () => submit("dry"));
$("#send").addEventListener("click", () => submit("send"));
$("#msg").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit("dry"); });
poll(); pollHealth();
</script>
</body>
</html>
"""
