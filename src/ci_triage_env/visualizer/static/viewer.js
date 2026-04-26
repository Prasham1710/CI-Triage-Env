// CI-Triage replay visualizer — vanilla JS, no framework, no CDN.
// Loads an EpisodeTrace JSON (and optionally a Scenario JSON for the failure
// summary + ground truth) and renders a read-only replay.

(() => {
  "use strict";

  let state = {
    trace: null,
    trace2: null,
    scenario: null,
    selectedStep: null,
  };

  // ----- DOM helpers ------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function el(tag, opts = {}, ...children) {
    const e = document.createElement(tag);
    if (opts.cls) e.className = opts.cls;
    if (opts.text != null) e.textContent = opts.text;
    if (opts.html != null) e.innerHTML = opts.html;
    if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) e.setAttribute(k, v);
    for (const c of children) if (c) e.appendChild(c);
    return e;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function fmtCost(c) {
    if (c == null) return "—";
    return c.toFixed(c < 0.01 ? 4 : 3);
  }

  function costClass(c) {
    if (c == null) return "";
    if (c < 0.05) return "cost-low";
    if (c < 0.30) return "cost-mid";
    return "cost-high";
  }

  function safeGet(obj, path, dflt = null) {
    let cur = obj;
    for (const key of path) {
      if (cur == null) return dflt;
      cur = cur[key];
    }
    return cur == null ? dflt : cur;
  }

  // ----- Loaders ----------------------------------------------------------
  function loadFile(file, cb) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try { cb(JSON.parse(e.target.result)); }
      catch (err) { alert("Failed to parse JSON: " + err.message); }
    };
    reader.readAsText(file);
  }

  function onTrace(json) { state.trace = json; render(); }
  function onTrace2(json) { state.trace2 = json; render(); }
  function onScenario(json) { state.scenario = json; render(); }

  function reset() {
    state = { trace: null, trace2: null, scenario: null, selectedStep: null };
    for (const id of ["trace-loader", "scenario-loader", "trace-loader-2"]) {
      $(id).value = "";
    }
    render();
  }

  // ----- Render: failure summary -----------------------------------------
  function renderFailureSummary() {
    const node = $("failure-summary");
    clear(node);
    const sc = state.scenario;
    if (!sc) {
      node.appendChild(el("em", { text: "Upload a scenario JSON to see the failure summary." }));
      const sid = safeGet(state.trace, ["episode", "scenario_id"]);
      if (sid) node.appendChild(el("div", { cls: "kv-mini", text: "scenario_id: " + sid }));
      return;
    }
    const fs = sc.failure_summary || {};
    const dl = el("dl", { cls: "kv" });
    for (const [k, v] of Object.entries(fs)) {
      dl.appendChild(el("dt", { text: k }));
      dl.appendChild(el("dd", { text: String(v) }));
    }
    node.appendChild(dl);
  }

  // ----- Render: budget ---------------------------------------------------
  function renderBudget() {
    const node = $("budget");
    clear(node);
    const ep = safeGet(state.trace, ["episode"]);
    if (!ep) { node.textContent = "—"; return; }
    const dl = el("dl", { cls: "kv" });
    dl.appendChild(el("dt", { text: "tool_calls" }));
    dl.appendChild(el("dd", { text: ep.budget.tool_calls_remaining }));
    dl.appendChild(el("dt", { text: "cost left" }));
    dl.appendChild(el("dd", { text: ep.budget.cost_remaining.toFixed(3) }));
    dl.appendChild(el("dt", { text: "steps" }));
    dl.appendChild(el("dd", { text: ep.step }));
    dl.appendChild(el("dt", { text: "terminated" }));
    dl.appendChild(el("dd", { text: ep.is_terminated ? "yes" : "no" }));
    node.appendChild(dl);
  }

  // ----- Render: ground truth --------------------------------------------
  function renderGroundTruth() {
    const node = $("ground-truth");
    clear(node);
    const terminated = safeGet(state.trace, ["episode", "is_terminated"], false);
    node.classList.toggle("revealed", terminated);
    const sc = state.scenario;
    if (!sc) {
      node.appendChild(el("em", { text: "Upload scenario to reveal ground truth after termination." }));
      return;
    }
    if (!terminated) {
      node.appendChild(el("em", { text: "Hidden until episode terminates." }));
      return;
    }
    const gt = sc.ground_truth || {};
    const dl = el("dl", { cls: "kv" });
    dl.appendChild(el("dt", { text: "label" }));
    dl.appendChild(el("dd", { text: gt.label || "?" }));
    dl.appendChild(el("dt", { text: "ambiguous" }));
    dl.appendChild(el("dd", { text: gt.is_ambiguous ? "yes" : "no" }));
    if (gt.confidence_target != null) {
      dl.appendChild(el("dt", { text: "target conf" }));
      dl.appendChild(el("dd", { text: gt.confidence_target.toFixed(2) }));
    }
    if (gt.rationale) {
      dl.appendChild(el("dt", { text: "rationale" }));
      const dd = el("dd");
      dd.appendChild(el("pre", { text: gt.rationale }));
      dl.appendChild(dd);
    }
    node.appendChild(dl);

    // Color-code agent's diagnosis vs ground truth.
    const finalAction = safeGet(state.trace, ["episode", "final_action"]);
    if (finalAction && finalAction.diagnosis) {
      const correct = finalAction.diagnosis === gt.label;
      node.appendChild(el("div", {
        cls: correct ? "diag-correct" : "diag-incorrect",
        text: "Agent submitted: " + finalAction.diagnosis + (correct ? " ✓" : " ✗"),
      }));
    }
  }

  // ----- Render: timeline -------------------------------------------------
  function renderTimeline() {
    const node = $("timeline");
    clear(node);
    const history = safeGet(state.trace, ["episode", "history"], []);
    if (!history.length) {
      node.appendChild(el("em", { text: "No history records." }));
      return;
    }
    history.forEach((rec, i) => {
      const isTerminal = rec.action && rec.action.action_type === "submit_diagnosis";
      const cls = "step-card" + (isTerminal ? " terminal" : "")
        + (state.selectedStep === i ? " selected" : "");
      const card = el("div", { cls });
      const toolLabel = isTerminal
        ? "submit_diagnosis"
        : (rec.action && rec.action.tool_name) || "?";
      card.appendChild(el("div", { cls: "step-tool", text: "step " + rec.step + " · " + toolLabel }));
      const meta = el("div", { cls: "step-meta" });
      meta.appendChild(el("span", {
        cls: "cost " + costClass(rec.cost_charged),
        text: "$ " + fmtCost(rec.cost_charged),
      }));
      const obs = rec.observation || {};
      const budget = obs.budget_remaining || {};
      meta.appendChild(el("span", {
        text: "calls: " + (budget.tool_calls_remaining ?? "?"),
      }));
      card.appendChild(meta);
      card.addEventListener("click", () => {
        state.selectedStep = i;
        render();
      });
      node.appendChild(card);
    });
  }

  function renderStepDetail() {
    const node = $("step-detail");
    clear(node);
    const idx = state.selectedStep;
    const history = safeGet(state.trace, ["episode", "history"], []);
    if (idx == null || !history[idx]) {
      node.appendChild(el("em", { text: "Click a step on the timeline." }));
      return;
    }
    const rec = history[idx];
    const sec = (title, body) => {
      const wrap = el("div", { cls: "step-detail-section" });
      wrap.appendChild(el("div", { cls: "name", text: title }));
      wrap.appendChild(body);
      return wrap;
    };
    const action = rec.action || {};
    node.appendChild(sec("Action", el("pre", { text: JSON.stringify(action, null, 2) })));
    if (rec.observation && rec.observation.tool_response) {
      node.appendChild(sec("Tool response", el("pre", {
        text: JSON.stringify(rec.observation.tool_response, null, 2),
      })));
    }
    node.appendChild(sec("Cost charged", el("pre", { text: fmtCost(rec.cost_charged) })));
  }

  // ----- Render: reward breakdown ----------------------------------------
  function renderRewardBreakdown() {
    const node = $("reward-breakdown");
    clear(node);
    const rb = safeGet(state.trace, ["reward_breakdown"]);
    if (!rb) { node.textContent = "—"; return; }
    const summary = el("div", { cls: "kv-line" });
    summary.appendChild(el("div", { text: "total: " + (rb.total != null ? rb.total.toFixed(3) : "?") }));
    summary.appendChild(el("div", {
      text: "format gate: " + (rb.format_gate ? "✓" : "✗"),
      cls: rb.format_gate ? "diag-correct" : "diag-incorrect",
    }));
    node.appendChild(summary);

    const components = rb.components || {};
    const entries = Object.entries(components);
    if (!entries.length) {
      node.appendChild(el("em", { text: "Reward layer not yet run on this trace." }));
      return;
    }
    const max = Math.max(0.001, ...entries.map(([, v]) => Math.abs(v.weighted ?? 0)));
    for (const [name, comp] of entries) {
      const w = comp.weighted ?? 0;
      const row = el("div", { cls: "bar-row" });
      row.appendChild(el("div", { cls: "name", text: name }));
      const bar = el("div", { cls: "bar" });
      const fill = el("span");
      fill.style.width = (Math.abs(w) / max * 100).toFixed(1) + "%";
      bar.appendChild(fill);
      row.appendChild(bar);
      row.appendChild(el("div", { cls: "val", text: w.toFixed(3) }));
      node.appendChild(row);
    }
  }

  // ----- Render: counterfactual probe ------------------------------------
  function renderCounterfactual() {
    const node = $("counterfactual");
    clear(node);
    // v1: probe is dormant — counterfactual_replay is always null and
    // counterfactual on RewardBreakdown is null. Keep the panel as scaffolding
    // for v2 (see plan/branch-a-env-core/phase-a4.md).
    const cfReplay = safeGet(state.trace, ["counterfactual_replay"]);
    const cfReward = safeGet(state.trace, ["reward_breakdown", "counterfactual"]);
    if (!cfReplay && !cfReward) {
      node.appendChild(el("em", { text: "No probe fired (deferred to v2)." }));
      return;
    }
    if (cfReward) {
      const dl = el("dl", { cls: "kv" });
      for (const k of ["fired", "probe_step", "probe_action", "predicted_outcome", "actual_outcome", "brier_score"]) {
        if (cfReward[k] != null) {
          dl.appendChild(el("dt", { text: k }));
          dl.appendChild(el("dd", { text: String(cfReward[k]) }));
        }
      }
      node.appendChild(dl);
    }
  }

  // ----- Render: comparison ----------------------------------------------
  function renderComparison() {
    const node = $("comparison");
    clear(node);
    if (!state.trace2) {
      node.appendChild(el("em", { text: "Load a second trace to compare." }));
      return;
    }
    const summarize = (t, label) => {
      const ep = t.episode || {};
      const finalAction = ep.final_action;
      return {
        label,
        steps: ep.step,
        cost_used: 5.0 - (ep.budget && ep.budget.cost_remaining != null ? ep.budget.cost_remaining : 5.0),
        diagnosis: finalAction ? finalAction.diagnosis : "—",
        confidence: finalAction ? finalAction.confidence : null,
        reward_total: safeGet(t, ["reward_breakdown", "total"]),
      };
    };
    const a = summarize(state.trace, "A (primary)");
    const b = summarize(state.trace2, "B (compare)");
    const tbl = el("table", { cls: "compare-tbl" });
    const thead = el("tr");
    thead.appendChild(el("th", { text: "" }));
    thead.appendChild(el("th", { text: "A" }));
    thead.appendChild(el("th", { text: "B" }));
    tbl.appendChild(thead);
    for (const k of ["steps", "cost_used", "diagnosis", "confidence", "reward_total"]) {
      const tr = el("tr");
      tr.appendChild(el("td", { text: k }));
      tr.appendChild(el("td", { text: a[k] == null ? "—" : String(a[k]) }));
      tr.appendChild(el("td", { text: b[k] == null ? "—" : String(b[k]) }));
      tbl.appendChild(tr);
    }
    node.appendChild(tbl);
  }

  // ----- Master render ----------------------------------------------------
  function render() {
    $("empty-state").style.display = state.trace ? "none" : "block";
    if (!state.trace) {
      for (const id of ["failure-summary", "budget", "ground-truth", "timeline", "step-detail",
        "reward-breakdown", "counterfactual", "comparison"]) {
        const n = $(id);
        clear(n);
        n.appendChild(el("em", { text: "—" }));
      }
      return;
    }
    renderFailureSummary();
    renderBudget();
    renderGroundTruth();
    renderTimeline();
    renderStepDetail();
    renderRewardBreakdown();
    renderCounterfactual();
    renderComparison();
  }

  // ----- Wire up ----------------------------------------------------------
  document.getElementById("trace-loader").addEventListener("change", (e) => {
    if (e.target.files[0]) loadFile(e.target.files[0], onTrace);
  });
  document.getElementById("scenario-loader").addEventListener("change", (e) => {
    if (e.target.files[0]) loadFile(e.target.files[0], onScenario);
  });
  document.getElementById("trace-loader-2").addEventListener("change", (e) => {
    if (e.target.files[0]) loadFile(e.target.files[0], onTrace2);
  });
  document.getElementById("reset-btn").addEventListener("click", reset);

  render();
})();
