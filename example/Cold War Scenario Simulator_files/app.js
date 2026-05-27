/* Cold War Scenario Simulator - frontend */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const PROGRESS_STEPS = [
    "init", "evidence", "round1", "round2", "round3",
    "red_team", "synthesis", "image", "save",
  ];

  function setProgress(activeIdx) {
    const items = document.querySelectorAll("#progress li");
    items.forEach((li, i) => {
      li.classList.remove("active", "done");
      if (i < activeIdx) li.classList.add("done");
      else if (i === activeIdx) li.classList.add("active");
    });
  }
  function clearProgress() { setProgress(-1); }
  function completeProgress() {
    document.querySelectorAll("#progress li").forEach((li) => {
      li.classList.remove("active");
      li.classList.add("done");
    });
  }

  async function loadConfig() {
    try {
      const r = await fetch("/api/config");
      const cfg = await r.json();
      const pill = $("#modelPill");
      const mode = cfg.mock_mode ? "MOCK" : "LIVE";
      pill.textContent = "model: " + cfg.model + " · " + mode;
    } catch (e) { /* ignore */ }
  }

  async function loadSavedRuns() {
    try {
      const r = await fetch("/api/runs");
      const runs = await r.json();
      const ul = $("#savedRuns");
      ul.innerHTML = "";
      if (!runs.length) {
        ul.innerHTML = '<li class="muted">No saved runs yet.</li>';
        return;
      }
      runs.forEach((run) => {
        const li = document.createElement("li");
        li.innerHTML =
          '<div class="seed">' + escapeHtml(run.scenario_title || run.seed) + '</div>' +
          '<div class="ts">' + escapeHtml(run.created_at) +
          ' · ' + escapeHtml(run.scenario_mode) + '</div>';
        li.addEventListener("click", () => loadRun(run.run_id));
        ul.appendChild(li);
      });
    } catch (e) { /* ignore */ }
  }

  async function loadRun(runId) {
    try {
      $("#status").textContent = "Loading saved run...";
      const r = await fetch("/api/runs/" + encodeURIComponent(runId));
      if (!r.ok) throw new Error("not found");
      const data = await r.json();
      renderResult(data);
      $("#status").textContent = "";
    } catch (e) {
      $("#status").textContent = "Failed to load run.";
    }
  }

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderResult(data) {
    $("#empty").classList.add("hidden");
    $("#result").classList.remove("hidden");

    $("#scenarioTitle").textContent = data.scenario_title || "(untitled scenario)";
    $("#scenarioSummary").textContent = data.scenario_summary || "";

    const badge = $("#statusBadge");
    badge.textContent = data.event_status || "hypothetical";
    badge.className = "badge " + (data.event_status || "hypothetical");

    renderTimeline(data.timeline || []);
    renderDiscussion(data.discussion_summary || []);
    renderAgentSummaries(data.agent_summaries || {});
    renderList("#disagreements", data.main_disagreements || []);
    renderList("#redTeam", data.red_team_warnings || []);
    renderImage(data.image || {}, data.image_prompt || "", data.run_id);
    renderMetrics(data.run_metrics || {});
  }

  function renderTimeline(timeline) {
    const root = $("#timeline");
    root.innerHTML = "";
    if (!timeline.length) {
      root.innerHTML = '<div class="muted">No timeline produced.</div>';
      return;
    }
    timeline.forEach((yb) => {
      const card = document.createElement("div");
      card.className = "year-card";
      const events = (yb.events || []).map((ev) => {
        const prob = (ev.probability != null) ? (Math.round(ev.probability * 100) + "%") : "—";
        return (
          '<div class="event">' +
          '<div>' + escapeHtml(ev.event || "") + '</div>' +
          '<div class="event-meta">' +
          '<span class="tag">' + escapeHtml(ev.domain || "") + '</span>' +
          '<span class="tag impact-' + escapeHtml(ev.impact || "medium") + '">impact: ' + escapeHtml(ev.impact || "") + '</span>' +
          '<span class="tag confidence-' + escapeHtml(ev.confidence || "medium") + '">conf: ' + escapeHtml(ev.confidence || "") + '</span>' +
          '<span class="tag">p=' + prob + '</span>' +
          '</div>' +
          '<div class="muted small">' + escapeHtml(ev.rationale || "") + '</div>' +
          '</div>'
        );
      }).join("");
      card.innerHTML =
        '<div class="year">' + escapeHtml(String(yb.year)) + '</div>' +
        '<div class="headline">' + escapeHtml(yb.headline || "") + '</div>' +
        (events || '<div class="muted small">No events.</div>');
      root.appendChild(card);
    });
  }

  function renderDiscussion(rounds) {
    const root = $("#discussion");
    root.innerHTML = "";
    if (!rounds.length) {
      root.innerHTML = '<div class="muted">No discussion summary.</div>';
      return;
    }
    rounds.forEach((r) => {
      const block = document.createElement("div");
      block.className = "round-block";
      const agree = (r.areas_of_agreement || []).map(escapeHtml).join(" · ") || "—";
      const disagree = (r.areas_of_disagreement || []).map(escapeHtml).join(" · ") || "—";
      const uncert = (r.key_uncertainties || []).map(escapeHtml).join(" · ") || "—";
      block.innerHTML =
        '<div class="label">Round ' + escapeHtml(String(r.round_number || "?")) + '</div>' +
        '<div class="row"><strong>Agree:</strong> ' + agree + '</div>' +
        '<div class="row"><strong>Disagree:</strong> ' + disagree + '</div>' +
        '<div class="row"><strong>Uncertain:</strong> ' + uncert + '</div>';
      root.appendChild(block);
    });
  }

  function renderAgentSummaries(map) {
    const root = $("#agentSummaries");
    root.innerHTML = "";
    const order = [
      "geo_strategy", "economy_technology", "domestic_ideology",
      "security_taiwan", "historical_analogy", "red_team",
    ];
    let any = false;
    order.forEach((k) => {
      if (!map[k]) return;
      any = true;
      const div = document.createElement("div");
      div.className = "agent-card";
      div.innerHTML =
        '<div class="name">' + escapeHtml(k.replace(/_/g, " ")) + '</div>' +
        '<div>' + escapeHtml(map[k]) + '</div>';
      root.appendChild(div);
    });
    if (!any) root.innerHTML = '<div class="muted">No agent summaries.</div>';
  }

  function renderList(sel, items) {
    const root = $(sel);
    root.innerHTML = "";
    if (!items.length) {
      root.innerHTML = '<li class="muted">None.</li>';
      return;
    }
    items.forEach((x) => {
      const li = document.createElement("li");
      li.textContent = x;
      root.appendChild(li);
    });
  }

  function renderImage(image, prompt, runId) {
    const box = $("#imageBox");
    box.innerHTML = "";
    if (image && image.generated && image.path) {
      const img = document.createElement("img");
      const filename = image.path.split("/").pop();
      img.src = "/generated_images/" + filename + "?t=" + Date.now();
      img.alt = "Generated illustration";
      if (image.mock) {
        const note = document.createElement("div");
        note.className = "placeholder";
        note.innerHTML =
          "<strong>Mock image</strong><br/>Placeholder (no API key configured).";
        box.appendChild(note);
      } else {
        box.appendChild(img);
      }
    } else {
      const div = document.createElement("div");
      div.className = "placeholder";
      const err = (image && image.error) ? "<br/>Error: " + escapeHtml(image.error) : "";
      const dis = (image && image.enabled === false) ? "Image generation disabled." : "No image available.";
      div.innerHTML = "<strong>" + dis + "</strong>" + err;
      box.appendChild(div);
    }
    $("#imagePrompt").textContent = prompt || "(no prompt)";
  }

  function renderMetrics(m) {
    const root = $("#metrics");
    root.innerHTML = "";
    const entries = [
      ["LLM calls", m.llm_calls],
      ["Cache hits", m.cache_hits],
      ["Retrieved docs", m.retrieved_docs],
      ["Rounds completed", m.discussion_rounds_completed],
      ["Elapsed (s)", m.elapsed_seconds],
      ["Est. in tokens", m.estimated_input_tokens],
      ["Est. out tokens", m.estimated_output_tokens],
      ["Agents used", (m.agents_used || []).length],
    ];
    entries.forEach(([k, v]) => {
      const d = document.createElement("div");
      d.className = "metric";
      d.innerHTML = '<div class="k">' + escapeHtml(k) + '</div><div class="v">' + escapeHtml(String(v == null ? "—" : v)) + '</div>';
      root.appendChild(d);
    });
  }

  async function runScenario() {
    const seed = $("#seed").value.trim();
    const mode = $("#mode").value;
    if (!seed) {
      $("#status").textContent = "Please enter a seed sentence.";
      return;
    }
    $("#run").disabled = true;
    $("#status").textContent = "Running multi-agent simulation...";
    $("#empty").classList.add("hidden");
    $("#result").classList.add("hidden");

    // Optimistic progress animation; actual run is a single blocking call.
    clearProgress();
    let idx = 0;
    const interval = setInterval(() => {
      if (idx < PROGRESS_STEPS.length) {
        setProgress(idx);
        idx += 1;
      }
    }, 700);

    try {
      const r = await fetch("/api/run-scenario", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: seed, scenario_mode: mode }),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error("HTTP " + r.status + ": " + t);
      }
      const data = await r.json();
      clearInterval(interval);
      completeProgress();
      renderResult(data);
      $("#status").textContent = "Done. Saved run " + data.run_id + ".";
      loadSavedRuns();
    } catch (e) {
      clearInterval(interval);
      $("#status").textContent = "Error: " + e.message;
    } finally {
      $("#run").disabled = false;
    }
  }

  function wireExamples() {
    document.querySelectorAll(".chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        $("#seed").value = btn.dataset.seed || "";
      });
    });
  }

  function init() {
    wireExamples();
    $("#run").addEventListener("click", runScenario);
    loadConfig();
    loadSavedRuns();
  }
  document.addEventListener("DOMContentLoaded", init);
})();
