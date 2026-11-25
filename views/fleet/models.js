/* 模型链页。
   诊断「后台总结为什么不动」的第一现场：链上谁在冷却、最近 64 次调用
   有没有失手、单模型还能不能通。hermes_core 把这些塞在「概览」的一个角落里，
   百账号场景下它值得独占一页。 */

(function () {
  "use strict";

  const F = window.Fleet;
  if (!F) return;
  const { apiGet, apiPost, el, clear, toast, relTime, num, errMsg } = F;
  const $ = (id) => document.getElementById(id);

  function renderChain(chain) {
    const box = $("model-chain");
    clear(box);
    if (!chain || !chain.length) {
      box.appendChild(el("div", "empty", "没有配置任何模型。"));
      return;
    }
    chain.forEach((item, i) => {
      const node = el("div", "chain-node" + (item.cooling ? " cooling" : ""));
      node.appendChild(el("span", "chain-idx", String(i + 1)));
      node.appendChild(el("span", "chain-name mono", item.name || "(当前对话模型)"));
      if (item.cooling) {
        node.appendChild(el("span", "badge warn", "冷却 " + item.cooldown_left + "s"));
      } else {
        node.appendChild(el("span", "badge ok", i === 0 ? "主模型" : "可降级"));
      }
      box.appendChild(node);
      if (i < chain.length - 1) box.appendChild(el("span", "chain-arrow", "→"));
    });
  }

  function renderStats(stats) {
    const box = $("model-stats");
    clear(box);
    const s = stats || {};
    const cells = [
      ["调用", num(s.calls)],
      ["失败", num(s.failures)],
    ];
    Object.keys(s.by_provider || {}).forEach((k) => {
      cells.push([k, num(s.by_provider[k])]);
    });
    Object.keys(s.by_kind || {}).forEach((k) => {
      cells.push(["失败·" + k, num(s.by_kind[k])]);
    });
    cells.forEach(([k, v]) => {
      const cell = el("div", "kpi");
      cell.appendChild(el("b", "", v));
      cell.appendChild(el("span", "", k));
      box.appendChild(cell);
    });
  }

  function renderLog(log) {
    const tb = $("model-log");
    clear(tb);
    $("model-log-empty").classList.toggle("hidden", (log || []).length > 0);
    // 后端的 log 是 deque append（旧 → 新），这里倒过来：最新的排最上面，
    // 与「最近调用」这个标题一致。
    (log || []).slice().reverse().forEach((it) => {
      const tr = el("tr");
      tr.appendChild(el("td", "w-90 dim", relTime(it.ts)));
      tr.appendChild(el("td", "w-40p mono", it.provider || "—"));
      tr.appendChild(el("td", "w-80", it.purpose || "—"));
      tr.appendChild(el("td", "w-70", it.ok ? "成功" : "失败"));
      tr.appendChild(el("td", "num w-70", it.latency_ms ? it.latency_ms + " ms" : "—"));
      const td = el("td", "dim small", it.ok ? "" : (it.kind || "") + " " + (it.detail || ""));
      td.title = it.detail || "";
      tr.appendChild(td);
      tb.appendChild(tr);
    });
  }

  async function load() {
    try {
      const d = await apiGet("models");
      renderChain(d.chain);
      $("model-summary").textContent = d.summary || "—";
      renderStats(d.stats);
      renderLog(d.log);
    } catch (err) {
      toast("读取模型链失败：" + errMsg(err), true);
    }
  }

  async function testModels() {
    const btn = $("btn-model-test");
    const box = $("model-test-result");
    btn.disabled = true;
    btn.textContent = "测试中…";
    clear(box);
    box.appendChild(el("div", "hint", "逐个发一句最小请求（会真的消耗配额，最多 3 个）…"));
    try {
      const d = await apiPost("models/test", {});
      clear(box);
      (d.items || []).forEach((it) => {
        const row = el("div", "test-row " + (it.ok ? "ok" : "bad"));
        row.appendChild(el("span", "mono grow", it.name));
        row.appendChild(el("span", "", it.ok
          ? (it.latency_ms + " ms · " + (it.model || "") + " · " + (it.reply || ""))
          : (it.error || "失败")));
        box.appendChild(row);
      });
      await load();
    } catch (err) {
      clear(box);
      box.appendChild(el("div", "hint", "测试失败：" + errMsg(err)));
    } finally {
      btn.disabled = false;
      btn.textContent = "逐个测连通";
    }
  }

  $("btn-model-test").addEventListener("click", testModels);
  $("btn-model-refresh").addEventListener("click", () => { load(); toast("已刷新模型链"); });

  F.pages.models = { enter() { return load(); } };
})();
