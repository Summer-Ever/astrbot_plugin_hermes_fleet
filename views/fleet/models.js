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
      const src = (d.configured || {}).source;
      $("model-summary").textContent =
        (d.summary || "—") + (src ? "  ·  " + (SOURCE_TEXT[src] || src) : "");
      chainSt.catalog = d.catalog || [];
      resetDraft(d.configured);
      renderEditor();
      renderStats(d.stats);
      renderLog(d.log);
    } catch (err) {
      toast("读取模型链失败：" + errMsg(err), true);
    }
  }

  // ---------------------------------------------------------------- 链编辑器

  /* 为什么把「换模型」放在这一页：诊断出「主模型挂了」之后，下一步就是换掉它。
     跳去插件配置页改一串逗号分隔的 id，是这条链路里最容易写错的一步。

     候选只来自主应用（`catalog`）：记忆系统不自己维护模型清单，只是从
     「已经配好、能用的」里面挑。挑完写进本插件自己的配置，所以之后主应用
     换默认模型，这里不动——这就是与全局配置解耦的意思。 */

  const CURRENT = "__current__";
  const MAX_FALLBACKS = 4;
  const SOURCE_TEXT = {
    explicit: "已显式指定（与主应用解耦）",
    legacy: "旧的单值配置（保存一次即转成链）",
    current: "跟随主应用当前对话模型（未解耦）",
  };

  const chainSt = {
    catalog: [], current: "", configured: null,
    primary: CURRENT, fallbacks: [],
  };

  function optionLabel(row) {
    // id 里通常已经含了模型名（`阿里云/qwen3.8-flash`），再拼一遍是噪音。
    const model = row.model && row.id.indexOf(row.model) < 0 ? " · " + row.model : "";
    return row.id + model + (row.current ? "（主应用默认）" : "");
  }

  function resetDraft(cfg) {
    chainSt.configured = cfg || {};
    chainSt.primary = chainSt.configured.primary || CURRENT;
    chainSt.fallbacks = (chainSt.configured.fallbacks || []).slice(0, MAX_FALLBACKS);
  }

  function usedIds() {
    return [chainSt.primary].concat(chainSt.fallbacks);
  }

  function known(id) {
    return chainSt.catalog.find((c) => c.id === id);
  }

  function renderEditor() {
    const sel = $("chain-primary");
    clear(sel);
    const cur = el("option", "", "跟随主应用当前对话模型");
    cur.value = CURRENT;
    sel.appendChild(cur);
    chainSt.catalog.forEach((it) => {
      const o = el("option", "", optionLabel(it));
      o.value = it.id;
      sel.appendChild(o);
    });
    // 配置里的 id 可能已经从主应用删掉了。仍要放进下拉：否则 select.value
    // 设不上，页面会静默显示成第一个选项（一个完全不同的模型）。
    usedIds().forEach((id) => {
      if (id && id !== CURRENT && !known(id)) {
        const o = el("option", "", id + "（主应用里已没有这个模型）");
        o.value = id;
        sel.appendChild(o);
      }
    });
    sel.value = chainSt.primary;

    $("chain-primary-hint").textContent = chainSt.primary === CURRENT
      ? "未解耦：主应用换默认模型时，记忆的归纳风格会跟着变。"
      : "已固定。主应用换默认模型不影响记忆。";

    const box = $("chain-fallbacks");
    clear(box);
    if (!chainSt.fallbacks.length) {
      box.appendChild(el("div", "empty small",
        "没有降级模型：主模型失败时这一轮总结就直接放弃。"));
    }
    chainSt.fallbacks.forEach((id, i) => {
      const row = el("div", "fb-row");
      row.appendChild(el("b", "fb-idx", String(i + 1)));
      const hit = known(id);
      row.appendChild(el("span", "mono grow",
        hit ? optionLabel(hit) : id + "（已不在主应用里）"));

      const up = el("button", "btn tiny ghost", "↑");
      up.disabled = i === 0;
      up.addEventListener("click", () => moveFb(i, -1));
      const down = el("button", "btn tiny ghost", "↓");
      down.disabled = i === chainSt.fallbacks.length - 1;
      down.addEventListener("click", () => moveFb(i, 1));
      const del = el("button", "btn tiny danger", "移除");
      del.addEventListener("click", () => {
        chainSt.fallbacks.splice(i, 1);
        renderEditor();
      });
      row.appendChild(up);
      row.appendChild(down);
      row.appendChild(del);
      box.appendChild(row);
    });

    const addSel = $("chain-add-sel");
    clear(addSel);
    const taken = usedIds();
    const pool = chainSt.catalog.filter((c) => taken.indexOf(c.id) < 0);
    if (!pool.length) {
      const o = el("option", "",
        chainSt.catalog.length ? "没有其他可选模型" : "读取不到主应用的模型清单");
      o.value = "";
      addSel.appendChild(o);
    }
    pool.forEach((it) => {
      const o = el("option", "", optionLabel(it));
      o.value = it.id;
      addSel.appendChild(o);
    });
    $("chain-add").disabled = !pool.length || chainSt.fallbacks.length >= MAX_FALLBACKS;

    renderChainWarn();
  }

  function moveFb(i, delta) {
    const j = i + delta;
    const arr = chainSt.fallbacks;
    if (j < 0 || j >= arr.length) return;
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
    renderEditor();
  }

  function renderChainWarn() {
    const box = $("chain-warn");
    const cfg = chainSt.configured || {};
    const parts = [];
    if (!chainSt.catalog.length) {
      parts.push("读取不到主应用的模型清单（可能 AstrBot 还没完全启动）："
        + "现在只能调整已有顺序；新模型请先在主应用的「服务提供商」里添加。");
    }
    if (chainSt.primary === CURRENT) {
      parts.push("当前跟随主应用当前对话模型，这不是解耦状态。"
        + "想固定下来，请在上面选一个具体模型。");
    }
    if (cfg.source === "legacy") {
      parts.push("配置还在用旧的单值字段，保存一次就会转成链。");
    }
    if (!parts.length) {
      box.classList.add("hidden");
      box.textContent = "";
      return;
    }
    box.classList.remove("hidden");
    box.textContent = parts.join(" ");
  }

  async function saveChain() {
    const btn = $("chain-save");
    btn.disabled = true;
    $("chain-msg").textContent = "";
    try {
      const d = await apiPost("models/chain", {
        primary: chainSt.primary,
        fallbacks: chainSt.fallbacks,
      });
      $("chain-msg").textContent = d.message || "";
      toast(d.applied ? "模型链已保存并生效" : "已保存，重载插件后生效", !d.applied);
      await load();
    } catch (err) {
      // 后端的拒绝理由（例如「AstrBot 里没有这个模型」）要原样显示：
      // 这是唯一能告诉人「为什么存不进去」的地方。
      toast("保存失败：" + errMsg(err), true);
    } finally {
      btn.disabled = false;
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

  $("chain-add").addEventListener("click", () => {
    const id = $("chain-add-sel").value;
    if (!id) return;
    if (chainSt.fallbacks.length >= MAX_FALLBACKS) return;
    if (usedIds().indexOf(id) >= 0) return;
    chainSt.fallbacks.push(id);
    renderEditor();
  });
  $("chain-save").addEventListener("click", saveChain);
  $("chain-reset").addEventListener("click", () => {
    resetDraft(chainSt.configured);
    renderEditor();
    $("chain-msg").textContent = "";
  });
  $("chain-primary").addEventListener("change", (e) => {
    chainSt.primary = e.target.value;
    // 主模型换成了某个降级里的模型：把它从降级里摘掉，
    // 否则保存时会被后端以「重复」拒绝，人还得自己找哪里重了。
    chainSt.fallbacks = chainSt.fallbacks.filter((x) => x !== chainSt.primary);
    renderEditor();
  });

  $("btn-model-test").addEventListener("click", testModels);
  $("btn-model-refresh").addEventListener("click", () => { load(); toast("已刷新模型链"); });

  F.pages.models = { enter() { return load(); } };
})();
