/* Fleet 账号台 · 前端
   通过 AstrBotPluginView 暴露的 apiGet/apiPost 调后端（webapi.py）。

   两条规矩：
   1. **不用 innerHTML 插数据。** platform_id、文件名、记忆正文都来自运行
      环境，其中记忆正文是用户输入。全部走 textContent，避免在管理台上
      执行到用户写的内容。
   2. **不猜数据。** 没探测过的账号，记忆数显示「-」而不是 0；接口失败时
      显示错误，不用上次的值冒充。 */

(function () {
  "use strict";

  const bridge = window.AstrBotPluginView || window.AstrBotPluginPage || null;
  const $ = (id) => document.getElementById(id);

  const state = {
    q: "", sort: "recent", limit: 50, offset: 0, total: 0,
    rows: [], probed: false, busy: false, alerts: [], alertsOpen: false,
  };

  // ---------------------------------------------------------------- 基础

  function apiGet(path, params) {
    if (!bridge) return Promise.reject(new Error("no-bridge"));
    return bridge.apiGet(path, params || {});
  }

  function apiPost(path, body) {
    if (!bridge) return Promise.reject(new Error("no-bridge"));
    return bridge.apiPost(path, body || {});
  }

  function toast(msg, bad) {
    const t = $("toast");
    t.textContent = String(msg);
    t.className = "toast show" + (bad ? " bad" : "");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { t.className = "toast"; }, 2600);
  }

  /** 建元素。children 里字符串一律当文本节点，不做 HTML 解析。 */
  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function relTime(ts) {
    if (!ts) return "—";
    const sec = Date.now() / 1000 - ts;
    if (sec < 0) return "刚刚";
    if (sec < 90) return "刚刚";
    if (sec < 3600) return Math.floor(sec / 60) + " 分钟前";
    if (sec < 86400) return Math.floor(sec / 3600) + " 小时前";
    const d = Math.floor(sec / 86400);
    if (d < 30) return d + " 天前";
    const dt = new Date(ts * 1000);
    return dt.getFullYear() + "-" + String(dt.getMonth() + 1).padStart(2, "0") +
           "-" + String(dt.getDate()).padStart(2, "0");
  }

  function num(v) {
    return (v === undefined || v === null) ? "—" : String(v);
  }

  // ---------------------------------------------------------------- 渲染

  function renderSide(data) {
    const s = data.summary || {};
    $("s-accounts").textContent = num(s.accounts);
    $("s-bytes").textContent = s.total_text || "—";

    const open = $("s-open");
    open.textContent = num(s.opened);
    open.className = "";

    const stale = $("s-stale");
    stale.textContent = num(s.stale);
    stale.className = s.stale ? "warn" : "";

    const orphan = $("s-orphan");
    orphan.textContent = num(s.orphan);
    orphan.className = s.orphan ? "danger" : "";

    const missing = $("s-missing");
    missing.textContent = num(s.missing);
    missing.className = s.missing ? "danger" : "";
  }

  function renderModel(model) {
    const chain = $("m-chain");
    const fail = $("m-fail");
    if (!model || !model.available) {
      chain.textContent = model && model.error ? "不可用" : "不可用";
      chain.title = (model && model.error) || "拿不到模型链信息";
      fail.textContent = "";
      return;
    }
    // 侧栏只有 200px 宽，整条 summary 会折成五行。只取「链」那一段，
    // 失败统计另起一行；完整文本放进 title，需要时悬停看。
    const full = model.summary || "—";
    const cut = full.split("；")[0];
    chain.textContent = cut;
    chain.title = full;
    fail.textContent = model.calls
      ? "调用 " + model.calls + " · 失败 " + model.failures
      : "本次启动尚未调用";
  }

  function renderAlerts(alerts) {
    const box = $("alerts");
    clear(box);
    state.alerts = alerts || [];
    const list = state.alerts;

    if (!list.length) return;

    // 首屏不能全被告警吃掉：error 全展示，warn 只放两条，其余折叠。
    // 100 个账号里静默 40 个是常态，全列出来等于把表格推到屏幕外，
    // 而这一页的主要用途恰恰是看表格。
    const errs = list.filter((a) => a.level === "error");
    const warns = list.filter((a) => a.level !== "error");
    const shown = state.alertsOpen ? list : errs.concat(warns.slice(0, 2));

    shown.forEach((a) => {
      const row = el("div", "alert " + (a.level === "error" ? "error" : "warn"));
      row.appendChild(el("span", "who", a.platform_id || "(未知账号)"));
      row.appendChild(el("span", "msg", a.message));
      box.appendChild(row);
    });

    const rest = list.length - shown.length;
    if (rest > 0) {
      const more = el("button", "alert-more", "另有 " + rest + " 条，展开");
      more.addEventListener("click", () => {
        state.alertsOpen = true;
        renderAlerts(state.alerts);
      });
      box.appendChild(more);
    } else if (state.alertsOpen && list.length > 4) {
      const less = el("button", "alert-more", "收起");
      less.addEventListener("click", () => {
        state.alertsOpen = false;
        renderAlerts(state.alerts);
      });
      box.appendChild(less);
    }
  }

  function statusCell(it) {
    const wrap = el("div", "badges");
    if (it.orphan) wrap.appendChild(el("span", "badge danger", "孤立"));
    else if (it.missing) wrap.appendChild(el("span", "badge danger", "库丢失"));
    else if (it.stale) wrap.appendChild(el("span", "badge warn", "静默 " + it.age_days + " 天"));
    else wrap.appendChild(el("span", "badge ok", "正常"));
    if (it.opened) wrap.appendChild(el("span", "badge accent", "已连接"));
    return wrap;
  }

  function renderTable(items) {
    const tb = $("tbody");
    clear(tb);
    $("empty").classList.toggle("hidden", items.length > 0);

    items.forEach((it) => {
      const tr = el("tr");
      tr.addEventListener("click", () => openDrawer(it));

      const tdName = el("td", "w-40p");
      tdName.appendChild(el("div", "pid mono", it.platform_id || "(索引缺失)"));
      tdName.appendChild(el("div", "stem mono", it.file || it.stem));
      tr.appendChild(tdName);

      tr.appendChild(el("td", "num w-90", it.size_text));
      tr.appendChild(el("td", "w-130", relTime(it.last_write_at)));

      const st = it.stats;
      const tdMem = el("td", "num w-90");
      tdMem.appendChild(el("span", st ? "" : "dash",
        st ? num(st.memories) : "—"));
      tr.appendChild(tdMem);

      const tdUn = el("td", "num w-90");
      const un = st ? st.unsummarized : null;
      const tdEl = el("span", st && un ? "badge warn" : (st ? "dash" : "dash"),
        st ? num(un) : "—");
      tdUn.appendChild(tdEl);
      tr.appendChild(tdUn);

      const tdSt = el("td", "w-110");
      tdSt.appendChild(statusCell(it));
      tr.appendChild(tdSt);

      tb.appendChild(tr);
    });
  }

  function renderPager(data) {
    const from = data.total === 0 ? 0 : data.offset + 1;
    const to = Math.min(data.offset + data.items.length, data.total);
    $("page-info").textContent =
      "共 " + data.total + " 个账号，当前显示 " + from + "–" + to +
      (state.probed ? "（本页已取真实统计）" : "（记忆数为未探测，点「取统计」）");
    $("prev").disabled = data.offset <= 0 || state.busy;
    $("next").disabled = data.offset + data.limit >= data.total || state.busy;
  }

  // ---------------------------------------------------------------- 加载

  async function load() {
    if (state.busy) return;
    state.busy = true;
    $("btn-refresh").disabled = true;
    try {
      const data = await apiGet("overview", {
        q: state.q, sort: state.sort, limit: state.limit, offset: state.offset,
        probe: state.probed ? "1" : "0",
      });
      state.total = data.total || 0;
      state.rows = data.items || [];
      $("s-version").textContent = data.version || "—";
      renderSide(data);
      renderAlerts(data.alerts);
      renderTable(state.rows);
      renderPager(data);
      loadHealth();
    } catch (err) {
      toast("加载失败：" + (err && err.message ? err.message : err), true);
    } finally {
      state.busy = false;
      $("btn-refresh").disabled = false;
    }
  }

  async function loadHealth() {
    try {
      const h = await apiGet("health");
      $("s-root").textContent = h.root || "—";
      $("s-root").title = h.root || "";
      const cap = h.capacity || {};
      const open = $("s-open");
      open.textContent = num(cap.opened) + " / " + num(cap.max_open);
      open.className = cap.saturated ? "warn" : "";
      open.title = cap.saturated
        ? "已达上限：每次访问新账号都要驱逐一个连接"
        : "已打开 / 上限";
      renderModel(h.model);
    } catch (err) {
      /* 健康信息拿不到不影响主列表，不弹提示打断操作 */
    }
  }

  // ---------------------------------------------------------------- 详情

  function statLine(dl, k, v, cls) {
    dl.appendChild(el("dt", "", k));
    const dd = el("dd", cls || "", v);
    dl.appendChild(dd);
  }

  /** 统计块。`stats` 为 null 时显式写「未探测」——不拿 0 冒充。 */
  function buildStatList(it, stats) {
    const dl = el("dl", "kv");
    statLine(dl, "占用", it.size_text + (it.wal_bytes ? "（含 WAL）" : ""));
    statLine(dl, "最后写入", relTime(it.last_write_at) +
             (it.last_write_at ? " · " + new Date(it.last_write_at * 1000).toLocaleString() : ""));
    if (stats) {
      statLine(dl, "归档记忆", num(stats.memories));
      statLine(dl, "常驻词条", num(stats.resident));
      statLine(dl, "消息总数", num(stats.messages));
      statLine(dl, "未总结", num(stats.unsummarized));
    } else {
      statLine(dl, "记忆统计", "未探测", "dash");
    }
    return dl;
  }

  async function openDrawer(it) {
    const drawer = $("drawer");
    const body = $("d-body");
    $("d-title").textContent = it.platform_id || "(索引缺失)";
    $("d-sub").textContent = it.file || it.stem;
    clear(body);

    body.appendChild(buildStatList(it, it.stats));

    // 说明节点先攒着：下面拿到详情后会 clear(body) 重建，
    // 直接 append 会被抹掉，孤立库/丢库的提示就消失了。
    const notes = [];
    if (it.orphan) {
      notes.push(el("div", "note",
        "这个库文件不在索引里，任何流程都读不到它。可能是手工拷贝或索引损坏；" +
        "当前版本不提供删除入口，需手工处理。"));
    }
    if (it.missing) {
      notes.push(el("div", "note",
        "索引里有这个账号，但库文件不见了。放入备份文件后重启插件即可恢复。"));
    }
    notes.forEach((n) => body.appendChild(n));

    drawer.classList.remove("hidden");
    $("mask").classList.remove("hidden");

    if (!it.platform_id) {
      body.appendChild(el("div", "section-title", "常驻设定"));
      body.appendChild(el("div", "note", "索引缺失，无法确定账号 ID，读不到常驻内容。"));
      return;
    }

    body.appendChild(el("div", "section-title", "常驻设定（私聊场景）"));
    const holder = el("div", "note", "读取中…");
    body.appendChild(holder);

    try {
      const d = await apiGet("account", { platform_id: it.platform_id });
      // 详情接口一定拿到了真实统计，用它重建统计块。
      // 之前这里沿用了列表项的 it.stats，于是「点进去看详情」反而比
      // 「点取统计」信息更少——明明请求已经回来了。
      clear(body);
      body.appendChild(buildStatList(it, d.stats || it.stats));
      notes.forEach((n) => body.appendChild(n));

      body.appendChild(el("div", "section-title", "常驻设定（私聊场景）"));

      const labels = {
        address: "称呼", assistant: "助手设定", user: "用户信息",
        preference: "偏好与忌讳", style: "回复风格", rules: "规则与避雷",
      };
      const res = d.resident || {};
      Object.keys(labels).forEach((k) => {
        const box = el("div", "block");
        box.appendChild(el("div", "label", labels[k]));
        const txt = res[k] || "";
        box.appendChild(el("div", txt ? "text" : "text empty-text",
          txt || "（未设置）"));
        body.appendChild(box);
      });
      if (d.resident_hint) body.appendChild(el("div", "note", d.resident_hint));
    } catch (err) {
      holder.textContent = "读取失败：" + (err && err.message ? err.message : err);
    }
  }

  function closeDrawer() {
    $("drawer").classList.add("hidden");
    $("mask").classList.add("hidden");
  }

  // ---------------------------------------------------------------- 探测

  async function probeCurrentPage() {
    const ids = state.rows.map((r) => r.platform_id).filter(Boolean);
    if (!ids.length) { toast("本页没有可探测的账号"); return; }
    if (ids.length > 20) {
      toast("本页账号超过 20 个，请先用搜索缩小范围", true);
      return;
    }
    const btn = $("btn-probe");
    btn.disabled = true;
    btn.textContent = "读取中…";
    try {
      const d = await apiPost("probe", { platform_ids: ids });
      const got = {};
      (d.items || []).forEach((x) => { got[x.platform_id] = x; });
      state.rows = state.rows.map((r) => {
        const p = got[r.platform_id];
        return p ? Object.assign({}, r, { stats: p.stats || null }) : r;
      });
      const failed = (d.items || []).filter((x) => x.stats_error).length;
      toast("已读取 " + (d.count || 0) + " 个账号" +
            (failed ? "，" + failed + " 个失败" : ""));
      renderTable(state.rows);
      renderPager({ total: state.total, offset: state.offset,
                    limit: state.limit, items: state.rows });
    } catch (err) {
      toast("读取失败：" + (err && err.message ? err.message : err), true);
    } finally {
      btn.disabled = false;
      btn.textContent = "取统计";
    }
  }

  // ---------------------------------------------------------------- 事件

  function bind() {
    $("btn-refresh").addEventListener("click", () => { state.probed = false; load(); });
    $("btn-probe").addEventListener("click", probeCurrentPage);
    $("d-close").addEventListener("click", closeDrawer);
    $("mask").addEventListener("click", closeDrawer);

    $("btn-health").addEventListener("click", () => {
      loadHealth();
      toast("已刷新健康信息");
    });

    let timer = null;
    $("q").addEventListener("input", (e) => {
      clearTimeout(timer);
      const v = e.target.value;
      timer = setTimeout(() => {
        state.q = v; state.offset = 0; state.probed = false; load();
      }, 260);
    });

    $("sort").addEventListener("change", (e) => {
      state.sort = e.target.value; state.offset = 0; state.probed = false; load();
    });

    $("prev").addEventListener("click", () => {
      state.offset = Math.max(0, state.offset - state.limit);
      state.probed = false; load();
    });

    $("next").addEventListener("click", () => {
      if (state.offset + state.limit >= state.total) return;
      state.offset += state.limit; state.probed = false; load();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeDrawer();
    });
  }

  // ---------------------------------------------------------------- 启动

  function init() {
    if (!bridge) {
      $("banner").classList.remove("hidden");
      $("btn-refresh").disabled = true;
      $("btn-probe").disabled = true;
      return;
    }
    bind();
    load();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
