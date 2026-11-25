/* Fleet 账号台 · 启动 / 路由 / 账号索引页
   其他页面（detail / models / settings）各自一个文件，注册到 Fleet.pages。

   三条规矩：
   1. **不用 innerHTML 插数据。** platform_id、文件名、记忆正文都来自运行环境，
      其中记忆正文是用户输入。全部走 textContent / el()。
   2. **不猜数据。** 没探测过的账号，记忆数显示「—」而不是 0；接口失败时显示
      错误，不用上次的值冒充。
   3. **桥接对象每次现取。** AstrBot 把 bridge-sdk.js 注入在页面末尾，
      也就是本脚本之后，顶层缓存会拿到 null。 */

(function () {
  "use strict";

  const getBridge = () => window.AstrBotPluginView || window.AstrBotPluginPage || null;
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- 基础

  function apiGet(path, params) {
    const bridge = getBridge();
    if (!bridge) return Promise.reject(new Error("no-bridge"));
    return bridge.apiGet(path, params || {});
  }

  function apiPost(path, body) {
    const bridge = getBridge();
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

  /** 建元素。文本一律走 textContent，不做 HTML 解析。 */
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
    if (sec < 90) return "刚刚";
    if (sec < 3600) return Math.floor(sec / 60) + " 分钟前";
    if (sec < 86400) return Math.floor(sec / 3600) + " 小时前";
    const d = Math.floor(sec / 86400);
    if (d < 30) return d + " 天前";
    const dt = new Date(ts * 1000);
    return dt.getFullYear() + "-" + String(dt.getMonth() + 1).padStart(2, "0") +
           "-" + String(dt.getDate()).padStart(2, "0");
  }

  function fullTime(ts) {
    return ts ? new Date(ts * 1000).toLocaleString() : "—";
  }

  const num = (v) => (v === undefined || v === null) ? "—" : String(v);

  function errMsg(err) {
    const m = err && err.message ? String(err.message) : String(err || "未知错误");
    return m === "no-bridge" ? "面板桥接未就绪，请刷新页面" : m;
  }

  // ---------------------------------------------------------------- 状态

  const state = {
    accounts: { q: "", sort: "recent", limit: 50, offset: 0, total: 0,
                rows: [], probed: false, busy: false, alertsOpen: false },
    alerts: [],
  };

  // ---------------------------------------------------------------- 侧栏

  function setLine(id, value, cls, title) {
    const n = $(id);
    if (!n) return;
    n.textContent = num(value);
    n.className = cls || "";
    if (title !== undefined) n.title = title;
  }

  function renderSide(data) {
    const s = (data && data.summary) || {};
    setLine("s-accounts", s.accounts);
    setLine("s-bytes", s.total_text);
    setLine("s-stale", s.stale, s.stale ? "warn" : "");
    setLine("s-orphan", s.orphan, s.orphan ? "danger" : "");
    setLine("s-missing", s.missing, s.missing ? "danger" : "");
    if (data && data.version) $("s-version").textContent = data.version;
  }

  function renderHealth(h) {
    const cap = (h && h.capacity) || {};
    const open = $("s-open");
    open.textContent = num(cap.opened) + " / " + num(cap.max_open);
    open.className = cap.saturated ? "warn" : "";
    open.title = cap.saturated
      ? "已达上限：每次访问新账号都要驱逐一个连接"
      : "已打开 / 上限";
    $("s-root").textContent = h.root || "—";
    $("s-root").title = h.root || "";
    if (h.version) $("s-version").textContent = h.version;
  }

  // ---------------------------------------------------------------- 告警

  function renderAlerts(alerts) {
    const box = $("alerts");
    clear(box);
    state.alerts = alerts || [];
    const list = state.alerts;
    if (!list.length) return;

    // 首屏不能全被告警吃掉：error 全展示，warn 只放两条，其余折叠。
    // 100 个账号里静默 40 个是常态，全列出来等于把表格推到屏幕外。
    const errs = list.filter((a) => a.level === "error");
    const warns = list.filter((a) => a.level !== "error");
    const shown = state.accounts.alertsOpen ? list : errs.concat(warns.slice(0, 2));

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
        state.accounts.alertsOpen = true;
        renderAlerts(state.alerts);
      });
      box.appendChild(more);
    } else if (state.accounts.alertsOpen && list.length > 4) {
      const less = el("button", "alert-more", "收起");
      less.addEventListener("click", () => {
        state.accounts.alertsOpen = false;
        renderAlerts(state.alerts);
      });
      box.appendChild(less);
    }
  }

  // ---------------------------------------------------------------- 账号表

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
      const tr = el("tr", it.platform_id ? "clickable" : "");
      if (it.platform_id) {
        tr.addEventListener("click", () => {
          location.hash = "#/a/" + encodeURIComponent(it.platform_id);
        });
      }

      const tdName = el("td", "w-40p");
      tdName.appendChild(el("div", "pid mono", it.platform_id || "(索引缺失)"));
      tdName.appendChild(el("div", "stem mono", it.file || it.stem));
      tr.appendChild(tdName);

      tr.appendChild(el("td", "num w-90", it.size_text));
      tr.appendChild(el("td", "w-130", relTime(it.last_write_at)));

      const st = it.stats;
      const tdMem = el("td", "num w-80");
      tdMem.appendChild(el("span", st ? "" : "dash", st ? num(st.memories) : "—"));
      tr.appendChild(tdMem);

      const tdUn = el("td", "num w-80");
      const un = st ? st.unsummarized : null;
      tdUn.appendChild(el("span", st && un ? "badge warn" : "dash", st ? num(un) : "—"));
      tr.appendChild(tdUn);

      const tdSt = el("td", "w-110");
      tdSt.appendChild(statusCell(it));
      tr.appendChild(tdSt);

      const tdGo = el("td", "w-60 right");
      tdGo.appendChild(el("span", "go", it.platform_id ? "查看 →" : ""));
      tr.appendChild(tdGo);

      tb.appendChild(tr);
    });
  }

  function renderPager(data, rows) {
    const from = data.total === 0 ? 0 : data.offset + 1;
    const to = Math.min(data.offset + rows.length, data.total);
    $("page-info").textContent =
      "共 " + data.total + " 个账号，当前显示 " + from + "–" + to +
      (state.accounts.probed ? "（本页已取真实统计）" : "（记忆数未探测，点「取统计」）");
    $("prev").disabled = data.offset <= 0 || state.accounts.busy;
    $("next").disabled = data.offset + data.limit >= data.total || state.accounts.busy;
  }

  async function loadAccounts() {
    const st = state.accounts;
    if (st.busy) return;
    st.busy = true;
    $("btn-refresh").disabled = true;
    try {
      const data = await apiGet("overview", {
        q: st.q, sort: st.sort, limit: st.limit, offset: st.offset,
        probe: st.probed ? "1" : "0",
      });
      st.total = data.total || 0;
      st.rows = data.items || [];
      if (data.version) $("s-version").textContent = data.version;
      renderSide(data);
      renderAlerts(data.alerts);
      renderTable(st.rows);
      renderPager(data, st.rows);
    } catch (err) {
      toast("加载失败：" + errMsg(err), true);
    } finally {
      st.busy = false;
      $("btn-refresh").disabled = false;
    }
  }

  async function loadHealth(silent) {
    try {
      renderHealth(await apiGet("health"));
      if (!silent) toast("已刷新全局状态");
    } catch (err) {
      if (!silent) toast("读取健康信息失败：" + errMsg(err), true);
    }
  }

  async function probeCurrentPage() {
    const st = state.accounts;
    const ids = st.rows.map((r) => r.platform_id).filter(Boolean);
    if (!ids.length) { toast("本页没有可探测的账号"); return; }
    if (ids.length > 20) { toast("本页账号超过 20 个，请先用搜索缩小范围", true); return; }
    const btn = $("btn-probe");
    btn.disabled = true;
    btn.textContent = "读取中…";
    try {
      const d = await apiPost("probe", { platform_ids: ids });
      const got = {};
      (d.items || []).forEach((x) => { got[x.platform_id] = x; });
      st.rows = st.rows.map((r) => {
        const p = got[r.platform_id];
        return p ? Object.assign({}, r, { stats: p.stats || null }) : r;
      });
      st.probed = true;
      const failed = (d.items || []).filter((x) => x.stats_error).length;
      toast("已读取 " + (d.count || 0) + " 个账号" + (failed ? "，" + failed + " 个失败" : ""));
      renderTable(st.rows);
      renderPager({ total: st.total, offset: st.offset, limit: st.limit, items: st.rows }, st.rows);
    } catch (err) {
      toast("读取失败：" + errMsg(err), true);
    } finally {
      btn.disabled = false;
      btn.textContent = "取统计";
    }
  }

  // ---------------------------------------------------------------- 路由

  const pages = {};

  function parseHash() {
    const raw = (location.hash || "").replace(/^#\/?/, "");
    if (raw.startsWith("a/")) {
      return { name: "detail", pid: decodeURIComponent(raw.slice(2)) };
    }
    if (raw.startsWith("models")) return { name: "models" };
    if (raw.startsWith("settings")) return { name: "settings" };
    return { name: "accounts" };
  }

  function showView(name) {
    ["accounts", "detail", "models", "settings"].forEach((key) => {
      const node = $("view-" + key);
      if (node) node.classList.toggle("hidden", key !== name);
    });
    const navKey = name === "detail" ? "accounts" : name;
    document.querySelectorAll("#nav a").forEach((a) => {
      a.classList.toggle("active", a.dataset.nav === navKey);
    });
  }

  async function route() {
    const r = parseHash();
    showView(r.name);
    const page = pages[r.name];
    if (!page) return;
    try {
      await page.enter(r);
    } catch (err) {
      toast("页面加载失败：" + errMsg(err), true);
    }
  }

  // ---------------------------------------------------------------- 事件

  function bindAccounts() {
    $("btn-refresh").addEventListener("click", () => {
      state.accounts.probed = false;
      loadAccounts();
    });
    $("btn-health").addEventListener("click", () => loadHealth(false));
    $("btn-probe").addEventListener("click", probeCurrentPage);

    let timer = null;
    $("q").addEventListener("input", (e) => {
      clearTimeout(timer);
      const v = e.target.value;
      timer = setTimeout(() => {
        state.accounts.q = v;
        state.accounts.offset = 0;
        state.accounts.probed = false;
        loadAccounts();
      }, 260);
    });

    $("sort").addEventListener("change", (e) => {
      state.accounts.sort = e.target.value;
      state.accounts.offset = 0;
      state.accounts.probed = false;
      loadAccounts();
    });

    $("prev").addEventListener("click", () => {
      state.accounts.offset = Math.max(0, state.accounts.offset - state.accounts.limit);
      state.accounts.probed = false;
      loadAccounts();
    });

    $("next").addEventListener("click", () => {
      const st = state.accounts;
      if (st.offset + st.limit >= st.total) return;
      st.offset += st.limit;
      st.probed = false;
      loadAccounts();
    });

    pages.accounts = {
      enter() { return loadAccounts(); },
    };
  }

  // ---------------------------------------------------------------- 启动

  function waitBridge(timeoutMs) {
    return new Promise((resolve) => {
      const deadline = Date.now() + timeoutMs;
      (function tick() {
        const b = getBridge();
        if (b) { resolve(b); return; }
        if (Date.now() >= deadline) { resolve(null); return; }
        setTimeout(tick, 50);
      })();
    });
  }

  function showBanner() {
    const banner = $("banner");
    const framed = window.parent !== window;
    banner.textContent = framed
      ? "面板桥接脚本没有加载成功，接口不可用。请刷新页面；" +
        "仍然如此请确认插件已启用，并查看面板控制台是否有脚本加载错误。"
      : "本页需要从 AstrBot WebUI 的插件页进入（现在像是直接打开了页面地址），" +
        "接口因此不可用。";
    banner.classList.remove("hidden");
    document.querySelectorAll("#view-accounts .btn").forEach((b) => { b.disabled = true; });
  }

  async function init() {
    const bridge = await waitBridge(1500);
    if (!bridge) { showBanner(); return; }

    bindAccounts();
    window.addEventListener("hashchange", route);
    await route();
    loadHealth(true);
  }

  // 暴露给其他页面文件
  window.Fleet = {
    apiGet, apiPost, el, clear, toast, relTime, fullTime, num, errMsg, state, pages,
    loadHealth,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
