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

  /** 就地编辑：把 node 换成输入框，回车/失焦保存，Esc 取消。
   *
   *  用插入而非 replaceChild，是为了「取消」时能把原节点原样还回去——
   *  备注为空时原来显示的是账号 ID，取消后必须回到 ID，不能变成空白。
   */
  function inlineEdit(node, value, opts) {
    const o = opts || {};
    if (!node || node._editing) return;
    node._editing = true;

    const input = el("input", "input inline-input");
    input.value = value || "";
    input.placeholder = o.placeholder || "";
    if (o.maxlength) input.maxLength = o.maxlength;
    // 行点击会跳到详情页；点输入框不能跳。
    input.addEventListener("click", (e) => e.stopPropagation());
    input.addEventListener("mousedown", (e) => e.stopPropagation());

    node.classList.add("hidden");
    node.parentNode.insertBefore(input, node);
    input.focus();
    if (input.select) input.select();

    let done = false;
    const finish = (save) => {
      if (done) return;
      done = true;
      node._editing = false;
      const next = input.value;
      if (input.parentNode) input.parentNode.removeChild(input);
      node.classList.remove("hidden");
      if (save && o.onSave) o.onSave(next);
      else if (o.onCancel) o.onCancel();
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    });
    input.addEventListener("blur", () => finish(true));
  }

  /** 保存备注，并同步内存里的列表项与告警项（免得两处显示不一致）。 */
  async function saveNote(pid, value) {
    const r = await apiPost("notes", { platform_id: pid, note: value });
    const note = (r && r.note) || "";
    state.accounts.rows.forEach((row) => {
      if (row.platform_id === pid) row.note = note;
    });
    state.alerts.forEach((a) => {
      if (a.platform_id === pid) a.note = note;
    });
    return note;
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
      // 有备注就显示备注：告警条上写着一串适配器 id，等于让人自己去对表格。
      const who = el("span", "who", a.note || a.platform_id || "(未知账号)");
      who.title = a.platform_id || "";
      row.appendChild(who);
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
      const head = el("div", "acc-head");
      // 显示顺序：有备注就主显备注，没有就把 ID 顶上来——备注是别名，
      // 不是替代品，改完不该让人找不到这个账号到底叫什么。
      const nameEl = el("span", "acc-name" + (it.note ? "" : " unnamed"),
                        it.note || it.platform_id || "(索引缺失)");
      head.appendChild(nameEl);
      if (it.platform_id) {
        const btn = el("button", "note-btn", it.note ? "改" : "备注");
        btn.type = "button";
        btn.title = it.note ? "修改备注" : "起一个备注（只影响显示，不改账号 ID）";
        btn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          inlineEdit(nameEl, it.note || "", {
            maxlength: 40,
            placeholder: "例如：客服号（测试）",
            onSave: async (val) => {
              try {
                const note = await saveNote(it.platform_id, val);
                toast(note ? "备注已保存：" + note : "备注已清除");
              } catch (err) {
                toast("保存备注失败：" + errMsg(err), true);
              }
              renderTable(state.accounts.rows);
              renderAlerts(state.alerts);
            },
          });
        });
        head.appendChild(btn);
      }
      tdName.appendChild(head);
      // 有备注时第二行同时给 ID 与文件名：备注是给人看的，排障时还得看真身。
      tdName.appendChild(el(
        "div", "stem mono",
        it.note ? (it.platform_id + " · " + (it.file || it.stem))
                : (it.file || it.stem),
      ));
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

  /** 用系统文件管理器打开服务端目录。仅本机部署有效。 */
  async function openDir(which, label) {
    try {
      await apiPost("open-root", { which: which });
      toast("已打开" + label + "（在本机文件管理器中）");
    } catch (err) {
      toast("打开失败：" + errMsg(err), true);
    }
  }

  /** 备份全部账号库。结果给出文件名，并可直接打开备份目录拷走。 */
  async function runBackup() {
    const btn = $("btn-backup");
    btn.disabled = true;
    btn.textContent = "备份中…";
    try {
      const d = await apiPost("backup", {});
      const line = $("backup-result-line");
      line.hidden = false;
      $("backup-result").textContent =
        (d.message || "备份完成") + "  " + d.filename;
      toast("备份完成：" + d.bots + " 个账号库");
    } catch (err) {
      toast("备份失败：" + errMsg(err), true);
    } finally {
      btn.disabled = false;
      btn.textContent = "备份全部记忆";
    }
  }

  // ---------------------------------------------------------------- 恢复

  function fmtSize(n) {
    const v = Number(n) || 0;
    if (v < 1024) return v + " B";
    if (v < 1024 * 1024) return (v / 1024).toFixed(1) + " KB";
    return (v / 1024 / 1024).toFixed(1) + " MB";
  }

  function backupRow(item) {
    const row = el("div", "backup-row");
    row.appendChild(el("span", "name", item.filename));
    row.appendChild(el("span", "dim small",
      fullTime(item.created_at) + " · " + (item.bots || 0) + " 账号 · " +
      fmtSize(item.size_bytes)));
    const dl = el("button", "btn tiny", "下载");
    dl.addEventListener("click", () => downloadBackup(item.filename));
    row.appendChild(dl);
    if (item.kind === "restore-point") {
      const rb = el("button", "btn tiny danger", "回滚");
      rb.addEventListener("click", () => rollbackBackup(item.filename));
      row.appendChild(rb);
    } else if (item.kind === "backup") {
      const rs = el("button", "btn tiny", "恢复");
      rs.addEventListener("click", () => restoreBackup(item.filename));
      row.appendChild(rs);
    }
    return row;
  }

  async function openRestore() {
    const modal = $("restore-modal");
    const list = $("restore-list");
    clear(list);
    list.appendChild(el("p", "dim small", "读取中…"));
    modal.classList.remove("hidden");
    try {
      const d = await apiGet("backup/list");
      clear(list);
      if (!(d.backups || []).length && !(d.restore_points || []).length) {
        list.appendChild(el("p", "dim small",
          "还没有备份。先点「备份全部记忆」。"));
        return;
      }
      if ((d.backups || []).length) {
        list.appendChild(el("div", "backup-sec", "备份（可恢复）"));
        d.backups.forEach((it) => list.appendChild(backupRow(it)));
      }
      if ((d.restore_points || []).length) {
        list.appendChild(el("div", "backup-sec", "恢复前快照（可回滚）"));
        d.restore_points.forEach((it) => list.appendChild(backupRow(it)));
      }
    } catch (err) {
      clear(list);
      list.appendChild(el("p", "dim small", "读取备份列表失败：" + errMsg(err)));
    }
  }

  async function downloadBackup(name) {
    const bridge = getBridge();
    if (!bridge || typeof bridge.download !== "function") {
      toast("当前面板不支持下载，请点「打开备份目录」拷走", true);
      return;
    }
    try {
      await bridge.download("backup/download", { filename: name }, name);
      // 桌面端（Tauri WebView）不一定会弹出下载提示，文件可能已经安静地
      // 落在下载目录。给一句明确回执，并指一条一定能走通的路。
      toast("已发起下载：" + name + "。若没反应，请点「打开备份目录」");
    } catch (err) {
      toast("下载失败：" + errMsg(err) + "，可点「打开备份目录」拷走", true);
    }
  }

  // `window.prompt` 在面板里不可用：插件页面跑在沙箱 iframe 里，宿主只给了
  // `allow-scripts allow-forms allow-downloads`，没有 `allow-modals`——prompt
  // 会被静默忽略并返回 null，表现就是点「恢复」什么都不发生。所以确认区
  // 做成页面内元素，不依赖宿主弹窗。
  let confirmBox = null;

  function clearConfirm() {
    if (confirmBox && confirmBox.parentNode) {
      confirmBox.parentNode.removeChild(confirmBox);
    }
    confirmBox = null;
  }

  function askConfirm(endpoint, name, verb) {
    clearConfirm();
    const box = el("div", "backup-confirm");
    box.appendChild(el("p", "dim small",
      verb + "会覆盖当前全部账号数据：" + name + "。输入 RESTORE 确认："));
    const input = el("input", "input tiny-input");
    input.type = "text";
    input.placeholder = "RESTORE";
    input.autocomplete = "off";
    const ok = el("button", "btn tiny danger", verb);
    ok.disabled = true;
    const cancel = el("button", "btn tiny ghost", "取消");
    input.addEventListener("input", () => {
      ok.disabled = input.value.trim() !== "RESTORE";
    });
    cancel.addEventListener("click", clearConfirm);
    ok.addEventListener("click", () => {
      clearConfirm();
      applyBackup(endpoint, name, verb);
    });
    const line = el("div", "backup-confirm-line");
    line.appendChild(input);
    line.appendChild(ok);
    line.appendChild(cancel);
    box.appendChild(line);
    $("restore-list").appendChild(box);
    confirmBox = box;
    try {
      input.focus();
      box.scrollIntoView({ block: "nearest" });
    } catch (_) {
      /* 老 WebView 不支持 scrollIntoView 参数时忽略 */
    }
  }

  /** 恢复/回滚共用：确认后调接口，成功后刷新账号与健康信息。 */
  async function applyBackup(endpoint, name, verb) {
    try {
      const d = await apiPost(endpoint, { filename: name, confirm: "RESTORE" });
      $("restore-modal").classList.add("hidden");
      state.accounts.probed = false;
      await loadAccounts();
      await loadHealth(true);
      const need = (d.needs_reload || []).length > 0;
      toast(d.message || (verb + "完成"), need);
    } catch (err) {
      toast(verb + "失败：" + errMsg(err), true);
    }
  }

  const restoreBackup = (name) => askConfirm("backup/restore", name, "恢复");
  const rollbackBackup = (name) => askConfirm("backup/rollback", name, "回滚");

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
  // 当前页名。用来在切页时调用上一页的 `leave()`——向量页有 1s 轮询，
  // 不显式停掉的话离开页面后还在持续发请求。
  let currentPage = "";

  function parseHash() {
    const raw = (location.hash || "").replace(/^#\/?/, "");
    if (raw.startsWith("a/")) {
      return { name: "detail", pid: decodeURIComponent(raw.slice(2)) };
    }
    if (raw.startsWith("vector")) return { name: "vector" };
    if (raw.startsWith("models")) return { name: "models" };
    if (raw.startsWith("settings")) return { name: "settings" };
    return { name: "accounts" };
  }

  function showView(name) {
    ["accounts", "detail", "vector", "models", "settings"].forEach((key) => {
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
    if (currentPage && currentPage !== r.name) {
      const prev = pages[currentPage];
      if (prev && typeof prev.leave === "function") {
        try { prev.leave(); } catch (_) { /* 页面自己的清理失败不该挡住切页 */ }
      }
    }
    currentPage = r.name;
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
    $("btn-open-root").addEventListener("click", () => openDir("root", "库目录"));
    $("btn-backup").addEventListener("click", runBackup);
    $("backup-result").addEventListener("click", () => openDir("backups", "备份目录"));
    $("btn-restore").addEventListener("click", openRestore);
    $("restore-cancel").addEventListener("click",
      () => $("restore-modal").classList.add("hidden"));
    $("restore-open-dir").addEventListener("click",
      () => openDir("backups", "备份目录"));
    $("restore-modal").addEventListener("click", (e) => {
      if (e.target === $("restore-modal")) $("restore-modal").classList.add("hidden");
    });

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
    inlineEdit, saveNote, loadHealth,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
