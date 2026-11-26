/* 向量模型页：本机模型的下载 / 启停 / 日志，以及按账号维护向量库。

   三条约束（与 app.js 一致）：
   1. 不用 innerHTML 插数据：日志行、账号备注、错误信息都来自运行环境。
   2. 不猜状态：接口失败就显示错误，不沿用上一次的值。
   3. 轮询随页面进出启停：离开页面必须清掉定时器，否则后台一直在发请求。 */

(function () {
  "use strict";

  const F = window.Fleet;
  if (!F) return;
  const { apiGet, apiPost, el, clear, toast, num, errMsg, fullTime } = F;
  const $ = (id) => document.getElementById(id);

  const POLL_MS = 1000;
  const MAX_LOG_NODES = 600;

  const st = {
    timer: null,
    busy: false,
    seq: 0,
    accountsLoaded: false,
    entering: false,
  };

  const STATE_TEXT = {
    idle: "未安装",
    installed: "已下载，未启动",
    downloading: "下载中",
    verifying: "校验 / 解压中",
    starting: "启动中",
    ready: "运行中",
    stopped: "已停止",
    error: "出错",
  };

  function clock(ts) {
    const d = new Date((ts || 0) * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  // ---------------------------------------------------------------- 状态

  function rows(box, pairs) {
    clear(box);
    pairs.forEach(([k, v]) => {
      box.appendChild(el("dt", "", k));
      box.appendChild(el("dd", "", v));
    });
  }

  function fmtMB(bytes) {
    const v = Number(bytes) || 0;
    if (!v) return "—";
    return (v / 1024 / 1024).toFixed(1) + " MB";
  }

  function renderStatus(d) {
    const state = String(d.state || "idle");
    const ready = !!d.ready;
    const text = STATE_TEXT[state] || state;
    const port = d.port ? " · 端口 " + d.port : "";
    const pid = d.pid ? " · PID " + d.pid : "";
    const install = [
      d.runtime_installed ? "运行库 ✓" : "运行库 ✗",
      d.model_installed ? "模型 ✓" : "模型 ✗",
    ].join(" · ");
    const pairs = [
      ["运行状态", ready ? text + port + pid : text],
      ["模型", d.model || "—"],
      ["维度", d.dim ? String(d.dim) : "—"],
      ["安装", install],
      ["推理线程", num(d.threads)],
      ["目录", d.dir || "—"],
    ];
    if (d.last_error) pairs.push(["最近错误", d.last_error]);
    rows($("vec-status"), pairs);

    const mode = [];
    if (d.provider) mode.push("来源：" + d.provider);
    if (d.config_provider && d.config_provider !== d.provider) {
      mode.push("设置已改为 " + d.config_provider + "，重载插件后生效");
    }
    if (d.configured === false) mode.push("未启用（设置里把来源设为 local）");
    else if (d.vector_enabled) mode.push("已接入检索");
    else if (d.not_ready) mode.push(d.not_ready);
    $("vec-mode-hint").textContent = mode.join(" · ") || "—";

    const p = d.progress || {};
    const show = !!d.installing || state === "downloading" || state === "verifying";
    $("vec-progress").classList.toggle("hidden", !show);
    if (show) {
      const percent = Math.max(0, Math.min(100, Number(p.percent) || 0));
      $("vec-bar").style.width = percent + "%";
      const size = p.total ? "（" + fmtMB(p.done) + " / " + fmtMB(p.total) + "）" : "";
      $("vec-progress-text").textContent =
        (p.stage || "准备中") + " " + percent + "%" + size;
    }

    $("btn-vec-install").disabled = st.busy || !!d.installed || !!d.installing;
    $("btn-vec-start").disabled = st.busy || !d.installed || ready
      || state === "starting";
    $("btn-vec-stop").disabled = st.busy || (!ready && !d.pid);
    $("btn-vec-restart").disabled = st.busy || !d.installed;
  }

  async function loadStatus() {
    try {
      renderStatus(await apiGet("local-vector/status"));
    } catch (err) {
      rows($("vec-status"), [["运行状态", "读取失败：" + errMsg(err)]]);
    }
  }

  // ---------------------------------------------------------------- 日志

  function appendLogs(lines) {
    const box = $("vec-log");
    if (!lines || !lines.length) return;
    const atBottom =
      box.scrollTop + box.clientHeight >= box.scrollHeight - 24;
    lines.forEach((item) => {
      box.appendChild(el("div", "logline",
        clock(item.ts) + "  " + String(item.line || "")));
    });
    while (box.childElementCount > MAX_LOG_NODES) {
      box.removeChild(box.firstChild);
    }
    if (atBottom) box.scrollTop = box.scrollHeight;
    st.seq = lines[lines.length - 1].seq || st.seq;
  }

  async function loadLogs() {
    try {
      const d = await apiGet("local-vector/logs", { after: st.seq });
      appendLogs(d.lines);
    } catch (_) {
      /* 日志拉取失败不打扰用户：下一轮会再试 */
    }
  }

  // ---------------------------------------------------------------- 动作

  async function act(path, label, body) {
    if (st.busy) return;
    st.busy = true;
    loadStatus();
    try {
      const d = await apiPost(path, body || {});
      if (path === "local-vector/install") {
        toast(d.message || "已开始下载");
      } else if (d && d.last_error) {
        toast(label + "失败：" + d.last_error, true);
      } else {
        toast(label + "完成");
      }
    } catch (err) {
      toast(label + "失败：" + errMsg(err), true);
    } finally {
      st.busy = false;
      await loadStatus();
      await loadLogs();
    }
  }

  // ---------------------------------------------------------------- 账号

  async function loadAccounts() {
    const sel = $("vec-account");
    if (st.accountsLoaded) return;
    clear(sel);
    sel.appendChild(el("option", "", "读取账号中…"));
    try {
      const d = await apiGet("overview", { limit: 200, sort: "name" });
      const items = (d.items || []).filter((x) => x.platform_id);
      clear(sel);
      if (!items.length) {
        sel.appendChild(el("option", "", "还没有账号"));
        return;
      }
      items.forEach((it) => {
        const opt = el("option", "", it.note || it.platform_id);
        opt.value = it.platform_id;
        sel.appendChild(opt);
      });
      st.accountsLoaded = true;
    } catch (err) {
      clear(sel);
      sel.appendChild(el("option", "", "读取失败：" + errMsg(err)));
    }
  }

  function accountRows(d) {
    const mode = d.recall_mode === "vector" ? "向量版（只用语义）" : "经典版（只字面）";
    const pairs = [
      ["数据库模式", mode],
      ["向量状态", d.enabled ? "可用" : (d.not_ready || "未启用")],
      ["索引 / 活跃", num(d.indexed) + " / " + num(d.active)],
      ["待索引", num(d.pending)],
      ["最后同步", d.last_sync_ts ? fullTime(d.last_sync_ts) : "—"],
    ];
    if (d.layout !== "v2") {
      pairs.push(["布局", d.layout === "legacy"
        ? "v1 平铺布局（迁移后自动启用）" : "库文件缺失"]);
    }
    if (d.last_error) pairs.push(["最近错误", d.last_error]);
    rows($("vec-account-status"), pairs);
    $("btn-vec-rebuild").disabled = !(d.enabled && d.layout === "v2");
    $("btn-vec-clear").disabled = !(d.indexed > 0);
  }

  async function loadAccountStatus() {
    const pid = $("vec-account").value;
    const box = $("vec-account-status");
    if (!pid) return;
    rows(box, [["状态", "读取中…"]]);
    try {
      accountRows(await apiGet("vector/status", { platform_id: pid }));
    } catch (err) {
      rows(box, [["状态", "读取失败：" + errMsg(err)]]);
    }
  }

  async function accountAction(path, label) {
    const pid = $("vec-account").value;
    if (!pid) { toast("先选一个账号", true); return; }
    const out = $("vec-result");
    out.textContent = label + "中…";
    try {
      const d = await apiPost(path, { platform_id: pid });
      out.textContent = d.message || (d.error ? "失败：" + d.error : "完成");
      toast(d.error ? label + "失败：" + d.error : label + "完成", !!d.error);
      await loadAccountStatus();
    } catch (err) {
      out.textContent = label + "失败：" + errMsg(err);
      toast(label + "失败：" + errMsg(err), true);
    }
  }

  // ---------------------------------------------------------------- 轮询

  function tick() {
    if (document.hidden) return;
    loadStatus();
    loadLogs();
  }

  function startPolling() {
    stopPolling();
    st.timer = setInterval(tick, POLL_MS);
  }

  function stopPolling() {
    if (st.timer) clearInterval(st.timer);
    st.timer = null;
  }

  // ---------------------------------------------------------------- 绑定

  $("btn-vec-refresh").addEventListener("click", async () => {
    await loadStatus();
    await loadLogs();
    toast("已刷新");
  });
  $("btn-vec-install").addEventListener("click",
    () => act("local-vector/install", "下载"));
  $("btn-vec-start").addEventListener("click",
    () => act("local-vector/start", "启动"));
  $("btn-vec-stop").addEventListener("click",
    () => act("local-vector/stop", "停止"));
  $("btn-vec-restart").addEventListener("click",
    () => act("local-vector/restart", "重启"));
  $("btn-vec-load").addEventListener("click", loadAccountStatus);
  $("vec-account").addEventListener("change", loadAccountStatus);
  $("btn-vec-rebuild").addEventListener("click",
    () => accountAction("vector/rebuild", "重建索引"));
  $("btn-vec-clear").addEventListener("click", () => {
    if (!window.confirm("清空这个账号的向量索引？记忆本身不会删除，之后可以重建。")) {
      return;
    }
    accountAction("vector/clear", "清除索引");
  });

  F.pages.vector = {
    async enter() {
      if (st.entering) return;
      st.entering = true;
      try {
        await loadAccounts();
        await loadStatus();
        await loadLogs();
        if ($("vec-account").value) await loadAccountStatus();
        startPolling();
      } finally {
        st.entering = false;
      }
    },
    leave() {
      stopPolling();
    },
  };
})();
