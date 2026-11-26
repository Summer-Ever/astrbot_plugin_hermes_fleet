/* 单账号下钻页。
   这是「精细化」的那一半：hermes_core 的记忆/常驻/工具三个 tab 在这里，
   但组织方式不同——先选作用域，再看该作用域里的记忆。

   为什么按作用域而不是「一张全表」：一个 bot 后面可能有几百个用户，
   全表翻页既慢又容易看串行（把 A 的偏好当成 B 的）。左栏选人，
   右栏看这个人的记忆，与「这个账号记住了谁什么」的心智一致。 */

(function () {
  "use strict";

  const F = window.Fleet;
  if (!F) return;
  const { apiGet, apiPost, el, clear, toast, relTime, fullTime, num, errMsg } = F;
  const $ = (id) => document.getElementById(id);

  const BLOCKS = [
    ["address", "怎么称呼用户", "用户希望被怎么叫。这一条必须在最前，否则会出现叫错人。"],
    ["assistant", "助手设定", "助手在这个场景里的自我设定（名字、身份、语气）。"],
    ["user", "用户信息", "稳定的客观信息：年龄、城市、职业、家庭成员等。"],
    ["preference", "偏好与忌讳", "喜欢什么、讨厌什么、过敏与禁忌。"],
    ["style", "回复风格", "希望怎样被回复：长度、语气、格式。"],
    ["rules", "规则与避雷", "明确要求「不要做什么」。"],
  ];

  const KINDS = ["fact", "preference", "profile", "summary", "other"];
  const KIND_LABEL = {
    fact: "事实", preference: "偏好", profile: "画像", summary: "摘要", other: "其他",
  };
  const VIS_LABEL = { private: "私有", group: "本群", public: "公开" };

  const st = {
    pid: "",
    note: "",
    file: "",
    stats: null,
    scopes: [],
    scope: null,      // 当前选中的作用域 {scope, owner_id, group_id, count}
    items: [],
    selected: new Set(),
    tab: "memories",
    q: "",
    after: 0,
    busy: false,
    resOwner: "",
    resScope: "private",
    loadedTabs: {},
    importFile: null,
    overrides: {},
  };

  // 单份导入文件的硬上限，与后端 MAX_EXPORT_BYTES 保持一致（100MB）。
  // 前端先挡一道：100MB 的文本走 FileReader + JSON 桥接会很慢，
  // 与其传到后端再被拒，不如选文件时立刻提示。
  const IMPORT_MAX_BYTES = 100 * 1024 * 1024;

  // ---------------------------------------------------------------- 工具

  function statOf(key, fallback) {
    const s = st.stats || {};
    return s[key] === undefined ? fallback : s[key];
  }

  function scopeLabel(s) {
    if (!s) return "—";
    if (s.scope === "global") return "全局";
    if (s.scope === "group") return "群 " + (s.group_id || "?");
    const who = s.owner_id || "(空)";
    return who;
  }

  function scopeKey(s) {
    return [s.scope, s.owner_id || "", s.group_id || ""].join("|");
  }

  // ---------------------------------------------------------------- 头部

  function renderHead() {
    $("d-pid").textContent = st.pid;
    // 有备注就主显备注（人认名字不认适配器 id），ID 留在面包屑和下面那行。
    $("d-title").textContent = st.note || st.pid;
    $("d-file").textContent = st.note ? (st.pid + " · " + (st.file || "")) : (st.file || "");

    const box = $("d-kpis");
    clear(box);
    const s = st.stats || {};
    [
      ["归档记忆", num(s.memories)],
      ["常驻词条", num(s.resident)],
      ["消息总数", num(s.messages)],
      ["未总结", num(s.unsummarized)],
    ].forEach(([k, v]) => {
      const cell = el("div", "kpi");
      cell.appendChild(el("b", "", v));
      cell.appendChild(el("span", "", k));
      box.appendChild(cell);
    });

    const info = $("d-info");
    clear(info);
    [
      ["备注", st.note || "（未设置）"],
      ["账号 ID", st.pid],
      ["库文件", st.file || "—"],
      ["记忆条数", num(s.memories)],
      ["常驻词条", num(s.resident)],
      ["消息总数", num(s.messages)],
      ["未总结消息", num(s.unsummarized)],
    ].forEach(([k, v]) => {
      info.appendChild(el("dt", "", k));
      info.appendChild(el("dd", "mono", v));
    });
  }

  // ---------------------------------------------------------------- 作用域

  function renderScopes() {
    const box = $("scope-list");
    clear(box);
    if (!st.scopes.length) {
      box.appendChild(el("div", "empty small", "这个账号还没有任何记忆。"));
      return;
    }
    st.scopes.forEach((s) => {
      const row = el("button", "scope" + (st.scope && scopeKey(st.scope) === scopeKey(s) ? " active" : ""));
      row.appendChild(el("span", "scope-name mono", scopeLabel(s)));
      row.appendChild(el("span", "scope-kind", s.scope === "user" ? "按人" : (s.scope === "group" ? "按群" : "全局")));
      row.appendChild(el("b", "scope-count", s.count));
      row.addEventListener("click", () => {
        st.scope = s;
        st.q = "";
        $("mem-q").value = "";
        renderScopes();
        loadMemories(true);
      });
      box.appendChild(row);
    });
  }

  async function loadScopes() {
    const d = await apiGet("scopes", { platform_id: st.pid });
    st.scopes = d.scopes || [];
    st.stats = d.stats || st.stats;
    if (!st.scope && st.scopes.length) st.scope = st.scopes[0];
    // 选中的作用域可能因为删除而消失
    if (st.scope) {
      const found = st.scopes.find((s) => scopeKey(s) === scopeKey(st.scope));
      st.scope = found || st.scopes[0] || null;
    }
    renderScopes();
    renderHead();
  }

  // ---------------------------------------------------------------- 记忆列表

  function memCard(it) {
    const card = el("div", "mem" + (st.selected.has(it.id) ? " selected" : ""));

    const head = el("div", "mem-head");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "checkbox";
    cb.checked = st.selected.has(it.id);
    cb.addEventListener("change", () => {
      if (cb.checked) st.selected.add(it.id); else st.selected.delete(it.id);
      card.classList.toggle("selected", cb.checked);
      syncBulk();
    });
    head.appendChild(cb);
    head.appendChild(el("span", "mem-id mono", "#" + it.id));
    head.appendChild(el("span", "badge", KIND_LABEL[it.kind] || it.kind));
    head.appendChild(el("span", "badge accent", VIS_LABEL[it.visibility] || it.visibility));
    if (it.hits) head.appendChild(el("span", "badge", "命中 " + it.hits));
    head.appendChild(el("span", "grow"));
    head.appendChild(el("span", "dim small", relTime(it.ts)));
    card.appendChild(head);

    const body = el("div", "mem-body", it.content);
    body.title = fullTime(it.ts);
    card.appendChild(body);

    const foot = el("div", "mem-foot");
    foot.appendChild(el("span", "dim small",
      it.chars + " 字" + (it.updated_ts ? " · 改于 " + relTime(it.updated_ts) : "")));
    foot.appendChild(el("span", "grow"));
    const edit = el("button", "btn tiny ghost", "编辑");
    edit.addEventListener("click", () => startEdit(card, it));
    foot.appendChild(edit);
    const del = el("button", "btn tiny danger", "删除");
    del.addEventListener("click", () => removeOne(it));
    foot.appendChild(del);
    card.appendChild(foot);

    return card;
  }

  function startEdit(card, it) {
    clear(card);
    const ta = document.createElement("textarea");
    ta.className = "input mem-edit";
    ta.value = it.content;
    card.appendChild(ta);

    const foot = el("div", "mem-foot");
    const kind = document.createElement("select");
    kind.className = "input tiny-input";
    KINDS.forEach((k) => {
      const o = document.createElement("option");
      o.value = k;
      o.textContent = KIND_LABEL[k];
      if (k === it.kind) o.selected = true;
      kind.appendChild(o);
    });
    foot.appendChild(kind);
    foot.appendChild(el("span", "grow"));

    const save = el("button", "btn tiny primary", "保存");
    save.addEventListener("click", async () => {
      save.disabled = true;
      try {
        const d = await apiPost("memories/update", {
          platform_id: st.pid, id: it.id, content: ta.value, kind: kind.value,
        });
        const idx = st.items.findIndex((x) => x.id === it.id);
        if (idx >= 0 && d.item) st.items[idx] = d.item;
        renderMemories();
        toast("已保存 #" + it.id);
      } catch (err) {
        toast("保存失败：" + errMsg(err), true);
        save.disabled = false;
      }
    });
    const cancel = el("button", "btn tiny ghost", "取消");
    cancel.addEventListener("click", renderMemories);
    foot.appendChild(save);
    foot.appendChild(cancel);
    card.appendChild(foot);
    ta.focus();
  }

  async function removeOne(it) {
    if (!confirm("删除 #" + it.id + "？\n\n软删除：不会物理消失，可回溯。")) return;
    try {
      await apiPost("memories/delete", { platform_id: st.pid, ids: [it.id] });
      st.items = st.items.filter((x) => x.id !== it.id);
      st.selected.delete(it.id);
      renderMemories();
      await loadScopes();
      toast("已删除 #" + it.id);
    } catch (err) {
      toast("删除失败：" + errMsg(err), true);
    }
  }

  function renderMemories() {
    const box = $("mem-list");
    clear(box);
    if (!st.scope) {
      box.appendChild(el("div", "empty", "先在左侧选择一个作用域。"));
      $("mem-info").textContent = "—";
      syncBulk();
      return;
    }
    if (!st.items.length) {
      box.appendChild(el("div", "empty", st.q ? "没有匹配的记忆。" : "这个作用域还没有记忆。"));
    }
    st.items.forEach((it) => box.appendChild(memCard(it)));

    $("mem-info").textContent = st.q
      ? "检索到 " + st.items.length + " 条（按相关度）"
      : "已加载 " + st.items.length + " 条 / 该作用域共 " + (st.scope ? st.scope.count : 0) + " 条";
    $("mem-more").classList.toggle("hidden", !!st.q || st.items.length >= (st.scope ? st.scope.count : 0));
    syncBulk();
  }

  function syncBulk() {
    const n = st.selected.size;
    $("mem-del").disabled = n === 0;
    $("mem-merge").disabled = n < 2;
    $("mem-del").textContent = n ? "删除选中（" + n + "）" : "删除选中";
    $("mem-merge").textContent = n >= 2 ? "合并选中（" + n + "）" : "合并选中";
  }

  async function loadMemories(reset) {
    if (!st.scope) { renderMemories(); return; }
    if (st.busy) return;
    st.busy = true;
    try {
      const params = {
        platform_id: st.pid,
        scope: st.scope.scope,
        owner_id: st.scope.owner_id || "",
        group_id: st.scope.group_id || "",
        limit: 50,
      };
      if (st.q) params.q = st.q;
      else params.after_id = reset ? 0 : st.after;

      const d = await apiGet("memories", params);
      const items = d.items || [];
      st.items = reset ? items : st.items.concat(items);
      st.after = d.next_after_id || 0;
      if (reset) st.selected.clear();
      renderMemories();
    } catch (err) {
      toast("读取记忆失败：" + errMsg(err), true);
    } finally {
      st.busy = false;
    }
  }

  // ---------------------------------------------------------------- 合并

  function openMerge() {
    const chosen = st.items.filter((x) => st.selected.has(x.id));
    if (chosen.length < 2) return;

    const mask = el("div", "mask");
    const box = el("div", "modal");
    box.appendChild(el("h3", "", "合并 " + chosen.length + " 条记忆"));
    box.appendChild(el("p", "hint", "选一条作为保留的内容；其余会被软删（可回溯）。"));

    const list = el("div", "merge-list");
    let target = chosen[0].id;
    chosen.forEach((it) => {
      const row = el("label", "merge-row");
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "merge-target";
      radio.value = String(it.id);
      radio.checked = it.id === target;
      radio.addEventListener("change", () => { target = it.id; });
      row.appendChild(radio);
      row.appendChild(el("span", "mono", "#" + it.id));
      row.appendChild(el("span", "merge-text", it.content));
      list.appendChild(row);
    });
    box.appendChild(list);

    const foot = el("div", "modal-foot");
    const cancel = el("button", "btn ghost", "取消");
    const ok = el("button", "btn primary", "确认合并");
    ok.addEventListener("click", async () => {
      ok.disabled = true;
      try {
        const sources = chosen.map((x) => x.id).filter((id) => id !== target);
        await apiPost("memories/merge", {
          platform_id: st.pid, source_ids: sources, into_id: target,
        });
        close();
        st.selected.clear();
        await loadMemories(true);
        await loadScopes();
        toast("已合并 " + sources.length + " 条到 #" + target);
      } catch (err) {
        toast("合并失败：" + errMsg(err), true);
        ok.disabled = false;
      }
    });
    cancel.addEventListener("click", close);
    foot.appendChild(cancel);
    foot.appendChild(ok);
    box.appendChild(foot);

    function close() {
      document.body.removeChild(mask);
    }
    mask.appendChild(box);
    mask.addEventListener("click", (e) => { if (e.target === mask) close(); });
    document.body.appendChild(mask);
  }

  async function removeSelected() {
    const ids = Array.from(st.selected);
    if (!ids.length) return;
    if (!confirm("删除选中的 " + ids.length + " 条？\n\n软删除：不会物理消失，可回溯。")) return;
    try {
      const d = await apiPost("memories/delete", { platform_id: st.pid, ids });
      st.selected.clear();
      await loadMemories(true);
      await loadScopes();
      toast("已删除 " + (d.deleted || 0) + " 条");
    } catch (err) {
      toast("删除失败：" + errMsg(err), true);
    }
  }

  // ---------------------------------------------------------------- 常驻

  function ownersFromScopes() {
    const seen = new Set();
    const out = [];
    st.scopes.forEach((s) => {
      if (s.owner_id && !seen.has(s.owner_id)) {
        seen.add(s.owner_id);
        out.push(s.owner_id);
      }
    });
    return out;
  }

  function groupsFromScopes() {
    const seen = new Set();
    const out = [];
    st.scopes.forEach((s) => {
      if (s.group_id && !seen.has(s.group_id)) {
        seen.add(s.group_id);
        out.push(s.group_id);
      }
    });
    return out;
  }

  function renderResidentSelectors() {
    const ownerSel = $("res-owner");
    const scopeSel = $("res-scope");
    const owners = ownersFromScopes();
    const keepOwner = st.resOwner;

    clear(ownerSel);
    owners.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o;
      opt.textContent = o;
      ownerSel.appendChild(opt);
    });
    if (!owners.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "（该账号还没有记忆，先在上面选一个作用域）";
      ownerSel.appendChild(opt);
    }
    st.resOwner = owners.includes(keepOwner) ? keepOwner : (owners[0] || "");
    ownerSel.value = st.resOwner;

    clear(scopeSel);
    const priv = document.createElement("option");
    priv.value = "private";
    priv.textContent = "私聊";
    scopeSel.appendChild(priv);
    groupsFromScopes().forEach((g) => {
      const opt = document.createElement("option");
      opt.value = "group:" + g;
      opt.textContent = "群 " + g;
      scopeSel.appendChild(opt);
    });
    if (st.resScope !== "private" && !groupsFromScopes().includes(st.resScope.slice(6))) {
      st.resScope = "private";
    }
    scopeSel.value = st.resScope;
  }

  function renderResident(values) {
    const grid = $("res-grid");
    clear(grid);
    BLOCKS.forEach(([key, label, hint]) => {
      const box = el("div", "res-block");
      const head = el("div", "res-head");
      head.appendChild(el("b", "", label));
      head.appendChild(el("span", "hint", hint));
      box.appendChild(head);

      const ta = document.createElement("textarea");
      ta.className = "input res-text";
      ta.dataset.block = key;
      ta.value = (values && values[key]) || "";
      ta.placeholder = "（未设置）";
      box.appendChild(ta);
      grid.appendChild(box);
    });
  }

  async function loadResident() {
    renderResidentSelectors();
    if (!st.resOwner) {
      renderResident({});
      $("res-hint").textContent = "这个账号还没有任何记忆，无法确定用户，常驻无从读取。";
      return;
    }
    try {
      const d = await apiGet("resident", {
        platform_id: st.pid, owner_id: st.resOwner, scope: st.resScope,
      });
      renderResident(d.resident || d.values || {});
      $("res-hint").textContent = st.resOwner + " · " +
        (st.resScope === "private" ? "私聊" : "群 " + st.resScope.slice(6));
    } catch (err) {
      // 读不到就显示空表单，而不是沿用上一次的内容——
      // 管理员接着编辑再保存，会把 A 的常驻写到 B 名下。
      toast("读取常驻失败：" + errMsg(err), true);
      renderResident({});
    }
  }

  async function saveResident() {
    if (!st.resOwner) { toast("没有可保存的用户", true); return; }
    const areas = Array.from(document.querySelectorAll("#res-grid .res-text"));
    const btn = $("btn-res-save");
    btn.disabled = true;
    btn.textContent = "保存中…";
    let ok = 0;
    let bad = 0;
    for (const ta of areas) {
      try {
        await apiPost("resident/save", {
          platform_id: st.pid,
          owner_id: st.resOwner,
          scope: st.resScope,
          block: ta.dataset.block,
          text: ta.value,
        });
        ok += 1;
      } catch (err) {
        bad += 1;
        toast("保存「" + ta.dataset.block + "」失败：" + errMsg(err), true);
      }
    }
    btn.disabled = false;
    btn.textContent = "保存全部改动";
    if (ok) toast("已保存 " + ok + " 个词条" + (bad ? "，" + bad + " 个失败" : ""));
    await loadScopes();
  }

  // ---------------------------------------------------------------- 消息

  async function loadMessages() {
    const box = $("msg-list");
    clear(box);
    box.appendChild(el("div", "empty", "读取中…"));
    try {
      const d = await apiGet("messages", { platform_id: st.pid, limit: 50 });
      clear(box);
      $("msg-hint").textContent = "未总结消息 " + (d.total || 0) + " 条，涉及 " +
        ((d.sessions || []).length) + " 个会话。";
      if (!d.sessions || !d.sessions.length) {
        box.appendChild(el("div", "empty", "没有待总结的消息。"));
        return;
      }
      d.sessions.forEach((s) => {
        const sec = el("div", "msg-session");
        const head = el("div", "msg-session-head");
        head.appendChild(el("b", "mono", s.session_id));
        head.appendChild(el("span", "badge warn", s.count + " 条"));
        head.appendChild(el("span", "dim small", "最后 " + relTime(s.last_ts)));
        sec.appendChild(head);
        if (s.count > s.items.length) {
          sec.appendChild(el("div", "hint", "只列最近 " + s.items.length + " 条"));
        }
        s.items.forEach((m) => {
          const row = el("div", "msg " + (m.role === "assistant" ? "bot" : "user"));
          row.appendChild(el("span", "msg-role", m.role === "assistant" ? "BOT" : "用户"));
          row.appendChild(el("span", "msg-text", m.content));
          sec.appendChild(row);
        });
        box.appendChild(sec);
      });
    } catch (err) {
      clear(box);
      box.appendChild(el("div", "empty", "读取失败：" + errMsg(err)));
    }
  }

  // ---------------------------------------------------------------- 操作

  async function runSummarize() {
    const btn = $("btn-summarize");
    btn.disabled = true;
    btn.textContent = "总结中…";
    const out = $("summarize-result");
    out.textContent = "正在调用模型…";
    try {
      const d = await apiPost("actions/summarize", { platform_id: st.pid });
      out.textContent = d.result || "完成";
      toast("总结完成：" + (d.result || "").slice(0, 40));
      await loadScopes();
      if (st.tab === "messages") loadMessages();
    } catch (err) {
      out.textContent = "失败：" + errMsg(err);
      toast("总结失败：" + errMsg(err), true);
    } finally {
      btn.disabled = false;
      btn.textContent = "开始总结";
    }
  }

  // ---------------------------------------------------------------- 导入记忆

  /** 选文件：只记住 File 对象，不读内容。真正读取在点击导入时做。
   *  这样选错文件不会白读几十 MB，且能立刻把 100MB 超限挡在前面。 */
  function onImportPick(e) {
    const file = e.target.files && e.target.files[0];
    const btn = $("btn-import");
    const out = $("import-result");
    st.importFile = null;
    if (!file) {
      btn.disabled = true;
      btn.textContent = "选择文档后导入";
      return;
    }
    if (file.size > IMPORT_MAX_BYTES) {
      out.textContent = "文件超过 100MB，拒绝导入。";
      e.target.value = "";
      btn.disabled = true;
      return;
    }
    st.importFile = file;
    btn.disabled = false;
    btn.textContent = "导入「" + file.name + "」";
    out.textContent = "已选择 " + file.name + "（" + file.size + " 字节），点按钮确认导入。";
  }

  /** 读成 UTF-8 文本。用 FileReader 而非 file.text()，兼容老一点的浏览器。 */
  function readAsText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error("读取文件失败"));
      reader.readAsText(file, "utf-8");
    });
  }

  async function runImport() {
    if (!st.importFile) { toast("请先选择记忆配置文档", true); return; }
    const btn = $("btn-import");
    const out = $("import-result");
    const file = st.importFile;
    const owner = $("import-owner").value.trim();
    // 二次确认：常驻是整段覆盖，不可撤销，必须让管理员明确知道。
    if (!confirm("把「" + file.name + "」导入到账号 " + st.pid +
                 "？\n\n常驻记忆会按词条整段覆盖，归档记忆会追加。" +
                 (owner ? "\n目标用户：" + owner : "\n目标用户：文档记录的原始用户"))) {
      return;
    }
    btn.disabled = true;
    btn.textContent = "读取中…";
    out.textContent = "正在读取文件…";
    let text;
    try {
      text = await readAsText(file);
    } catch (err) {
      out.textContent = "读取失败：" + errMsg(err);
      toast("读取文件失败：" + errMsg(err), true);
      btn.disabled = false;
      btn.textContent = "导入「" + file.name + "」";
      return;
    }
    if (new Blob([text]).size > IMPORT_MAX_BYTES) {
      out.textContent = "文本超过 100MB，拒绝导入。";
      btn.disabled = false;
      btn.textContent = "导入「" + file.name + "」";
      return;
    }
    btn.textContent = "导入中…";
    out.textContent = "正在写入记忆库…";
    try {
      const d = await apiPost("memory/import", {
        platform_id: st.pid,
        owner_id: owner,
        text: text,
      });
      out.textContent = d.message || "导入完成。";
      toast("记忆已导入");
      // 导入后刷新：作用域条数、记忆列表都可能变了。
      await loadScopes();
      if (st.scope) await loadMemories(true);
    } catch (err) {
      out.textContent = "导入失败：" + errMsg(err);
      toast("导入失败：" + errMsg(err), true);
    } finally {
      btn.disabled = false;
      btn.textContent = "导入「" + file.name + "」";
    }
  }

  // ---------------------------------------------------------------- 开关

  // 只列「策略」类开关：模型链、并发、间隔属于全局资源，按账号拆开会让
  // 「为什么这个号半夜在跑」无法归因，所以页面上也不给这些项。
  const OV_BOOLS = [
    ["capture_messages", "记录对话"],
    ["summarize_enabled", "后台总结"],
    ["consolidate_enabled", "记忆整理"],
    ["group_private_memory", "群内引用私聊记忆"],
  ];
  const OV_STATES = [["inherit", "跟随全局"], ["on", "开启"], ["off", "关闭"]];

  function renderOverrides() {
    const box = $("ov-form");
    clear(box);
    OV_BOOLS.forEach(([key, label]) => {
      const row = el("div", "ov-row");
      row.appendChild(el("span", "ov-label", label));
      const sel = el("select", "input");
      sel.id = "ov-" + key;
      OV_STATES.forEach(([value, text]) => {
        const opt = el("option", "", text);
        opt.value = value;
        sel.appendChild(opt);
      });
      const cur = st.overrides[key];
      sel.value = cur === undefined ? "inherit" : (cur ? "on" : "off");
      row.appendChild(sel);
      box.appendChild(row);
    });

    const row = el("div", "ov-row");
    row.appendChild(el("span", "ov-label", "检索条数 recall_k（1–20）"));
    const rk = el("input", "input");
    rk.id = "ov-recall_k";
    rk.type = "number";
    rk.min = "1";
    rk.max = "20";
    rk.placeholder = "跟随全局";
    if (st.overrides.recall_k !== undefined) rk.value = String(st.overrides.recall_k);
    row.appendChild(rk);
    box.appendChild(row);
  }

  async function saveOverrides() {
    const updates = {};
    OV_BOOLS.forEach(([key]) => {
      const value = $("ov-" + key).value;
      updates[key] = value === "inherit" ? null : value === "on";
    });
    const raw = $("ov-recall_k").value.trim();
    if (raw === "") {
      updates.recall_k = null;
    } else {
      const n = Number(raw);
      if (!Number.isInteger(n) || n < 1 || n > 20) {
        toast("检索条数必须是 1–20 的整数", true);
        return;
      }
      updates.recall_k = n;
    }
    const btn = $("btn-ov-save");
    btn.disabled = true;
    try {
      const d = await apiPost("account-overrides",
        { platform_id: st.pid, overrides: updates });
      st.overrides = d.overrides || {};
      renderOverrides();
      toast("开关已保存，从下一条消息起生效");
    } catch (err) {
      toast("保存开关失败：" + errMsg(err), true);
    } finally {
      btn.disabled = false;
    }
  }

  // ---------------------------------------------------------------- tab

  function switchTab(name) {
    st.tab = name;
    document.querySelectorAll("#d-tabs .tab").forEach((b) => {
      b.classList.toggle("active", b.dataset.tab === name);
    });
    ["memories", "resident", "messages", "actions", "overrides"].forEach((k) => {
      $("pane-" + k).classList.toggle("hidden", k !== name);
    });
    if (name === "overrides") renderOverrides();
    if (name === "resident" && !st.loadedTabs.resident) {
      st.loadedTabs.resident = true;
      loadResident();
    }
    if (name === "messages") loadMessages();
  }

  // ---------------------------------------------------------------- 进入

  async function enter(r) {
    const pid = r.pid;
    const changed = pid !== st.pid;
    st.pid = pid;
    if (changed) {
      st.stats = null;
      st.scopes = [];
      st.scope = null;
      st.items = [];
      st.selected.clear();
      st.q = "";
      st.after = 0;
      st.loadedTabs = {};
      st.resOwner = "";
      st.resScope = "private";
      st.importFile = null;
      st.overrides = {};
      $("mem-q").value = "";
      $("import-file").value = "";
      $("import-owner").value = "";
      $("import-result").textContent = "";
      $("btn-import").disabled = true;
      $("btn-import").textContent = "选择文档后导入";
      clear($("mem-list"));
      renderHead();
      renderMemories();
    }

    // 详情接口给 stats 与文件名（列表里可能有，但直接进来时要现取）
    try {
      const d = await apiGet("account", { platform_id: pid });
      st.stats = d.stats || null;
      st.note = d.note || "";
      st.overrides = d.overrides || {};
      st.file = (d.file || "").replace(/\.db$/, "") || st.file || "";
      renderHead();
    } catch (err) {
      toast("读取账号详情失败：" + errMsg(err), true);
    }

    await loadScopes();
    if (st.scope) await loadMemories(true);
    switchTab(st.tab || "memories");
  }

  // ---------------------------------------------------------------- 绑定

  function bind() {
    $("d-tabs").addEventListener("click", (e) => {
      const btn = e.target.closest(".tab");
      if (btn) switchTab(btn.dataset.tab);
    });

    $("btn-scopes").addEventListener("click", () => { loadScopes(); });
    $("btn-res-load").addEventListener("click", loadResident);
    $("btn-res-save").addEventListener("click", saveResident);
    $("btn-msg").addEventListener("click", loadMessages);
    $("btn-summarize").addEventListener("click", runSummarize);
    $("btn-ov-save").addEventListener("click", saveOverrides);
    $("import-file").addEventListener("change", onImportPick);
    $("btn-import").addEventListener("click", runImport);

    // 备注就在标题上改：改完不用回列表确认，标题当场变。
    $("d-note-edit").addEventListener("click", () => {
      const title = $("d-title");
      F.inlineEdit(title, st.note, {
        maxlength: 40,
        placeholder: "例如：客服号（测试）",
        onSave: async (val) => {
          try {
            st.note = await F.saveNote(st.pid, val);
            toast(st.note ? "备注已保存：" + st.note : "备注已清除");
          } catch (err) {
            toast("保存备注失败：" + errMsg(err), true);
          }
          renderHead();
        },
      });
    });

    $("res-owner").addEventListener("change", (e) => {
      st.resOwner = e.target.value;
      loadResident();
    });
    $("res-scope").addEventListener("change", (e) => {
      st.resScope = e.target.value;
      loadResident();
    });

    $("mem-selall").addEventListener("click", () => {
      st.items.forEach((it) => st.selected.add(it.id));
      renderMemories();
    });
    $("mem-merge").addEventListener("click", openMerge);
    $("mem-del").addEventListener("click", removeSelected);
    $("mem-clear").addEventListener("click", () => {
      st.q = "";
      $("mem-q").value = "";
      loadMemories(true);
    });
    $("mem-more").addEventListener("click", () => loadMemories(false));

    let timer = null;
    $("mem-q").addEventListener("input", (e) => {
      clearTimeout(timer);
      const v = e.target.value.trim();
      timer = setTimeout(() => {
        st.q = v;
        st.after = 0;
        loadMemories(true);
      }, 300);
    });
  }

  bind();
  F.pages.detail = { enter };
})();
