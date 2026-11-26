/* 设置页。
   直接渲染 `_conf_schema.json` 的分组，写回同一份配置文件（与 AstrBot
   插件配置页是同一份）。标注「需重启」的项保存后要重载插件——不标的话，
   用户会以为「保存了没反应」是 bug。 */

(function () {
  "use strict";

  const F = window.Fleet;
  if (!F) return;
  const { apiGet, apiPost, el, clear, toast, num, errMsg } = F;
  const $ = (id) => document.getElementById(id);

  /** 每个控件的取值函数。类型由 schema 的 type 决定，不由值猜。 */
  function control(item) {
    const id = "cfg-" + item.key;
    const wrap = el("div", "cfg-item");
    const label = el("label", "cfg-label");
    label.setAttribute("for", id);
    label.appendChild(el("span", "cfg-key mono", item.key));
    if (item.needs_reload) label.appendChild(el("span", "badge warn", "需重启"));
    wrap.appendChild(label);

    if (item.description) wrap.appendChild(el("div", "hint", item.description));

    let input;
    if (item.type === "bool") {
      input = document.createElement("input");
      input.type = "checkbox";
      input.className = "checkbox";
      input.checked = !!item.value;
      input.id = id;
      input.dataset.kind = "bool";
    } else if (item.type === "list") {
      input = document.createElement("textarea");
      input.className = "input cfg-list";
      input.rows = 2;
      const v = item.value;
      input.value = Array.isArray(v) ? v.join(", ") : (v || "");
      input.id = id;
      input.dataset.kind = "list";
    } else if (item.type === "int" || item.type === "float" || item.type === "number") {
      input = document.createElement("input");
      input.type = "number";
      input.className = "input";
      if (item.type === "int") input.step = "1";
      else input.step = "any";
      if (item.min !== undefined && item.min !== null) input.min = item.min;
      if (item.max !== undefined && item.max !== null) input.max = item.max;
      input.value = item.value === undefined || item.value === null ? "" : item.value;
      input.id = id;
      input.dataset.kind = item.type;
    } else if (Array.isArray(item.options) && item.options.length) {
      input = document.createElement("select");
      input.className = "input";
      item.options.forEach((o) => {
        const opt = document.createElement("option");
        opt.value = String(o);
        opt.textContent = String(o);
        if (String(item.value) === String(o)) opt.selected = true;
        input.appendChild(opt);
      });
      input.id = id;
      input.dataset.kind = "string";
    } else {
      input = document.createElement("input");
      input.type = "text";
      input.className = "input";
      input.value = item.value === undefined || item.value === null ? "" : item.value;
      input.id = id;
      input.dataset.kind = "string";
    }

    input.dataset.key = item.key;
    wrap.appendChild(input);
    if (item.hint) wrap.appendChild(el("div", "hint dim", item.hint));
    return wrap;
  }

  // 向量相关参数在「向量模型」页渲染（见 vector.js）：来源、模型、索引状态
  // 与这些参数是同一件事，分两页会让人来回找。
  const VECTOR_GROUP = "向量检索";

  function render(groups) {
    const box = $("cfg-groups");
    clear(box);
    (groups || []).filter((g) => g.title !== VECTOR_GROUP).forEach((g) => {
      const panel = el("section", "panel pad");
      panel.appendChild(el("h2", "pane-title", g.title));
      const grid = el("div", "cfg-grid");
      g.items.forEach((it) => grid.appendChild(control(it)));
      panel.appendChild(grid);
      box.appendChild(panel);
    });
  }

  async function load() {
    try {
      const d = await apiGet("config");
      render(d.groups);
    } catch (err) {
      toast("读取配置失败：" + errMsg(err), true);
    }
  }

  /** 取控件当前值。传 root 只取那一块——设置页与向量页各存各的。 */
  function collect(root) {
    const values = {};
    (root || document).querySelectorAll("[data-key]").forEach((node) => {
      const kind = node.dataset.kind;
      if (kind === "bool") values[node.dataset.key] = node.checked;
      else values[node.dataset.key] = node.value;
    });
    return values;
  }

  async function save() {
    const buttons = [$("btn-cfg-save"), $("btn-cfg-save2")];
    buttons.forEach((b) => { b.disabled = true; });
    try {
      const d = await apiPost("config/save", { values: collect($("cfg-groups")) });
      const reload = d.needs_reload || [];
      toast(reload.length
        ? "已保存 " + d.saved.length + " 项；其中 " + reload.length + " 项需重载插件才生效"
        : "已保存 " + d.saved.length + " 项");
      await load();
    } catch (err) {
      toast("保存失败：" + errMsg(err), true);
    } finally {
      buttons.forEach((b) => { b.disabled = false; });
    }
  }

  $("btn-cfg-save").addEventListener("click", save);
  $("btn-cfg-save2").addEventListener("click", save);

  // 给「向量模型」页复用同一个控件渲染与取值，避免两页对同一份 schema
  // 各写一套解析（类型、选项、需重启标记都会漂）。
  F.configUi = { control, collect, VECTOR_GROUP };

  F.pages.settings = { enter() { return load(); } };
})();
