// Toggle buttons that show or hide index rows by the value of one column, and a
// switch that hides fixed entries (rows whose ID cell holds a .bb-fixed marker; the
// exporter adds it when the entry's head run no longer reproduces the defect).
document$.subscribe(function () {
  document.querySelectorAll(".bb-filters").forEach(function (box) {
    var table = box.parentElement.querySelector("table");
    var headers = Array.from(table.tHead.rows[0].cells).map(function (c) { return c.textContent.trim(); });
    var col = headers.indexOf(box.dataset.column);
    if (col < 0) { throw new Error("bb-filters: no column " + box.dataset.column); }
    var rows = Array.from(table.tBodies[0].rows);
    var values = [];
    rows.forEach(function (r) {
      var v = r.cells[col].textContent.trim();
      if (values.indexOf(v) < 0) { values.push(v); }
    });
    var shown = new Set(values);
    var showFixed = true;
    function fixed(r) { return r.querySelector(".bb-fixed") !== null; }
    function visible(r) { return showFixed || !fixed(r); }
    var nFixed = rows.filter(fixed).length;
    function apply() {
      rows.forEach(function (r) {
        r.hidden = !shown.has(r.cells[col].textContent.trim()) || !visible(r);
      });
      box.querySelectorAll("button").forEach(function (b) {
        var on = b.dataset.value ? shown.has(b.dataset.value) : shown.size === values.length;
        b.setAttribute("aria-pressed", on ? "true" : "false");
        if (b.dataset.value) {
          var n = rows.filter(function (r) {
            return r.cells[col].textContent.trim() === b.dataset.value && visible(r);
          }).length;
          b.textContent = b.dataset.value + " (" + n + ")";
        }
      });
    }
    function button(value) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = value || "All";
      if (value) { b.dataset.value = value; }
      b.addEventListener("click", function () {
        if (!value) {
          shown = shown.size === values.length ? new Set() : new Set(values);
        } else if (shown.has(value)) {
          shown.delete(value);
        } else {
          shown.add(value);
        }
        apply();
      });
      box.appendChild(b);
    }
    box.replaceChildren();
    button();
    values.forEach(button);
    if (nFixed) {
      var label = document.createElement("label");
      label.className = "bb-switch";
      var input = document.createElement("input");
      input.type = "checkbox";
      input.setAttribute("role", "switch");
      input.checked = showFixed;
      input.addEventListener("change", function () { showFixed = input.checked; apply(); });
      label.append(input, " Show fixed (" + nFixed + ")");
      box.appendChild(label);
    }
    apply();
  });
});
