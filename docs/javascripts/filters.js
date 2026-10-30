// Toggle buttons that show or hide index rows by the value of one column.
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
    function apply() {
      rows.forEach(function (r) { r.hidden = !shown.has(r.cells[col].textContent.trim()); });
      box.querySelectorAll("button").forEach(function (b) {
        var on = b.dataset.value ? shown.has(b.dataset.value) : shown.size === values.length;
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
    }
    function button(label, value) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
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
    button("All");
    values.forEach(function (v) {
      var n = rows.filter(function (r) { return r.cells[col].textContent.trim() === v; }).length;
      button(v + " (" + n + ")", v);
    });
    apply();
  });
});
