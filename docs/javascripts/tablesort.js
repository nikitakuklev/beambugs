document$.subscribe(function () {
  document.querySelectorAll("article table:not(:has(th:empty))").forEach(function (table) {
    table.dataset.defects = "";
    new Tablesort(table);
  });
});
