// Sortable columns for the code tables (the table under each filter bar).
document$.subscribe(function () {
  document.querySelectorAll(".bb-filters").forEach(function (box) {
    new Tablesort(box.parentElement.querySelector("table"));
  });
});
