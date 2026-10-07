// RiceLeaf front-end: drag-and-drop upload with preview, and a busy state while the model runs.
(function () {
  "use strict";

  document.querySelectorAll("[data-dropzone]").forEach(function (zone) {
    var input = zone.querySelector("input[type=file]");
    var preview = zone.querySelector("[data-preview]");
    var prompt = zone.querySelector("[data-prompt]");
    var form = zone.closest("form");
    var nameLabel = form ? form.querySelector("[data-filename]") : null;
    if (!input) return;

    function show(file) {
      if (!file) return;
      if (!file.type || file.type.indexOf("image/") !== 0) {
        if (nameLabel) nameLabel.textContent = file.name + " is not an image. Choose a JPG, PNG, WebP or BMP.";
        return;
      }
      if (preview) {
        if (preview.src) URL.revokeObjectURL(preview.src);
        preview.src = URL.createObjectURL(file);
        preview.hidden = false;
      }
      if (prompt) prompt.hidden = true;
      zone.classList.add("has-preview");
      zone.classList.remove("has-error");
      if (nameLabel) nameLabel.textContent = file.name + " (" + (file.size / 1048576).toFixed(1) + " MB)";
    }

    input.addEventListener("change", function () { show(input.files[0]); });

    ["dragenter", "dragover"].forEach(function (type) {
      zone.addEventListener(type, function (e) { e.preventDefault(); zone.classList.add("is-dragover"); });
    });
    ["dragleave", "drop"].forEach(function (type) {
      zone.addEventListener(type, function (e) { e.preventDefault(); zone.classList.remove("is-dragover"); });
    });
    zone.addEventListener("drop", function (e) {
      var files = e.dataTransfer && e.dataTransfer.files;
      if (!files || !files.length) return;
      var dt = new DataTransfer();
      dt.items.add(files[0]);
      input.files = dt.files;
      show(files[0]);
    });
  });

  // Busy state: the model takes a moment, so say what is happening and stop double submits.
  document.querySelectorAll("[data-upload-form]").forEach(function (form) {
    form.addEventListener("submit", function (e) {
      var fileInput = form.querySelector("input[type=file]");
      if (fileInput && !fileInput.files.length) {
        e.preventDefault();
        var label = form.querySelector("[data-filename]");
        if (label) label.textContent = "Choose a photo first.";
        fileInput.click();
        return;
      }
      var button = form.querySelector("[data-submit]");
      if (!button) return;
      button.disabled = true;
      var spinner = button.querySelector(".spinner-border");
      if (spinner) spinner.hidden = false;
      var text = button.querySelector("[data-label]");
      if (text) text.textContent = "Diagnosing leaf…";
    });
  });
})();
