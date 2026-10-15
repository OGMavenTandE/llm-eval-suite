export function mount(root, ctx) {
  const title = document.createElement("h2");
  title.textContent = ctx.title || "Sample notes";
  const note = document.createElement("p");
  note.className = "lede";
  note.textContent = "A local sample panel. It runs on this computer only.";
  const line = document.createElement("p");
  line.id = "sample-ping";
  line.textContent = "Checking the local route.";
  root.appendChild(title);
  root.appendChild(note);
  root.appendChild(line);
  const prefix = ctx.apiPrefix || "";
  fetch(prefix + "/ping")
    .then((response) => response.json())
    .then((body) => {
      line.textContent = body && body.ok ? "Sample panel route answered." : "Sample panel route failed.";
    })
    .catch(() => {
      line.textContent = "Sample panel route did not answer.";
    });
}
