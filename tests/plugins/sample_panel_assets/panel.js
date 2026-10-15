export function mount(root, ctx) {
  const line = document.createElement("p");
  line.id = "sample-ping";
  line.textContent = "Checking the local route.";
  root.appendChild(line);
  const prefix = (ctx && ctx.apiPrefix) || "";
  fetch(prefix + "/ping")
    .then((response) => response.json())
    .then((body) => {
      line.textContent = body && body.ok ? "Sample panel route answered." : "Sample panel route failed.";
    })
    .catch(() => {
      line.textContent = "Sample panel route did not answer.";
    });
}
