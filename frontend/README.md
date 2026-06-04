# AI Evaluation Workbench UI

Local frontend for the Offline AI Evaluation Workbench (Milestone 3).

## Development

Start the API first:

```bash
pip install -e ".[api]"
llm-eval-api
```

Then start the UI:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Tests

```bash
npm test
npm run build
```

The Vite dev server proxies API requests to `http://127.0.0.1:8000`.
