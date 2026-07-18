"""FastAPI app factory + launcher for the relabeling UI (spec §9).

Serves the JSON API and the built Vue single-page app. Predictions for the current prompt
version are fetched once at startup (best-effort) so the review pane can show gold next to
the model's answer (§9.1).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..config.loader import load_config
from .api import router
from .predictions import latest_predictions
from .session import RelabelSession

WEB_DIST = Path(__file__).parent / 'web' / 'dist'


def create_app(repo: str | Path, tool_name: str, tracking_uri: str = './mlruns') -> FastAPI:
    config = load_config(repo)
    tool = config.tool(tool_name)

    app = FastAPI(title='Opaque Relabeling', docs_url='/api/docs')
    app.state.session = RelabelSession(repo, tool)
    app.state.project = config.project
    app.state.tool = tool
    app.state.repo = str(repo)
    app.state.tracking_uri = tracking_uri
    app.state.predictions = latest_predictions(repo, config.project, tool, tracking_uri)

    app.include_router(router)
    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    if WEB_DIST.is_dir():
        # html=True serves index.html for '/' and unknown paths (SPA routing).
        app.mount('/', StaticFiles(directory=str(WEB_DIST), html=True), name='web')
    else:
        @app.get('/', response_class=HTMLResponse)
        def _unbuilt() -> str:
            return (
                '<h1>Opaque relabeling</h1>'
                '<p>The web UI is not built yet. Build it with:</p>'
                '<pre>cd src/opaque/relabel/web &amp;&amp; npm install &amp;&amp; npm run build</pre>'
                '<p>The JSON API is available under <code>/api</code>.</p>'
            )


def serve(
    repo: str | Path,
    tool_name: str,
    *,
    tracking_uri: str = './mlruns',
    host: str = '127.0.0.1',
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    import threading
    import webbrowser

    import uvicorn

    app = create_app(repo, tool_name, tracking_uri=tracking_uri)
    url = f'http://{host}:{port}'
    print(f'Opaque relabeling UI → {url}  (project {app.state.project}, tool {tool_name})')
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level='warning')
