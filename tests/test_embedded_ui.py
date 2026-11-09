"""The ready iframe must not navigate to the service-token tunnel.

Upstream serves ComfyUI by pointing the iframe at backend_url. That hostname is
Access service-auth only in this deployment, so the browser gets Access 403.
The gate already proxies HTML, assets, API and /ws once the session is ready.
The shell keeps `/` and loads the proxied document at `/index.html`.
"""
from pathlib import Path

import aiohttp
import pytest
import yarl
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gate.app import PROXY, SESSION, build_app
from gate.config import Settings
from gate.profiles import load_profile
from gate.proxy import OVERLAY_TAG
from gate.state import State
from tests.fakes import FakeColab

ROOT = Path(__file__).resolve().parent.parent
TUNNEL = "colab-tunnel.test.invalid"
SHELL = "Comfy Colab Kit"


async def _backend(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    app.router.add_route("*", "/", handler)
    server = TestServer(app)
    await server.start_server()
    return server


async def _gate(tmp_path, backend_port, *, auth=False, agent_token=""):
    agent = tmp_path / "agent_token"
    if agent_token:
        agent.write_text(agent_token)
    env = {
        "AUTH_DISABLED": "false" if auth else "true",
        "STATE_DIR": str(tmp_path / "state"),
        "DEFAULT_PROFILE": "tiny",
        "COLAB_TUNNEL_HOST": TUNNEL,
        "AGENT_TOKEN_FILE": str(agent) if agent_token else "",
        "PUBLIC_HOST": "colab.example.com",
    }
    if auth:
        env["CF_ACCESS_TEAM_DOMAIN"] = "team.cloudflareaccess.com"
        env["CF_ACCESS_AUD"] = "aud-test"
    settings = Settings.from_env(env)
    app = build_app(
        settings, FakeColab(),
        {"tiny": load_profile(ROOT / "tests/fixtures/profiles/tiny")},
        run_background=False,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    # Production stores a hostname and the proxy uses https://hostname.
    # Point that proxy at the local fake without changing Settings.
    app[PROXY].backend = f"http://127.0.0.1:{backend_port}"
    return client, app


def _ready(app):
    app[SESSION].machine.state = State.READY


async def test_shell_iframe_targets_same_origin_document(tmp_path):
    hits = []

    async def handler(req):
        hits.append(req.path)
        return web.Response(text="backend")

    server = await _backend(handler)
    try:
        client, app = await _gate(tmp_path, server.port)
        try:
            page = await client.get("/")
            body = await page.text()
            assert page.status == 200 and SHELL in body
            assert 'frame.src="/index.html"' in body
            assert "S.backend_url" not in body
            assert hits == []

            status = await (await client.get("/_gate/status")).json()
            # The status field is unchanged. The iframe no longer reads it.
            assert status["backend_url"] == f"https://{TUNNEL}"

            _ready(app)
            shell = await client.get("/")
            shell_body = await shell.text()
            assert SHELL in shell_body and 'id="profile"' in shell_body
            assert 'id="comfy-frame"' in shell_body
            assert hits == []
        finally:
            await client.close()
    finally:
        await server.close()


async def test_ready_document_is_proxied_and_shell_is_not_nested(tmp_path):
    hits = []

    async def handler(req):
        hits.append({"path": req.path, "raw": req.raw_path, "upgrade": req.headers.get("Upgrade", "")})
        if req.headers.get("Upgrade", "").lower() == "websocket":
            ws = web.WebSocketResponse()
            await ws.prepare(req)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await ws.send_str("echo:" + msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await ws.send_bytes(msg.data)
                else:
                    break
            return ws
        if req.path == "/index.html":
            return web.Response(text="<html><body><h1>ComfyUI</h1></body></html>", content_type="text/html")
        if req.path == "/assets/app.js":
            resp = web.StreamResponse(status=200, headers={"Content-Type": "text/javascript"})
            await resp.prepare(req)
            await resp.write(b"window.COMFY=1;")
            await resp.write(b"window.DONE=1;")
            await resp.write_eof()
            return resp
        if req.path == "/queue":
            return web.json_response({"queue_running": [], "queue_pending": []})
        if req.path.startswith("/api/userdata/"):
            return web.json_response({"name": req.raw_path})
        return web.Response(status=404, text="no")

    server = await _backend(handler)
    try:
        client, app = await _gate(tmp_path, server.port)
        try:
            asleep = await client.get("/index.html")
            assert SHELL in await asleep.text()
            assert hits == []
            assert (await client.get("/queue")).status == 503

            _ready(app)
            doc = await client.get("/index.html")
            text = await doc.text()
            assert doc.status == 200
            assert "ComfyUI" in text and OVERLAY_TAG in text
            assert SHELL not in text
            assert TUNNEL not in text

            asset = await client.get("/assets/app.js")
            assert await asset.read() == b"window.COMFY=1;window.DONE=1;"
            assert (await (await client.get("/queue")).json())["queue_pending"] == []

            raw = "/api/userdata/workflows%2Fmy%20flow.json"
            wf = await client.get(yarl.URL(raw, encoded=True))
            assert wf.status == 200
            assert "%2F" in (await wf.json())["name"]

            ws = await client.ws_connect("/ws?clientId=abc")
            await ws.send_str("hi")
            assert (await ws.receive()).data == "echo:hi"
            await ws.send_bytes(b"xyz")
            assert (await ws.receive()).data == b"xyz"
            await ws.close()
            assert any(h["path"] == "/ws" and "clientId=abc" in h["raw"] for h in hits)
        finally:
            await client.close()
    finally:
        await server.close()


async def test_unauthorized_browser_does_not_reach_the_tunnel(tmp_path):
    hits = []

    async def handler(req):
        hits.append(req.path)
        return web.Response(text="<html><body>ComfyUI</body></html>", content_type="text/html")

    server = await _backend(handler)
    try:
        client, app = await _gate(tmp_path, server.port, auth=True, agent_token="agent-test-token")
        try:
            _ready(app)
            for path in ("/index.html", "/assets/app.js", "/queue", "/api/userdata/workflows%2Fx.json"):
                assert (await client.get(path)).status == 401
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await client.ws_connect("/ws")
            assert (await client.get("/index.html", headers={"Authorization": "Bearer nope"})).status == 401
            assert hits == []
            ok = await client.get("/index.html", headers={"Authorization": "Bearer agent-test-token"})
            assert ok.status == 200 and "ComfyUI" in await ok.text()
            assert hits == ["/index.html"]
        finally:
            await client.close()
    finally:
        await server.close()


async def test_document_redirect_is_not_followed_by_the_proxy(tmp_path):
    hits = []

    async def handler(req):
        hits.append(req.path)
        raise web.HTTPFound("/")

    server = await _backend(handler)
    try:
        client, app = await _gate(tmp_path, server.port)
        try:
            _ready(app)
            res = await client.get("/index.html", allow_redirects=False)
            loc = res.headers.get("Location", "")
            assert res.status == 302
            assert hits == ["/index.html"]
            assert TUNNEL not in loc
        finally:
            await client.close()
    finally:
        await server.close()
