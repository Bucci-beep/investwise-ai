"""Local OAuth credentials for the isolated D5 ChatGPT-plan experiment.

No Codex credentials, API keys, inference, application state or tool execution.
Official contract: developers.openai.com/siwc/token-sharing-open-source/sign-in
and /profiles-and-sessions. All HTTP access is injectable for offline tests.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as HTTPServer
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser

DEFAULT_PROFILE_DIR = Path.home() / ".config" / "investwise" / "d5-chatgpt-plan"
MODEL = "gpt-6.1-sol"
ISSUER = "https://auth.openai.com"
DISCOVERY_URL = ISSUER + "/.well-known/openid-configuration"
AUTHORIZE_URL = ISSUER + "/api/accounts/authorize"
TOKEN_URL = ISSUER + "/api/accounts/oauth/token"
JWKS_URL = ISSUER + "/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
MODELS_URL = RESOURCE + "/models"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + DIRECT_SCOPE
CALLBACK_PATH = "/auth/callback"
MAX_RESPONSE_BYTES = 1_048_576
HTTP_TIMEOUT = 20
_TOKEN_FIELDS = ("access_token", "refresh_token", "id_token")
_TERMINAL_REFRESH = frozenset({"invalid_grant", "invalid_refresh_token", "token_expired",
    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_invalid", "refresh_token_reused"})
_SAFE_PROVIDER_CODES = _TERMINAL_REFRESH | {"invalid_client", "access_denied",
    "subscription_sharing_user_not_eligible", "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_usage_unavailable", "subscription_sharing_unsupported_capability",
    "subscription_sharing_route_not_supported", "subscription_sharing_invalid_user",
    "chatpass_v2_scope_not_authorized", "chatpass_v2_invalid_authorization_context"}


class AuthBlocked(RuntimeError):
    """Only a bounded code and HTTP status, never a provider body or credential."""

    def __init__(self, code: str, *, status: int | None = None):
        self.code = code
        self.status = status
        super().__init__(code if status is None else f"{code} (HTTP {status})")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_json(method, url, data, headers):
    """One request, no redirects/retries; errors never include raw response text."""
    if url not in {DISCOVERY_URL, JWKS_URL, TOKEN_URL, MODELS_URL}:
        raise AuthBlocked("endpoint_not_allowed")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=HTTP_TIMEOUT) as reply:
            body = reply.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        code = "oauth_http_error"
        try:
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
            parsed = json.loads(raw) if len(raw) <= MAX_RESPONSE_BYTES else {}
            error = parsed.get("error") if isinstance(parsed, dict) else None
            candidate = error.get("code") if isinstance(error, dict) else error
            if isinstance(candidate, str) and candidate in _SAFE_PROVIDER_CODES:
                code = candidate
        except Exception:
            pass
        raise AuthBlocked(code, status=exc.code) from None
    except Exception:
        raise AuthBlocked("oauth_network_error") from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise AuthBlocked("oauth_response_too_large")
    try:
        result = json.loads(body)
    except (ValueError, UnicodeError):
        raise AuthBlocked("oauth_response_invalid") from None
    if not isinstance(result, dict):
        raise AuthBlocked("oauth_response_invalid")
    return result


class ChatGPTCredentials:
    """Explicit authorisation only; construction has no I/O or browser effects.

    transport(method, url, data_bytes_or_None, headers) must return a JSON object.
    No method retries inference. Token renewal happens before a caller obtains
    a token; a failed model request must not trigger renewal/replay here.
    """

    def __init__(self, profile_dir=DEFAULT_PROFILE_DIR, *, transport=None, browser_open=None):
        self.profile_dir = Path(profile_dir).absolute()
        self._transport = transport or _http_json
        self._browser_open = browser_open or webbrowser.open
        self._sensitive: set[str] = set()

    def sensitive_values(self) -> tuple[str, ...]:
        """Includes replaced tokens seen by this process, for receipt redaction."""
        return tuple(sorted(self._sensitive))

    def _remember(self, record):
        for name in _TOKEN_FIELDS:
            value = record.get(name)
            if isinstance(value, str) and value:
                self._sensitive.add(value)

    def _prepare(self):
        repo = Path(__file__).resolve().parents[4]
        if self.profile_dir.is_symlink() or self.profile_dir.resolve().is_relative_to(repo):
            raise AuthBlocked("credential_storage_must_be_outside_repository")
        self.profile_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        info = self.profile_dir.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise AuthBlocked("credential_directory_not_owned")
        self.profile_dir.chmod(0o700)

    @contextmanager
    def _locked(self):
        self._prepare()
        descriptor = os.open(self.profile_dir / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
                raise AuthBlocked("credential_lock_invalid")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            os.close(descriptor)

    def _read(self, name):
        path = self.profile_dir / name
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError:
            raise AuthBlocked("credential_file_invalid") from None
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                raise AuthBlocked("credential_file_permissions_invalid")
            try:
                payload = handle.read(MAX_RESPONSE_BYTES + 1)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise ValueError
                result = json.loads(payload)
            except (ValueError, UnicodeError):
                raise AuthBlocked("credential_file_invalid") from None
        if not isinstance(result, dict):
            raise AuthBlocked("credential_file_invalid")
        self._remember(result)
        return result

    def _write(self, name, value):
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.profile_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(value, handle, ensure_ascii=True, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.profile_dir / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _record_name(client_id):
        if not isinstance(client_id, str) or not client_id or client_id == "dynamic_agent_client":
            raise AuthBlocked("issued_client_id_missing")
        return "account-" + hashlib.sha256(client_id.encode()).hexdigest() + ".json"

    def _active(self):
        active = self._read("active.json")
        if active is None:
            return None
        record = self._read(self._record_name(active.get("client_id")))
        if record is None or record.get("client_id") != active.get("client_id"):
            raise AuthBlocked("account_registration_incomplete")
        return record

    def _host_id(self):
        record = self._read("host.json")
        if record is None:
            record = {"ext_agent_host_id": "urn:uuid:" + str(uuid.uuid4())}
            self._write("host.json", record)
        value = record.get("ext_agent_host_id")
        try:
            parsed = uuid.UUID(value.removeprefix("urn:uuid:"))
            if value != "urn:uuid:" + str(parsed) or parsed.version != 4:
                raise ValueError
        except (AttributeError, ValueError, TypeError):
            raise AuthBlocked("host_id_invalid") from None
        return value

    def _request(self, method, url, fields=None, token=None):
        headers = {"Accept": "application/json"}
        data = None
        if fields is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            data = urllib.parse.urlencode(fields).encode("ascii")
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        try:
            result = self._transport(method, url, data, headers)
        except AuthBlocked:
            raise
        except Exception:
            raise AuthBlocked("oauth_transport_failed") from None
        if not isinstance(result, dict):
            raise AuthBlocked("oauth_response_invalid")
        self._remember(result)
        return result

    def _discovery(self):
        data = self._request("GET", DISCOVERY_URL)
        expected = {"issuer": ISSUER, "authorization_endpoint": AUTHORIZE_URL,
                    "token_endpoint": TOKEN_URL, "jwks_uri": JWKS_URL}
        if any(data.get(key) != value for key, value in expected.items()):
            raise AuthBlocked("unexpected_openid_configuration")
        return data

    def _verify_identity(self, token, client_id, *, nonce=None, subject=None):
        try:
            import jwt
        except ImportError:
            raise AuthBlocked("install_isolated_oauth_dependencies") from None
        try:
            header = jwt.get_unverified_header(token)
            # Never let a JWT choose an HMAC key or bypass its signature.
            if header.get("alg") not in {"RS256", "ES256", "EdDSA"} or not header.get("kid"):
                raise ValueError
            keys = self._request("GET", JWKS_URL).get("keys", [])
            matches = [key for key in keys if isinstance(key, dict) and key.get("kid") == header["kid"]
                       and key.get("use", "sig") == "sig" and key.get("alg", header["alg"]) == header["alg"]]
            if len(matches) != 1:
                raise ValueError
            key = jwt.PyJWK.from_dict(matches[0], algorithm=header["alg"]).key
            claims = jwt.decode(token, key, algorithms=[header["alg"]], audience=client_id,
                issuer=ISSUER, leeway=5, options={"require": ["sub", "exp", "iat", "iss", "aud"]})
            if not isinstance(claims["sub"], str) or not claims["sub"]:
                raise ValueError
            if nonce is not None and (not isinstance(claims.get("nonce"), str)
                    or not secrets.compare_digest(claims["nonce"], nonce)):
                raise ValueError
            if subject is not None and claims["sub"] != subject:
                raise ValueError
            if isinstance(claims["aud"], list) and len(claims["aud"]) > 1 and claims.get("azp") != client_id:
                raise ValueError
            return claims
        except AuthBlocked:
            raise
        except Exception:
            raise AuthBlocked("id_token_validation_failed") from None

    def _token_record(self, response, *, client_id, host_id, identity, previous=None):
        # A refresh may omit its unchanged granted scope and ID token.
        scope = response.get("scope")
        scopes = scope.split() if isinstance(scope, str) else (previous or {}).get("scopes", [])
        expiry = response.get("expires_in")
        if (response.get("token_type", "").lower() != "bearer"
                or not isinstance(expiry, (int, float)) or isinstance(expiry, bool) or not 0 < expiry <= 86_400
                or any(not isinstance(response.get(key), str) or not response[key]
                       for key in ("access_token", "refresh_token"))):
            raise AuthBlocked("token_response_invalid")
        id_token = response.get("id_token") or (previous or {}).get("id_token")
        if not isinstance(id_token, str) or not id_token:
            raise AuthBlocked("id_token_missing")
        return {"issuer": ISSUER, "subject": identity["sub"], "client_id": client_id,
            "ext_agent_host_id": host_id, "access_token": response["access_token"],
            "refresh_token": response["refresh_token"], "id_token": id_token,
            "token_type": "Bearer", "scopes": scopes, "expires_at": time.time() + expiry,
            "email": identity.get("email"), "saved_at": time.time()}

    def authorize(self, *, timeout_seconds=300):
        """Open the system browser; never print the URL, callback or credentials."""
        if not 1 <= timeout_seconds <= 300:
            raise ValueError("OAuth wait must be between 1 and 300 seconds")
        with self._locked():
            self._discovery()
            previous = self._active()
            pending = previous or self._read("pending-registration.json") or {}
            client_id = pending.get("client_id")
            host_id = self._host_id()
            state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
            self._sensitive.update({state, nonce, verifier})
            result = {}
            callback_lock = threading.Lock()

            class Callback(BaseHTTPRequestHandler):
                def setup(handler):
                    # A partial local request must not hold up the sign-in deadline.
                    handler.request.settimeout(1.0)
                    super().setup()

                def log_message(self, format, *args):
                    pass

                def do_GET(handler):
                    parsed = urllib.parse.urlsplit(handler.path)
                    values = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                    valid = (parsed.path == CALLBACK_PATH and not parsed.fragment
                        and all(len(value) == 1 for value in values.values())
                        and secrets.compare_digest(values.get("state", [""])[0], state)
                        and handler.headers.get("Host") == f"127.0.0.1:{handler.server.server_port}")
                    with callback_lock:
                        if valid and not result:
                            result.update({name: value[0] for name, value in values.items()})
                    handler.send_response(200 if valid else 400)
                    handler.send_header("Content-Type", "text/plain; charset=utf-8")
                    handler.send_header("Cache-Control", "no-store")
                    handler.send_header("Referrer-Policy", "no-referrer")
                    handler.end_headers()
                    handler.wfile.write(b"Sign-in response received. Return to InvestWise."
                                        if valid else b"Invalid sign-in response.")

            with HTTPServer(("127.0.0.1", 0), Callback) as server:
                server.timeout = 0.5
                redirect_uri = f"http://127.0.0.1:{server.server_port}{CALLBACK_PATH}"
                parameters = {"client_id": client_id or "dynamic_agent_client", "ext_agent_host_id": host_id,
                    "response_type": "code", "redirect_uri": redirect_uri, "scope": SCOPES,
                    "resource": RESOURCE, "state": state, "nonce": nonce,
                    "code_challenge_method": "S256",
                    "code_challenge": base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()}
                if client_id:
                    if pending.get("id_token"):
                        parameters["id_token_hint"] = pending["id_token"]
                else:
                    parameters["agent_name_hint"] = "InvestWise D5 prototype"
                deadline = time.monotonic() + timeout_seconds
                try:
                    opened = self._browser_open(AUTHORIZE_URL + "?" + urllib.parse.urlencode(parameters))
                except Exception:
                    raise AuthBlocked("system_browser_unavailable") from None
                if not opened:
                    raise AuthBlocked("system_browser_unavailable")
                print("Continue with ChatGPT in the opened browser. Waiting for sign-in…", flush=True)
                while not result and time.monotonic() < deadline:
                    server.handle_request()
            if not result:
                raise AuthBlocked("oauth_callback_timeout")
            if result.get("error"):
                raise AuthBlocked("oauth_consent_denied")
            issued = result.get("client_id") or client_id
            self._record_name(issued)
            if client_id is not None and issued != client_id:
                raise AuthBlocked("oauth_client_mismatch")
            code = result.get("code")
            if not isinstance(code, str) or not code:
                raise AuthBlocked("authorization_code_missing")
            self._sensitive.add(code)
            # Retain an issued registration if a one-use code fails to exchange.
            self._write("pending-registration.json", {"client_id": issued})
            tokens = self._request("POST", TOKEN_URL, {"grant_type": "authorization_code",
                "client_id": issued, "code": code, "code_verifier": verifier,
                "redirect_uri": redirect_uri, "resource": RESOURCE})
            identity = self._verify_identity(tokens.get("id_token"), issued, nonce=nonce,
                                             subject=(previous or {}).get("subject"))
            record = self._token_record(tokens, client_id=issued, host_id=host_id, identity=identity)
            self._write(self._record_name(issued), record)
            self._write("active.json", {"client_id": issued})
            (self.profile_dir / "pending-registration.json").unlink(missing_ok=True)
            if DIRECT_SCOPE not in record["scopes"] or "resource.invoke" not in record["scopes"]:
                raise AuthBlocked("chatgpt_plan_permission_not_granted")
            return {"authorised": True, "direct_usage_enabled": True, "credential_storage": "protected_local"}

    def get_access_token(self):
        """Read only this app's account; refresh once under the same process lock."""
        with self._locked():
            record = self._active()
            if record is None:
                raise AuthBlocked("chatgpt_sign_in_required")
            if DIRECT_SCOPE not in record.get("scopes", []) or "resource.invoke" not in record.get("scopes", []):
                raise AuthBlocked("chatgpt_plan_permission_not_granted")
            if (record.get("issuer") != ISSUER or not record.get("subject")
                    or record.get("ext_agent_host_id") != self._host_id()):
                raise AuthBlocked("account_registration_invalid")
            expiry = record.get("expires_at")
            if not isinstance(expiry, (int, float)) or isinstance(expiry, bool):
                raise AuthBlocked("credential_expiry_invalid")
            if expiry <= time.time() + 60:
                refresh = record.get("refresh_token")
                if not isinstance(refresh, str) or not refresh:
                    raise AuthBlocked("chatgpt_sign_in_required")
                try:
                    response = self._request("POST", TOKEN_URL, {"grant_type": "refresh_token",
                        "client_id": record["client_id"], "refresh_token": refresh, "resource": RESOURCE})
                except AuthBlocked as exc:
                    if exc.code in _TERMINAL_REFRESH:
                        for field in _TOKEN_FIELDS:
                            record.pop(field, None)
                        record["expires_at"] = 0
                        self._write(self._record_name(record["client_id"]), record)
                    raise
                identity = {"sub": record["subject"], "email": record.get("email")}
                if response.get("id_token"):
                    identity = self._verify_identity(response["id_token"], record["client_id"], subject=record["subject"])
                record = self._token_record(response, client_id=record["client_id"],
                    host_id=record["ext_agent_host_id"], identity=identity, previous=record)
                self._write(self._record_name(record["client_id"]), record)
                if DIRECT_SCOPE not in record["scopes"] or "resource.invoke" not in record["scopes"]:
                    raise AuthBlocked("chatgpt_plan_permission_not_granted")
            token = record.get("access_token")
            if not isinstance(token, str) or not token:
                raise AuthBlocked("chatgpt_sign_in_required")
            self._sensitive.add(token)
            return token

    def list_models(self):
        token = self.get_access_token()
        response = self._request("GET", MODELS_URL, token=token)
        rows = response.get("models")
        if not isinstance(rows, list):
            raise AuthBlocked("model_catalogue_invalid")
        catalogue = []
        for row in rows:
            if not isinstance(row, dict) or row.get("visibility") != "list":
                continue
            if not all(isinstance(row.get(key), str) and 0 < len(row[key]) <= 200 for key in ("slug", "display_name")):
                raise AuthBlocked("model_catalogue_invalid")
            safe = {key: row[key] for key in ("slug", "display_name", "visibility")}
            if any(secret in json.dumps(safe) for secret in self._sensitive):
                raise AuthBlocked("credential_in_provider_response")
            catalogue.append(safe)
        if MODEL not in {row["slug"] for row in catalogue}:
            raise AuthBlocked("exact_model_unavailable")
        return catalogue
