"""TLS and URL rules for configured model endpoints.

OpenAI-compatible calls verify certificates. A CA bundle path is optional
for a closed network that uses a private certificate authority. ``verify``
is never turned off.

Configured base URLs must be http or https. Link-local and metadata hosts
are rejected, including 169.254.0.0/16. localhost stays allowed so Ollama
on this machine still connects. The check uses the host string only. It
does not open a socket.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

_DISABLED = {"0", "false", "no", "off"}
_ENABLED = {"1", "true", "yes", "on"}
_METADATA_HOSTS = {
    "metadata.google.internal",
    "metadata.goog",
}
_LINK_LOCAL_V4 = ipaddress.ip_network("169.254.0.0/16")
_LINK_LOCAL_V6 = ipaddress.ip_network("fe80::/10")

_BLOCKED_HOST = (
    "Base URL host is a link-local or metadata address. "
    "169.254.0.0/16 and IPv6 link-local addresses are rejected. "
    "localhost is still allowed for Ollama."
)


def reject_disabled_verify(value: object) -> None:
    """Raise when a caller tries to turn certificate checks off."""
    if value is False or _is_disabled_text(value):
        raise ValueError("SSL certificate verification cannot be disabled.")


def tls_verify(ca_bundle: str | None = None, *, verify: object = None) -> bool | str:
    """Value for ``requests`` ``verify``. Always ``True`` or a CA bundle path.

    ``ca_bundle`` wins, then a path passed as ``verify``, then
    ``LLM_EVAL_CA_BUNDLE``. An empty value keeps the default trust store,
    which also honors ``REQUESTS_CA_BUNDLE`` inside requests itself.
    """
    reject_disabled_verify(verify)
    chosen = _bundle_path(ca_bundle)
    if chosen is None:
        chosen = _bundle_path(verify)
    if chosen is None:
        chosen = _bundle_path(os.environ.get("LLM_EVAL_CA_BUNDLE"))
    if chosen is None:
        return True
    reject_disabled_verify(chosen)
    return chosen


def check_base_url(url: str) -> str:
    """Return the URL without a trailing slash, or raise if it is not allowed."""
    if not isinstance(url, str):
        raise ValueError("Base URL must be a string.")
    text = url.strip()
    if not text:
        raise ValueError("Base URL must include a host.")
    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise ValueError("Base URL must use http or https.")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("Base URL must include a host.")
    if _blocked_host(host):
        raise ValueError(_BLOCKED_HOST)
    return text.rstrip("/")


def _bundle_path(value: object) -> str | None:
    """A CA path. Affirmative words mean the default trust store."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() in _ENABLED:
        return None
    return text


def _is_disabled_text(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in _DISABLED


def _blocked_host(host: str) -> bool:
    if host in _METADATA_HOSTS or host.endswith(".metadata.google.internal"):
        return True
    ip = _ip_from_host(host)
    if ip is not None and _is_blocked_ip(ip):
        return True
    labels = host.split(".")
    if len(labels) > 4:
        embedded = _dotted_ipv4(".".join(labels[:4]))
        if embedded is not None and _is_blocked_ip(embedded):
            return True
    return False


def _ip_from_host(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    text = host.split("%", 1)[0]
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        ip = None
    if ip is None and (text.isdigit() or text.startswith("0x")):
        try:
            ip = ipaddress.ip_address(int(text, 0))
        except (ValueError, OverflowError):
            ip = None
    if ip is None:
        ip = _dotted_ipv4(text)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _dotted_ipv4(host: str) -> ipaddress.IPv4Address | None:
    parts = host.split(".")
    if len(parts) != 4:
        return None
    numbers: list[int] = []
    for part in parts:
        try:
            if part.lower().startswith("0x"):
                number = int(part, 16)
            elif len(part) > 1 and part.startswith("0") and part.isdigit():
                number = int(part, 8)
            elif part.isdigit():
                number = int(part, 10)
            else:
                return None
        except ValueError:
            return None
        if number > 255:
            return None
        numbers.append(number)
    return ipaddress.IPv4Address(".".join(str(number) for number in numbers))


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.version == 4:
        return ip in _LINK_LOCAL_V4
    return ip in _LINK_LOCAL_V6
