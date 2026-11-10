"""TLS and URL rules for configured model endpoints.

OpenAI-compatible calls verify certificates. A CA bundle path is optional
for a closed network that uses a private certificate authority. ``verify``
is never turned off.

Configured base URLs must be http or https. Link-local and metadata hosts
are rejected, including 169.254.0.0/16. localhost stays allowed so Ollama
on this machine still connects. The check uses the host string only. It
does not open a socket. Shortened, octal, and hexadecimal IPv4 spellings
are read the same way the system resolver reads them.
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
    "metadata.tencentyun.com",
    "instance-data.ec2.internal",
}
_LINK_LOCAL_V4 = ipaddress.ip_network("169.254.0.0/16")
_LINK_LOCAL_V6 = ipaddress.ip_network("fe80::/10")
# Cloud metadata addresses that are not in the link-local ranges above.
_EXTRA_BLOCKED_IPS = {
    ipaddress.ip_address("100.100.100.200"),
    ipaddress.ip_address("fd00:ec2::254"),
}
_REDIRECT_STATUS = {301, 302, 303, 307, 308}

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


def refuse_redirect(response) -> None:
    """Raise when ``response`` is a redirect.

    Callers pass ``allow_redirects=False``. A followed redirect would skip
    :func:`check_base_url` and can land on a metadata address.
    """
    status = getattr(response, "status_code", None)
    if getattr(response, "is_redirect", False) or status in _REDIRECT_STATUS:
        close = getattr(response, "close", None)
        if callable(close):
            close()
        raise RuntimeError("Refusing to follow a redirect from this endpoint.")


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
    if host.endswith(".metadata.tencentyun.com"):
        return True
    ip = _ip_from_host(host)
    if ip is not None:
        return _is_blocked_ip(ip)
    # 169.254.43518.nip.io and customer1.169.254.169.254.nip.io embed the
    # metadata address in a name the resolver would turn into that address.
    labels = host.split(".")
    if len(labels) < 2:
        return False
    for start in range(len(labels)):
        room = len(labels) - start
        for count in range(min(4, room), 0, -1):
            embedded = _ipv4_literal(".".join(labels[start : start + count]))
            if embedded is not None and _is_blocked_ip(embedded):
                return True
    return False


def _ip_from_host(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    text = host.split("%", 1)[0]
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        ip = _ipv4_literal(text)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _ipv4_component(part: str) -> int | None:
    text = part.strip()
    if not text:
        return None
    try:
        lowered = text.lower()
        if lowered.startswith("0x"):
            if len(lowered) == 2:
                return None
            return int(lowered, 16)
        if len(text) > 1 and text.startswith("0") and all(ch in "01234567" for ch in text):
            return int(text, 8)
        if text.isdigit():
            return int(text, 10)
    except ValueError:
        return None
    return None


def _ipv4_literal(host: str) -> ipaddress.IPv4Address | None:
    """One to four IPv4 parts, including the short forms ``inet_aton`` accepts."""
    parts = host.split(".")
    if not 1 <= len(parts) <= 4:
        return None
    numbers: list[int] = []
    for part in parts:
        number = _ipv4_component(part)
        if number is None or number < 0:
            return None
        numbers.append(number)
    if len(numbers) == 1:
        value = numbers[0]
        if value > 0xFFFFFFFF:
            return None
    elif len(numbers) == 2:
        if numbers[0] > 0xFF or numbers[1] > 0xFFFFFF:
            return None
        value = (numbers[0] << 24) | numbers[1]
    elif len(numbers) == 3:
        if numbers[0] > 0xFF or numbers[1] > 0xFF or numbers[2] > 0xFFFF:
            return None
        value = (numbers[0] << 24) | (numbers[1] << 16) | numbers[2]
    else:
        if any(number > 0xFF for number in numbers):
            return None
        value = (numbers[0] << 24) | (numbers[1] << 16) | (numbers[2] << 8) | numbers[3]
    return ipaddress.IPv4Address(value)


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip in _EXTRA_BLOCKED_IPS:
        return True
    if ip.version == 4:
        return ip in _LINK_LOCAL_V4
    return ip in _LINK_LOCAL_V6
