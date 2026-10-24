"""The network guard of tests/live_agent_progress_demo.py (a dev-only demo; not part of the app). New file.

install() makes THIS Python process refuse, through the standard socket module, every outbound connection or
datagram to a non-loopback address and every name or address lookup of a non-loopback host: socket.socket.connect /
connect_ex / sendto, socket.create_connection, and the resolvers socket.getaddrinfo, gethostbyname,
gethostbyname_ex, gethostbyaddr and getnameinfo (each refused BEFORE the real resolver is called). Loopback
(127.0.0.0/8, ::1, "localhost") stays allowed, because Streamlit's own server and asyncio's local socketpair need
it; accepting connections (the browser reaching the local Streamlit server) is not affected.

Boundaries, stated plainly: it does not cover other processes, native code that bypasses Python's socket module, or
a service listening on loopback that forwards traffic onwards. In the demo no provider key is present and every paid
authorization is refused as well, so a provider request could not be authorized even if a connection were made.
"""

import ipaddress
import socket

LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", ""})
RESOLVERS = ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo")
_installed = {}


class NetworkBlocked(ConnectionRefusedError):
    """A non-loopback network operation refused by the demo's guard."""


def is_loopback(host):
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    host = host.strip("[]").split("%", 1)[0].lower()
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check(address):
    if isinstance(address, (str, bytes)):          # an AF_UNIX path: local by definition
        return
    host = address[0] if isinstance(address, tuple) and address else None
    if not is_loopback(host):
        raise NetworkBlocked(f"the demo's network guard refuses non-loopback address {host!r}")


def install():
    """Patch the socket module (idempotent). Returns uninstall()."""
    if _installed:
        return uninstall
    real = {"connect": socket.socket.connect, "connect_ex": socket.socket.connect_ex,
            "sendto": socket.socket.sendto, **{n: getattr(socket, n) for n in RESOLVERS}}
    _installed.update(real)

    def connect(self, address):
        _check(address)
        return real["connect"](self, address)

    def connect_ex(self, address):
        _check(address)
        return real["connect_ex"](self, address)

    def sendto(self, data, *args):
        _check(args[-1] if args else None)
        return real["sendto"](self, data, *args)

    def resolver(name):
        def guarded(host, *args, **kwargs):
            target = host[0] if name == "getnameinfo" and isinstance(host, tuple) and host else host
            if not (target is None and name == "getaddrinfo") and not is_loopback(target):
                raise NetworkBlocked(f"the demo's network guard refuses {name}({target!r})")
            return real[name](host, *args, **kwargs)
        guarded.__name__ = name
        return guarded
    socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto = connect, connect_ex, sendto
    for name in RESOLVERS:                          # socket.create_connection resolves through getaddrinfo first
        setattr(socket, name, resolver(name))
    return uninstall


def uninstall():
    if not _installed:
        return
    socket.socket.connect, socket.socket.connect_ex = _installed["connect"], _installed["connect_ex"]
    socket.socket.sendto = _installed["sendto"]
    for name in RESOLVERS:
        setattr(socket, name, _installed[name])
    _installed.clear()
