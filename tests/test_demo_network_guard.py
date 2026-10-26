"""The UI-3 demo's network guard (tests/demo_network_guard.py). New file. No real connection leaves loopback: every
external attempt is refused by the guard before the operating system is asked."""

import socket

import pytest

import demo_network_guard as guard


@pytest.fixture
def installed():
    guard.install()
    try:
        yield
    finally:
        guard.uninstall()


EXTERNAL = [("openrouter.ai", 443), ("93.184.216.34", 80), ("2606:4700::6810:84e5", 443), ("8.8.8.8", 53)]


@pytest.mark.parametrize("address", EXTERNAL)
def test_direct_external_connections_are_refused(installed, address):
    family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        with pytest.raises(guard.NetworkBlocked):
            s.connect(address)
        with pytest.raises(guard.NetworkBlocked):
            s.connect_ex(address)


def test_external_datagrams_name_resolution_and_create_connection_are_refused(installed):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s, pytest.raises(guard.NetworkBlocked):
        s.sendto(b"x", ("8.8.8.8", 53))
    with pytest.raises(guard.NetworkBlocked):
        socket.getaddrinfo("openrouter.ai", 443)
    with pytest.raises(guard.NetworkBlocked):
        socket.create_connection(("openrouter.ai", 443), timeout=1)
    import httpx2
    with httpx2.Client() as client, pytest.raises(Exception) as err:       # the SDK's HTTP stack, too
        client.get("https://openrouter.ai/api/v1/models")
    assert any(isinstance(e, guard.NetworkBlocked) for e in _chain(err.value))     # refused by the guard


def _chain(exc):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__


def test_loopback_stays_usable_for_streamlit_and_asyncio(installed):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2) as client:
            peer, _ = server.accept()
            with peer:
                client.sendall(b"ok")
                assert peer.recv(2) == b"ok"
    assert socket.getaddrinfo("localhost", 80) and socket.getaddrinfo("127.0.0.1", 80)
    a, b = socket.socketpair()                                             # asyncio's self-pipe
    with a, b:
        a.sendall(b"y")
        assert b.recv(1) == b"y"
    import asyncio
    assert asyncio.run(asyncio.sleep(0, result="loop ok")) == "loop ok"


def guarded_functions():
    return (socket.socket.connect, socket.socket.connect_ex, socket.socket.sendto,
            *(getattr(socket, n) for n in guard.RESOLVERS))


def test_uninstall_restores_the_socket_module():
    before = guarded_functions()
    guard.install()
    assert all(a is not b for a, b in zip(guarded_functions(), before))   # every route is replaced
    guard.install()                                                        # idempotent
    guard.uninstall()
    assert guarded_functions() == before


# ---- the resolvers, over stubs: refused BEFORE the original resolver runs (no real DNS request is made) ----

@pytest.fixture
def stub_resolvers(monkeypatch):
    """Every guarded resolver replaced by a stub that records its calls; the guard installed over the stubs."""
    calls = []

    def stub(name):
        def resolve(*args, **kwargs):
            calls.append((name, args))
            return f"stub {name}"
        return resolve
    for name in guard.RESOLVERS:
        monkeypatch.setattr(socket, name, stub(name))
    guard.install()
    try:
        yield calls
    finally:
        guard.uninstall()


@pytest.mark.parametrize("name, args", [
    ("getaddrinfo", ("openrouter.ai", 443)),
    ("gethostbyname", ("openrouter.ai",)),
    ("gethostbyname_ex", ("api.openrouter.ai",)),
    ("gethostbyaddr", ("8.8.8.8",)),
    ("gethostbyaddr", ("openrouter.ai",)),
    ("getnameinfo", (("93.184.216.34", 443), 0)),
    ("getnameinfo", (("2606:4700::6810:84e5", 443, 0, 0), 0)),
    ("gethostbyname", ("localhost.evil.example",)),
])
def test_external_lookups_are_refused_before_the_resolver_runs(stub_resolvers, name, args):
    with pytest.raises(guard.NetworkBlocked):
        getattr(socket, name)(*args)
    assert stub_resolvers == []                                            # the original was never called


@pytest.mark.parametrize("name, args", [
    ("getaddrinfo", ("127.0.0.1", 80)),
    ("getaddrinfo", (None, 80)),                                           # a local bind address
    ("gethostbyname", ("localhost",)),
    ("gethostbyname_ex", ("localhost",)),
    ("gethostbyaddr", ("127.0.0.1",)),
    ("getnameinfo", (("127.0.0.1", 8501), 0)),
    ("getnameinfo", (("::1", 8501, 0, 0), 0)),
])
def test_loopback_lookups_reach_the_resolver(stub_resolvers, name, args):
    assert getattr(socket, name)(*args) == f"stub {name}"
    assert stub_resolvers == [(name, args)]


@pytest.mark.parametrize("host, loopback", [("127.0.0.1", True), ("127.8.9.1", True), ("::1", True), ("[::1]", True),
                                            ("localhost", True), ("10.0.0.1", False), ("0.0.0.0", False),
                                            ("openrouter.ai", False), ("localhost.evil.example", False),
                                            (None, False), (b"127.0.0.1", True)])
def test_is_loopback(host, loopback):
    assert guard.is_loopback(host) is loopback
