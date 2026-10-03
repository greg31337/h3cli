# CivetWeb 1.16

Source subset from the official `v1.16` release:
https://github.com/civetweb/civetweb/releases/tag/v1.16

`manifest.json` pins the release archive SHA-256 and each vendored source file.
`LICENSE.md` preserves the upstream MIT license. `http2.inl` is retained as an
upstream conditional include; HTTP/2 is not enabled.

Local naming changes use `MG_OPTIONAL_INTERFACES` for the disabled optional
interfaces and `optional_websocket_*` for their client wrappers. The interface
comment uses the same terminology. `upstream_files` in the manifest retains
the original hashes for changed files; `files` pins the vendored bytes.

The h3cli build enables IPv6 and disables TLS, generic file serving, CGI and
caching (`USE_IPV6`, `NO_SSL`, `NO_FILES`, `NO_CGI`, `NO_CACHING`). Lua,
JavaScript and WebSocket support are not enabled. TLS belongs at a reverse
proxy. Only explicit h3cli route handlers expose files. The system SQLite and
libcurl libraries supply durable metadata and optional bounded URL imports.

Validate the pinned files with `python3 tests/server_dependency.py`.
