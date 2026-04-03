import socket, sys
TELLO_IP, CMD_PORT, STATE_PORT = "192.168.10.1", 8889, 8890

cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); cmd.settimeout(2.0); cmd.bind(("", 0))
state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); state.settimeout(5.0)
try:
    state.bind(("", STATE_PORT))
except OSError as e:
    print(f"[ERR] bind(8890) 失敗: {e}"); sys.exit(1)

cmd.sendto(b"command", (TELLO_IP, CMD_PORT))
print("8890で状態受信を5秒待機…")
try:
    data, addr = state.recvfrom(2048)
    print("[OK] state受信:", addr, "len=", len(data))
    print(data.decode("ascii", "ignore"))
except socket.timeout:
    print("[NG] 5秒以内にstate未受信")
