import socket

HOST = "127.0.0.1"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #make port reusable in case python breaks prematurely
server.bind((HOST, PORT))
server.listen(1)

print("Waiting for connection...")
conn, addr = server.accept()
print(f"Connected by {addr}")

buffer = ""
while True:
    data = conn.recv(1024)
    if not data:
        break

    buffer += data.decode()

    while "\n" in buffer: #split messages on new linez
        line, buffer = buffer.split("\n", 1)
        print("Received:", line)

#close server
conn.close()
server.close()