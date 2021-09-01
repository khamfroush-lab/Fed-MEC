import struct

BUFFER = 4

def recvall(sock, buffer: int=BUFFER):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < buffer:
        packet = sock.recv(buffer - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock, buffer: int=BUFFER):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, buffer)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    return recvall(sock, msglen)