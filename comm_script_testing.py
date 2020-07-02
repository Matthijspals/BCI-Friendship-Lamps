#!/usr/bin/env python
# coding: utf-8

# In[81]:


import socket
import threading
import pickle


def send_processed_result_of_unicorn(msg):
    message = msg.encode(FORMAT)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))





HEADER = 64
PORT = 5150
SERVER = "127.0.0.1"  ## find and enter your PUBLIC IP address, disable firewall if it doesn't work
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "quit"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    connected = True
    while connected:
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                connected = False
            pickle.dump(msg, open('./lightMSG.pk', 'wb'))
            
            # store message in a pickle file for the philips hue to access
            
            print(f"[{addr}] {msg}")
            conn.send("Thanks for Letting me know how you feel".encode(FORMAT))

    conn.close()
        

def recieve_messages_from_processod_unicorn():
    server.listen()
    print(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
