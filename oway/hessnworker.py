"""
SSH worker for applying the one-way wave equation hessian

@author: Joseph Jennings
@version: 2020.08.18
"""
import zmq
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
from oway.coordgeomchunk import coordgeomchunk

# Connect to socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://oas.stanford.edu:5555")

# Listen for work from server
while True:
  # Notify we are ready
  notify_server(socket)
  # Get work
  chunk = recv_zipped_pickle(socket)
  # If chunk is empty, keep listening
  if(chunk == {}):
    continue
  # If I received something, do some work
  ochunk = {}
  # Create the modeling/imaging object
  wei = coordgeomchunk(**chunk[0])
  # Do the modeling
  chunk[2]['rec'] = wei.model_data(**chunk[1])
  # Do the imaging
  ochunk['result'] = wei.image_data(**chunk[2])
  # Return other parameters if desired
  ochunk['cid']  = chunk[2]
  # Tell server this is the result
  ochunk['msg'] = "result"
  # Send back the result
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

