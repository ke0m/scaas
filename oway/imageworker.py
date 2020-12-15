"""
SSH worker for imaging with one-way wave equation

@author: Joseph Jennings
@version: 2020.08.18
"""
import zmq
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
from oway.coordgeomchunk import coordgeomchunk
from genutils.ptyprint import progressbar

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
  # Create the modeling object
  wei = coordgeomchunk(**chunk[0])
  # Do the modeling
  timg = wei.image_data(**chunk[1])
  # Transfer the data back
  if(len(timg.shape) > 3):
    nhx  = timg.shape[1]
    # Send over the extended image in chunks
    for ihx in progressbar(range(nhx),"transfer",verb=True):
      # Return other parameters if desired
      ochunk['cid'] = chunk[2]
      # Tell server this is the result
      ochunk['msg'] = "result"
      # Send back the result
      ochunk['idx'] = ihx
      ochunk['result'] = timg[0,ihx]
      send_zipped_pickle(socket,ochunk)
      # Receive 'thank you'
      socket.recv()
  else:
    # Return other parameters if desired
    ochunk['cid'] = chunk[2]
    # Tell server this is the result
    ochunk['msg'] = "result"
    # Send back the result
    ochunk['result'] = timg
    send_zipped_pickle(socket,ochunk)
    # Receive 'thank you'
    socket.recv()

