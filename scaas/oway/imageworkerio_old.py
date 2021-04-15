"""
SSH worker for imaging with one-way wave equation

@author: Joseph Jennings
@version: 2020.08.18
"""
import os
import zmq
from comm.sendrecv import notify_server, send_zipped_pickle, recv_zipped_pickle
from oway.coordgeomchunk import coordgeomchunk
import inpout.seppy as seppy
import numpy as np

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
  # Write the image to file
  sep = seppy.sep()
  # Check if it exists
  if(os.path.exists(chunk[2]['ofname'])):
    # Read in the file, add to it, and overwrite
    iaxes,img = sep.read_file(chunk[2]['ofname'])
    img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32') # [nhx,nz,ny,nx]
    timg[0] += img
  sep.write_file(chunk[2]['ofname'],timg.T,os=chunk[2]['os'],ds=chunk[2]['ds'])
  # Send back a message
  ochunk['msg']    = "result"
  ochunk['cid']    = chunk[3]
  ochunk['ofname'] = chunk[2]['ofname']
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

