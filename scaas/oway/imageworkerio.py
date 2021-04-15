"""
SSH IO worker for imaging with one-way wave equation

@author: Joseph Jennings
@version: 2020.12.15
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
  exists,cdone = False,False
  fcntr = 0
  sep = seppy.sep()
  # Check if we have computed this chunk already
  if(os.path.exists(chunk[2]['ofname'])):
    exists = True
    # Check how many chunks have been computed
    fcntr = int(sep.from_header(chunk[2]['ofname'],['fcntr'])['fcntr'])
    if(fcntr > chunk[2]['ccntr']): cdone = True
  if(not cdone):
    # Create the imaging object
    wei = coordgeomchunk(**chunk[0])
    # Do the imaging
    timg = wei.image_data(**chunk[1])
    # Write the image to file
    if(exists):
      # Read in the file, add to it, and overwrite
      iaxes,img = sep.read_file(chunk[2]['ofname'])
      # Increment the file counter
      fcntr += 1
      img = np.ascontiguousarray(img.reshape(iaxes.n,order='F').T).astype('float32') # [nhx,nz,ny,nx]
      timg[0] += img
    sep.write_file(chunk[2]['ofname'],timg.T,os=chunk[2]['os'],ds=chunk[2]['ds'])
    sep.to_header(chunk[2]['ofname'],'fctnr=%d'%(fcntr))
  # Send back a message
  ochunk['msg']    = "result"
  ochunk['cid']    = chunk[3]
  ochunk['ofname'] = chunk[2]['ofname']
  send_zipped_pickle(socket,ochunk)
  # Receive 'thank you'
  socket.recv()

