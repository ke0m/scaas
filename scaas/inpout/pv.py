"""
Functions for reading and writing files for paraview

@author: Joseph Jennings
@version: 2020.02.29
"""
import os

def write_nrrd(ofname,dat,ds=[1.0,1.0,1.0],dpath='/scratch/',endian='little'):
  """ 
  Write a .nrrd file that can be read into paraview
  
  Parameters
    ofname - the name of the output file 
    dat    - numpy array to be written
    ds     - the samplings in a list [all 1s]
    endian - endianness of the file [little]
  """
  if(not os.path.exists(dpath)):
    raise Exception("The datapath %s is not valid. Please call write_nrrd with a valid datapath"%(dpath))
  # Get output binary
  bname = ofname.split('.nrrd')[0]
  opath = dpath + bname + '.H@'
  # Write nrrd header file
  hfo = open(ofname,'w+')
  hfo.write('NRRD0001\n')
  if(len(dat.shape) == 3):
    hfo.write('dimension: 3\n')
    hfo.write('type: float\n')
    hfo.write('sizes: %d %d %d\n'%(dat.shape[0],dat.shape[1],dat.shape[2]))
    hfo.write('spacings: %f %f %f\n'%(ds[0],ds[1],ds[2]))
    hfo.write('encoding: raw\n')
    hfo.write('endian: %s\n'%(endian))
    hfo.write('data file: %s\n'%(opath))
    hfo.close()
  elif(len(dat.shape) == 2):
    hfo.write('dimension: 2\n')
    hfo.write('type: float\n')
    hfo.write('sizes: %d %d\n'%(dat.shape[1],dat.shape[0]))
    hfo.write('spacings: %f %f\n'%(ds[0],ds[1]))
    hfo.write('encoding: raw\n')
    hfo.write('endian: %s\n'%(endian))
    hfo.write('data file: %s\n'%(opath))
    hfo.close()
  with open(opath,'wb') as f:
    if(endian == 'big'):
      dat.flatten().astype('>f').tofile(f)
    elif(endian == 'little'): 
      dat.flatten().astype('<f').tofile(f)
    else:
      print("Failed to write file. Format %s not recognized\n"%(form))

