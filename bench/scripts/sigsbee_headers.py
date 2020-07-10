import numpy as np
import segyio
import re
import matplotlib.pyplot as plt

def parse_text_header(segyfile):
  '''
  Format segy text header into a readable, clean dict
  '''
  raw_header = segyio.tools.wrap(segyfile.text[0])
  # Cut on C*int pattern
  cut_header = re.split(r'C ', raw_header)[1::]
  # Remove end of line return
  text_header = [x.replace('\n', ' ') for x in cut_header]
  text_header[-1] = text_header[-1][:-2]
  # Format in dict
  clean_header = {}
  i = 1
  for item in text_header:
    key = "C" + str(i).rjust(2, '0')
    i += 1
    clean_header[key] = item
  return clean_header

f = segyio.open("../oway/src/srmodmig/sigsbee2a_nfs.sgy",ignore_geometry=True)

data = f.trace.raw[:]

srcx = np.asarray(f.attributes(segyio.TraceField.SourceX),dtype='float32')
recx = np.asarray(f.attributes(segyio.TraceField.GroupX),dtype='float32')

# Convert to Km
srcx /= (3.28084*1000)
recx /= (3.28084*1000)

text_header = parse_text_header(f)
losrcx = np.min(srcx)
lorecx = np.min(recx)

idx = srcx == losrcx

print(np.sum(idx))

#plt.figure()
#plt.imshow(data[:100,:].T,cmap='gray',interpolation='sinc',vmin=-0.01,vmax=0.01)
#plt.show()



