"""Read split ZIPs through a seekable view without a second large joined copy."""
import bisect,io,os,struct,zipfile
from pathlib import Path

class SplitReader(io.RawIOBase):
 def __init__(self,paths):
  self.handles=[Path(p).open('rb') for p in paths];self.starts=[];size=0
  for f in self.handles:self.starts.append(size);size+=os.fstat(f.fileno()).st_size
  self.size=size;self.pos=0;self.patches=[]
  self.seek(max(0,size-131072));tail=self.read();tail_start=size-len(tail)
  index=tail.rfind(b'PK\x05\x06')
  if index<0:raise ValueError('ZIP EOCD missing')
  end=list(struct.unpack_from('<4s4H2IH',tail,index));cd_disk=end[2]
  global_cd=end[6]+self.starts[cd_disk]
  end[1]=end[2]=0;end[3]=end[4]
  end[6]=min(global_cd,0xffffffff)
  self.patches.append((tail_start+index,struct.pack('<4s4H2IH',*end)))
  locindex=index-20
  if tail[locindex:locindex+4]==b'PK\x06\x07':
   zindex=tail.rfind(b'PK\x06\x06',0,locindex)
   if zindex<0:raise ValueError('ZIP64 extensible records unsupported')
   z=list(struct.unpack_from('<4sQ2H2I4Q',tail,zindex));global_cd=z[9]+self.starts[z[5]]
   z[4]=z[5]=0;z[6]=z[7];z[9]=global_cd
   self.patches.append((tail_start+zindex,struct.pack('<4sQ2H2I4Q',*z)))
   self.patches.append((tail_start+locindex,struct.pack('<4sIQI',b'PK\x06\x07',0,tail_start+zindex,1)))
  self.pos=0
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if pos<0:raise ValueError('negative seek')
  self.pos=pos;return pos
 def read(self,n=-1):
  n=max(0,self.size-self.pos) if n<0 else min(n,max(0,self.size-self.pos))
  start=self.pos;chunks=[]
  while n:
   i=bisect.bisect_right(self.starts,self.pos)-1;f=self.handles[i];f.seek(self.pos-self.starts[i]);limit=(self.starts[i+1] if i+1<len(self.starts) else self.size)-self.pos
   b=f.read(min(n,limit))
   if not b:raise OSError('Truncated split archive')
   chunks.append(b);self.pos+=len(b);n-=len(b)
  data=bytearray(b''.join(chunks))
  for at,value in self.patches:
   a=max(at,start);b=min(at+len(value),self.pos)
   if a<b:data[a-start:b-start]=value[a-at:b-at]
  return bytes(data)
 def close(self):
  for f in self.handles:f.close()
  super().close()

class SplitZipFile(zipfile.ZipFile):
 def __init__(self,paths):
  self.view=SplitReader(paths)
  super().__init__(self.view,'r')
  for item in self.filelist:item.header_offset+=self.view.starts[item.volume]
  end=self.start_dir
  for item in sorted(self.filelist,key=lambda x:x.header_offset,reverse=True):item._end_offset=end;end=item.header_offset
 def close(self):
  super().close()
  if hasattr(self,'view'):self.view.close()
