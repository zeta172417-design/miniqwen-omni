import importlib.util,io,json,struct,sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data_pipeline'))
from sft_data_utils import json_array
from sft_splitzip import SplitZipFile
class PreparationTests(unittest.TestCase):
 def test_streaming_unicode_and_truncated_json(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'a.json';rows=[{'text':'多模态'+'x'*1100000},{'text':'a]b'}];p.write_text(json.dumps(rows,ensure_ascii=False))
   self.assertEqual(list(json_array(p)),rows)
   p.write_text('[{"a":1},')
   with self.assertRaises(ValueError):list(json_array(p))
 def test_split_archive_offsets_and_crc(self):
  b=io.BytesIO()
  with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
   z.writestr('first.bin',b'abc'*100);z.writestr('second.bin',b'def'*500)
  raw=bytearray(b.getvalue());e=raw.rfind(b'PK\x05\x06');end=list(struct.unpack_from('<4s4H2IH',raw,e));cd=end[6];cut=raw.find(b'PK\x03\x04',4)
  pos=cd
  while pos<e:
   fields=list(struct.unpack_from(zipfile.structCentralDir,raw,pos));offset=fields[18]
   if offset>=cut:fields[15]=1;fields[18]=offset-cut
   struct.pack_into(zipfile.structCentralDir,raw,pos,*fields);pos+=46+fields[12]+fields[13]+fields[14]
  end[1]=end[2]=1;end[6]=cd-cut;struct.pack_into('<4s4H2IH',raw,e,*end)
  with tempfile.TemporaryDirectory() as folder:
   a=Path(folder)/'a.z01';b=Path(folder)/'a.zip';a.write_bytes(raw[:cut]);b.write_bytes(raw[cut:])
   with SplitZipFile([a,b]) as z:
    self.assertEqual(z.read('first.bin'),b'abc'*100);self.assertEqual(z.read('second.bin'),b'def'*500)

 def test_split_zip64_directory(self):
  b=io.BytesIO()
  with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
   z.writestr('one.txt',b'one'*100);z.writestr('two.txt',b'two'*100)
  raw=bytearray(b.getvalue());e=raw.rfind(b'PK\x05\x06');end=list(struct.unpack_from('<4s4H2IH',raw,e));cd=end[6];cut=raw.find(b'PK\x03\x04',4)
  pos=cd
  while pos<e:
   fields=list(struct.unpack_from(zipfile.structCentralDir,raw,pos));offset=fields[18]
   if offset>=cut:fields[15]=1;fields[18]=offset-cut
   struct.pack_into(zipfile.structCentralDir,raw,pos,*fields);pos+=46+fields[12]+fields[13]+fields[14]
  z64=struct.pack('<4sQ2H2I4Q',b'PK\x06\x06',44,45,45,1,1,2,2,end[5],cd-cut)
  loc=struct.pack('<4sIQI',b'PK\x06\x07',1,e-cut,2)
  end[1]=end[2]=1;end[3]=end[4]=65535;end[5]=end[6]=0xffffffff
  raw=raw[:e]+z64+loc+struct.pack('<4s4H2IH',*end)
  with tempfile.TemporaryDirectory() as folder:
   a=Path(folder)/'a.z01';b=Path(folder)/'a.zip';a.write_bytes(raw[:cut]);b.write_bytes(raw[cut:])
   with SplitZipFile([a,b]) as z:
    self.assertEqual(z.read('one.txt'),b'one'*100);self.assertEqual(z.read('two.txt'),b'two'*100)

if __name__=='__main__':unittest.main()
