import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data_pipeline'))
from sft_range_download import fetch_ranges
class Response:
 def __init__(self,data,start,end):self.data=data[start:end+1];self.status_code=206;self.headers={'Content-Range':f'bytes {start}-{end}/{len(data)}'}
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def raise_for_status(self):pass
 def iter_content(self,n):yield self.data
class Session:
 def __init__(self,data):self.data=data
 def get(self,url,headers,**kw):start,end=map(int,headers['Range'].split('=')[1].split('-'));return Response(self.data,start,end)
 def close(self):pass
class RangeTest(unittest.TestCase):
 def test_retains_sequential_prefix_and_checks_full_hash(self):
  data=bytes(range(256))*4
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'file.zip';p.with_name('file.zip.partial').write_bytes(data[:150])
   with patch('sft_range_download.requests.Session',side_effect=lambda:Session(data)):
    fetch_ranges('test',p,len(data),hashlib.sha256(data).hexdigest(),workers=3,chunk_size=64)
   self.assertEqual(p.read_bytes(),data);self.assertFalse(p.with_name('file.zip.ranges.json').exists());self.assertTrue(p.with_name('file.zip.download.json').exists())
