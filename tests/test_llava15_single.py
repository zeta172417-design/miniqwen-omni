import io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data_pipeline'))
from prepare_llava15_single_sft import normalize
import prepare_llava15_media as media
from PIL import Image

class LlavaSingleTest(unittest.TestCase):
 def test_full_multiturn_preserved(self):
  r={'image':'coco/train2017/000000000001.jpg','conversations':[{'from':'human','value':'<image>\nDescribe it.'},{'from':'gpt','value':'A cat.'},{'from':'human','value':'What color?'},{'from':'gpt','value':'Black.'}]}
  path,m=normalize(r);self.assertEqual(len(m),4);self.assertEqual(m[2]['content'],'What color?');self.assertEqual(path,r['image'])
 def test_reject_no_image_marker_and_traversal(self):
  for image,text in [('x.jpg','Describe'),('../x.jpg','<image>'),('x.jpg','<image><image>')]:
   with self.assertRaises(ValueError):normalize({'image':image,'conversations':[{'from':'human','value':text},{'from':'gpt','value':'answer'}]})
 def test_extract_selected_only_validates_and_deletes_archive(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);raw=root/'raw';meta=root/'meta';path=root/'images.zip';b=io.BytesIO();Image.new('RGB',(4,5)).save(b,format='JPEG')
   with zipfile.ZipFile(path,'w') as z:z.writestr('train2017/1.jpg',b.getvalue());z.writestr('train2017/2.jpg',b.getvalue())
   spec={'Name':'coco.zip','Revision':'test','Size':path.stat().st_size,'Sha256':'test'}
   with patch.object(media,'RAW',raw),patch.object(media,'META',meta),patch.object(media,'R',root),patch.object(media,'fetch_ranges',return_value=path):
    result=media.extract(spec,{'coco/train2017/1.jpg'})
   self.assertEqual(result['decoded'],1);self.assertTrue(result['archive_deleted']);self.assertFalse(path.exists());self.assertTrue((raw/'images/coco/train2017/1.jpg').exists());self.assertFalse((raw/'images/coco/train2017/2.jpg').exists())

class QualityTest(unittest.TestCase):
 def test_corrupt_image_quarantined(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);raw=root/'raw';meta=root/'meta';path=root/'images.zip';b=io.BytesIO();Image.new('RGB',(4,5)).save(b,format='JPEG')
   with zipfile.ZipFile(path,'w') as z:z.writestr('1.jpg',b.getvalue());z.writestr('bad.jpg',b'broken jpeg')
   spec={'Name':'coco.zip','Revision':'test','Size':path.stat().st_size,'Sha256':'test'}
   with patch.object(media,'RAW',raw),patch.object(media,'META',meta),patch.object(media,'R',root),patch.object(media,'fetch_ranges',return_value=path):
    d=media.extract(spec,{'coco/1.jpg','coco/bad.jpg'})
   self.assertEqual(d['decoded'],1);self.assertEqual(len(d['failed_images']),1);self.assertEqual((root/d['failed_images'][0]['quarantine_path']).read_bytes(),b'broken jpeg')
 def test_quality_replacement_preserves_original_content(self):
  import pyarrow as pa
  import pyarrow.parquet as pq
  def record(i,image):return {'id':i,'image':image,'conversations':[{'from':'human','value':f'<image>\nQuestion{i}'},{'from':'gpt','value':f'Answer{i}'}]}
  with tempfile.TemporaryDirectory() as tmp:
   raw=Path(tmp);rows=[record(0,'coco/a.jpg'),record(1,'coco/bad.jpg'),record(2,'coco/b.jpg'),record(3,'coco/a.jpg')]
   pq.write_table(pa.Table.from_pylist([{'source_row':i,'record_json':json.dumps(r)} for i,r in enumerate(rows[:3])]),raw/'selected-307k.parquet');(raw/'llava_v1_5_mix665k.json').write_text(json.dumps(rows));quality={}
   with patch.object(media,'RAW',raw),patch.object(media,'META',raw/'meta'):
    got=list(media.accepted_entries({'coco/a.jpg':{},'coco/b.jpg':{}},quality,target=3))
   self.assertEqual(len(got),3);self.assertEqual(len(quality['rejected_rows']),1);self.assertEqual(got[-1]['source_row'],3);self.assertEqual(json.loads(got[-1]['record_json']),rows[3])
