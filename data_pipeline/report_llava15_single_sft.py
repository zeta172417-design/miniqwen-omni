#!/usr/bin/env python3
"""Wait for selected single-image media acceptance, then record final quantities."""
import collections,json,time,os
from pathlib import Path
R=Path(__file__).resolve().parents[1];W=R/'.runtime/sft-llava15';M=R/'dataset/raw/manifests/llava15';O=R/'dataset/sft/external/llava15_single'
def main():
 os.nice(10)
 while True:
  p=W/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
  if s.get('status')=='COMPLETE_NORMALIZED_POOL_ONLY':break
  p=W/'media-status.json'
  if p.exists() and json.loads(p.read_text()).get('status')=='FAILED':raise RuntimeError(p.read_text())
  time.sleep(10)
 import pyarrow.parquet as pq
 manifest=json.loads((O/'manifest.json').read_text());selection=json.loads((M/'selection.json').read_text());audits=[json.loads(p.read_text()) for p in (M/'media-audits').glob('*.json') if p.name.endswith(('.zip.json','.tar.json'))]
 assert len(audits)==6 and all(a['status']=='PASS' for a in audits)
 counts={split:pq.ParquetFile(R/path).metadata.num_rows for split,path in manifest['data_files'].items()};assert counts==manifest['counts'] and sum(counts.values())==307000
 assert sum(a['decoded']+len(a.get('failed_images',[])) for a in audits)==207023
 leftovers=[p for p in (R/'dataset/raw/external/llava15/downloads').iterdir() if p.suffix in ('.zip','.tar','.partial')];assert not leftovers,leftovers
 payload={'status':'NORMALIZED_SINGLE_IMAGE_POOL_READY_FINAL_SFT_MIX_PENDING','counts':counts,'selected_rows':307000,'unique_original_images':manifest['unique_original_image_paths'],'initial_selected_image_paths':207023,'quality_adjustment':manifest.get('quality_adjustment',{}),'selected_source_rows':manifest['image_source_rows'],'raw_annotation_counts':selection['raw_counts'],'excluded_before_sampling':selection['excluded'],'reused_existing_images':sum(a['reused'] for a in audits),'removed_archive_GiB':sum(a['size'] for a in audits)/2**30,'retained_selected_media_GiB':sum(a['retained_bytes'] for a in audits)/2**30,'source_revision':manifest['source_revision'],'manifest':str((O/'manifest.json').relative_to(R)),'tests_passed':6,'scope':'No SFT training or change to running pretrain. Complete conversations and official text retained. Source307K before image-group train/dev split. Token budget, final cross-pool split alignment and benchmark decontamination pending.'}
 dedup=M/'storage-deduplication.json'
 if dedup.exists():
  d=json.loads(dedup.read_text());payload['storage_deduplication']={k:v for k,v in d.items() if k!='files'}
 payload['full_pool_validation']=json.loads((W/'pool-validation.json').read_text()) if (W/'pool-validation.json').exists() else None
 (R/'reports/SFT_LLAVA15_SINGLE_READY.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 lines=['# LLaVA-1.5 单图 SFT 补充准备结果','', '按M4官方建议从LLaVA-1.5 stage2 mix665k中抽取307K单图记录；不是M4仓库单独发布的单图子集，也不是LLaVA-Pretrain558K。','',f"训练池 {counts['train']:,} 条，开发池 {counts['dev']:,} 条，共307,000条，涉及{manifest['unique_original_image_paths']:,}个合格原始图片路径（初始所选207,023个）。按图片字节SHA256分组划分，组间不跨train/dev。",'', '| 图片来源 | 样本 |','|---|---:|']
 for name,n in manifest['image_source_rows'].items():lines.append(f'| {name} | {n:,} |')
 lines+=['', f"官方注释665,298条，排除40,688条无图文本和2条格式异常，剩余624,608条为抽样候选；使用固定种子42的哈希排序抽样。保留原始多轮对话，不拆成多条曝光，不改写文字。媒体不合格记录排除后，按固定哈希顺序从已有合格媒体对应的原始未选中记录中补足307K，替补来源与排除记录保存在quality-adjustment.json。",'',f"六个资源包均按固定版本下载、整包SHA256检查；ZIP成员CRC校验，所选图片逐一解码审计并记录哈希与尺寸；不能解码的图片原件保留在隔离区，排除后才构建训练池，不容忍截断数据。复用已有完全相同字节图片{payload['reused_existing_images']:,}张。验收后删除全部约{payload['removed_archive_GiB']:.2f}GiB压缩包，只保留所需媒体、原始注释、抽样记录与规范化Parquet。",'', '当前为规范化候选池，尚未应用最终SFT token长度预算；跨M4/融合池的共享图片dev归属和固定评测重叠排除仍需在构建最终混合数据时完成，不能声称已是全部SFT训练就绪。','',f"输出：`{payload['manifest']}`。原始媒体：`dataset/raw/external/llava15/images/`；官方注释及版本、抽样/媒体审计记录：`dataset/raw/manifests/llava15/`。",'', '六项处理测试通过。本次没有启动SFT训练，没有修改正在运行的pretrain模型或计划。整体SFT来源与建议比例见`reports/SFT_DATA_PLAN_V4.md`。']
 if payload.get('storage_deduplication'):
  d=payload['storage_deduplication'];lines+=['',f"对{d['new_hardlinks']:,}个字节完全相同的图片路径进行SHA256复核后，以硬链接共享存储，额外去重约{d['logical_bytes_deduplicated']/2**30:.2f}GiB；保留各来源路径与全部任务记录。不同图片字节内容共{d['unique_byte_groups']:,}份。"]
 if payload.get('full_pool_validation'):lines+=['','全部307,000条Parquet记录逐行回读检查通过，样本ID无重复、train/dev图片字节组无交集、坏图记录已排除、替补记录与来源一致，压缩包和下载片段均无残留。']
 (R/'reports/SFT_LLAVA15_SINGLE_READY.md').write_text('\n'.join(lines)+'\n')
 (W/'report-status.json').write_text(json.dumps({'status':'COMPLETE','report':'reports/SFT_LLAVA15_SINGLE_READY.md','counts':counts},indent=2)+'\n')
 print('SFT_LLAVA15_REPORT_COMPLETE',json.dumps(payload,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
