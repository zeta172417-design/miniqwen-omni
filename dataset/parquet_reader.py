"""Map-style Parquet access with bounded process-local row-group caching."""
import bisect
import operator
import os
from collections import OrderedDict
import pyarrow.parquet as pq

class ParquetRows:
    def __init__(self, paths, cache_bytes=128*1024**2):
        self.paths=list(paths);self.cache_bytes=cache_bytes;self.groups=[];self.ends=[];self.row_group_ranges=[];total=0
        for path in self.paths:
            pf=pq.ParquetFile(path)
            for group in range(pf.num_row_groups):
                n=pf.metadata.row_group(group).num_rows
                if not n:continue
                self.groups.append((path,group,total));self.row_group_ranges.append((total,total+n));total+=n;self.ends.append(total)
            pf.close()
        self._pid=None;self._cache=OrderedDict();self._bytes=0;self._files=OrderedDict()
    def __len__(self):return self.ends[-1] if self.ends else 0
    def __getstate__(self):
        state=self.__dict__.copy();state.update(_pid=None,_cache=OrderedDict(),_bytes=0,_files=OrderedDict());return state
    def __getitem__(self,index):
        index=operator.index(index)
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        if self._pid!=os.getpid():
            self._cache=OrderedDict();self._files=OrderedDict();self._bytes=0;self._pid=os.getpid()
        key=bisect.bisect_right(self.ends,index)
        path,group,start=self.groups[key]
        table=self._cache.pop(key,None)
        if table is not None:self._cache[key]=table
        else:
            pf=self._files.pop(path,None)
            if pf is None:pf=pq.ParquetFile(path)
            self._files[path]=pf
            while len(self._files)>8:self._files.popitem(last=False)[1].close()
            table=pf.read_row_group(group,use_threads=False)
            # Nested list/binary columns written by the Kokoro SFT rebuild can
            # contain multiple Arrow chunks.  PyArrow cannot convert a sliced
            # chunked nested array directly to Python objects.
            table=table.combine_chunks()
            while self._cache and self._bytes+table.nbytes>self.cache_bytes:
                _,old=self._cache.popitem(last=False);self._bytes-=old.nbytes
            if table.nbytes<=self.cache_bytes:self._cache[key]=table;self._bytes+=table.nbytes
        return table.slice(index-start,1).to_pylist()[0]
