from socs.db import suprsync as ss
import os
import datetime
import copy
from pprint import pprint
from dataclasses import dataclass
from typing import List

db_path = '/data/so/databases/suprsync.db'
timestream_dir = '/data/so/timestreams'
srfm = ss.SupRsyncFilesManager(db_path)

min_ctime = 1701200000

@dataclass
class File:
    path: str
    stream_id: str
    timestamp: float

    @property
    def remote_path(self):
        return os.path.join(
            f"{str(int(self.timestamp)):.5}",  # 5 ctime digits
            self.stream_id,
            os.path.basename(self.path)
        )

def get_all_registered_files():
    with srfm.Session().begin() as session:
        session = srfm.Session()
        all_files = session.query(ss.SupRsyncFile).all()
        return {f.local_path for f in all_files}

def get_all_timestream_files() -> List[File]:
    fs = []
    for root, _, files in os.walk(timestream_dir):
        for f in files:
            ts = int(f.split('_')[0])
            if ts < min_ctime:
                continue
            stream_id = root.split('/')[5]
            path = os.path.join(root, f)
            fs.append(File(path, stream_id, ts))
    return fs

def add_files_to_suprsync(files):
    registered_files = get_all_registered_files()
    files_to_add = []
    for f in files:
        if f.path not in registered_files:
            files_to_add.append(f)

    print(f"{len(files_to_add)} files to add:")
    for f in sorted(files_to_add, key=lambda f:f.timestamp):
        print(f"{f.path} -> {f.remote_path}")
    resp  = input("Add files? (y/n):")
    if resp.lower().startswith('y'):
        print("Adding files...")
        for f in sorted(files_to_add, key=lambda f:f.timestamp):
            print(f" - Adding {f.path}")
            srfm.add_file(f.path, f.remote_path, 'timestreams')
    else:
        print("Not adding any files")

if __name__ == '__main__':
    g3_files = get_all_timestream_files()
    add_files_to_suprsync(g3_files)
