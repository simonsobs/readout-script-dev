from socs.db import suprsync as ss
import os
import datetime
import copy
from pprint import pprint
from dataclasses import dataclass

db_path = '/data/so/databases/suprsync.db'
srfm = ss.SupRsyncFilesManager(db_path)

min_ctime = 1701285680
max_ctime = min_ctime + 3600*24*5
min_dirdate = datetime.datetime.fromtimestamp(min_ctime - 24*3600*3).strftime("%Y%m%d")
max_dirdate = datetime.datetime.fromtimestamp(max_ctime + 24*3600*3).strftime("%Y%m%d")

@dataclass
class File:
    path: str
    timestamp: float
    stream_id: str
    action: str = None
    action_timestamp: int = None
    dir_type: str = 'outputs'

    @property
    def remote_path(self):
        return os.path.join(
            f"{str(int(self.timestamp)):.5}",  # 5 ctime digits
            self.stream_id,
            f"{self.action_timestamp}_{self.action}",
            self.dir_type,
            os.path.basename(self.path),
        )


def get_all_registered_files():
    with srfm.Session().begin() as session:
        session = srfm.Session()
        all_files = session.query(ss.SupRsyncFile).all()
        return {f.local_path for f in all_files}

def get_important_smurf_files(stream_id=None):
    smurf_basedir = '/data/smurf_data'
    def is_important(path):
        if path.endswith('png'):
            return False
        if path.endswith("_freq.txt"):
            return False
        if path.endswith("_mask.txt"):
            return False
        return True

    files = []
    for subdir in sorted(os.listdir(smurf_basedir)):
        if not subdir.startswith('2'):  # Directory is not a date, or is from 1000 years in the future
            continue
        
        for root, _, _files in os.walk(os.path.join(smurf_basedir, subdir)):
            for f in _files:
                try:
                    ts  = int(f.split('_')[0])
                except ValueError:
                    continue
                if ts < min_ctime or ts > max_ctime:
                    continue
                _stream_id = root.split('/')[4]

                if stream_id is not None:
                    if stream_id != _stream_id:
                        continue

                if is_important(f):
                    files.append(
                        File(os.path.join(root, f), ts, stream_id)
                    )
    return files

def get_bgmap_files(stream_id=None):
    basedir = '/data/smurf_data/bias_group_maps'
    files = []
    for root, _, _files in os.walk(basedir):
        for f in _files:
            try:
                ts  = int(f.split('_')[0])
            except ValueError:
                continue
            if ts < min_ctime or ts > max_ctime:
                continue
            path = os.path.join(root, f)
            _stream_id = root.split('/')[5]
            if stream_id is not None:
                if stream_id != _stream_id:
                    continue
            files.append(File(path, ts, stream_id))
    return files

def get_tunefiles(stream_id=None):
    basedir = '/data/smurf_data/tune'
    files = []
    for root, _, _files in os.walk(basedir):
        for f in _files:
            try:
                ts  = int(f.split('_')[0])
            except ValueError:
                continue
            if ts < min_ctime or ts > max_ctime:
                continue
            path = os.path.join(root, f)
            _stream_id = root.split('/')[4]
            if stream_id is not None:
                if stream_id != _stream_id:
                    continue
            files.append(File(path, ts, stream_id))
    return files

def get_local_files(stream_id):
    files = get_important_smurf_files(stream_id=stream_id)
    files.extend(get_tunefiles(stream_id=stream_id))
    files.extend(get_bgmap_files(stream_id=stream_id))
    return files

def determine_actions(files):
    cur_action = None
    cur_action_timestamp = None

    def set_action(f, action, action_timestamp):
        nonlocal cur_action, cur_action_timestamp
        f.action = action
        f.action_timestamp = action_timestamp
        cur_action = action
        cur_action_timestamp = action_timestamp
    
    def check_last_action(f, action, dur):
        if cur_action is None:
            return False

        if cur_action != action:
            return False
        if f.timestamp - cur_action_timestamp > dur:
            return False

        return True

    for f in sorted(files, key=lambda f:f.timestamp):
        if f.path.endswith('full_band_resp.txt'):
            set_action(f, 'full_band_resp', int(f.timestamp))
        if f.path.endswith("tracking_results.npy"):
            set_action(f, 'uxm_relock', int(f.timestamp))

        if  f.path.endswith("take_noise.npy"):
            if check_last_action(f, 'uxm_relock', 30):
                set_action(f, cur_action, cur_action_timestamp)
            else:
                set_action(f, 'take_noise', int(f.timestamp))
        
        if f.path.endswith("bg_map.npy"):
            set_action(f, 'take_bgmap', int(f.timestamp))
        
        if f.path.endswith('bias_step_analysis.npy'):
            if check_last_action(f, 'take_bgmap', 60):
                set_action(f, cur_action, cur_action_timestamp)
            else:
                set_action(f, 'take_bias_steps', int(f.timestamp))

        if f.path.endswith('iv_analysis.npy'):
            set_action(f, 'take_iv', int(f.timestamp))

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
        for f in files_to_add:
            print(f" - Adding {f.path}")
            srfm.add_file(f.path, f.remote_path, 'smurf')
    else:
        print("Not adding any files")

files = get_local_files('ufm_mv27')
determine_actions(files)
add_files_to_suprsync(files)
# pprint(sorted(files, key=lambda f:f.timestamp))


