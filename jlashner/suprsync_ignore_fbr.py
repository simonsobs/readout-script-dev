import socs.db.suprsync as ss

srfm = ss.SupRysyncFilesManager('/data/so/databases/suprsync.db')

session = srfm.Session()
files = session.query(ss.SupRsyncFile).filter(
    ss.SupRsyncFile.local_path.contains("full_band"),
    ss.SupRsyncFile.failed_copy_attempts > 5,
    ss.SupRsyncFile.ignore == False
).all()

print("Ignore file following files? ")
for f in files:
    print(f" - {f.local_path}")

resp = input("[y/n]: ")
if resp.lower() == "y":
    for f in files:
        f.ignore = True
    session.commit()

else:
    print("Not ignoring files")

