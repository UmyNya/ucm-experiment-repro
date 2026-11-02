"""Private L2 directory or explicitly requested private loopback filesystem."""
from pathlib import Path
import json
import os
import shutil
import subprocess

from common import save_json, safe_output, utc
from processes import OwnedProcess


def file_identity(path):
    st = Path(path).lstat()
    return {'device': st.st_dev, 'inode': st.st_ino}


def filesystem(path):
    info = os.statvfs(path)
    return {'total_bytes': info.f_blocks * info.f_frsize, 'free_bytes': info.f_bavail * info.f_frsize,
            'fragment_bytes': info.f_frsize, 'filesystem_id': info.f_fsid}


class PrivateStorage:
    def __init__(self, plan):
        self.plan, self.run = plan, Path(plan['run_dir'])
        self.path = safe_output(plan['l2_path'], plan['configuration'])
        self.image = self.run/'scratch/l2.ext4'
        self.created = self.mounted = False
        self.identity = None

    def command(self, name, argv):
        log = self.run/(name+'.log')
        with OwnedProcess(argv, cwd=self.run, environment=os.environ.copy(), log_path=log, name=name) as process:
            if process.wait(300):
                raise RuntimeError(f'{name} failed; see {log}. Loopback requires mount privileges inside the existing container.')

    def create(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir(mode=0o700, exist_ok=False)
        self.created, self.identity = True, file_identity(self.path)
        cap = self.plan['configuration']['capacity']['l2']
        mode, limit = cap['enforcement'], cap['filesystem_limit_gib']
        if mode in {'filesystem', 'loopback'}:
            if type(limit) is not int or limit <= 0 or limit < cap['gib']:
                raise ValueError('Physical L2 enforcement requires integer filesystem_limit_gib >= l2.gib')
        if mode == 'loopback':
            self.image.parent.mkdir(parents=True, exist_ok=True)
            with self.image.open('xb') as stream:
                stream.truncate(limit * (1 << 30))
            self.command('l2-format', ['mkfs.ext4', '-F', '-m', '0', '-E', 'lazy_itable_init=0,lazy_journal_init=0', str(self.image)])
            # The exact new mountpoint is recorded before mounting, including failure cases.
            save_json(self.run/'l2-mount-intent.json', {'image': str(self.image), 'target': str(self.path), 'identity': self.identity})
            self.command('l2-mount', ['mount', '-o', 'loop,noatime', str(self.image), str(self.path)])
            self.mounted = True
            if not os.path.ismount(self.path):
                raise RuntimeError('mount returned success but the private L2 mount is absent')
        actual = filesystem(self.path)
        if mode == 'filesystem' and actual['total_bytes'] > limit * (1 << 30):
            raise ValueError('L2 filesystem exceeds the specified hard limit. A directory on a larger filesystem is not an enforced quota.')
        self.evidence = {'created_utc': utc(), 'path': str(self.path), 'initially_empty': not any(self.path.iterdir())
                         if mode != 'loopback' else not any(p.name != 'lost+found' for p in self.path.iterdir()),
                         'enforcement': mode, 'ucm_gc_budget_gib': cap['gib'], 'filesystem': actual,
                         'filesystem_limit_gib': limit, 'exclusive_filesystem_created': mode == 'loopback',
                         'filesystem_capacity_shared_with_other_directories': mode != 'loopback',
                         'directory_identity_before_mount': self.identity}
        save_json(self.run/'l2-storage.json', self.evidence)
        return self.evidence

    def cleanup(self):
        if not self.created:
            return {'released': True, 'created': False}
        record = {'checked_utc': utc(), 'path': str(self.path), 'released': False}
        try:
            if self.path.exists():
                record['filesystem_before_cleanup'] = filesystem(self.path)
                # du records allocated blocks, including raw filesystem/metadata overhead.
                result = subprocess.run(['du', '-s', '-B1', '--', str(self.path)], capture_output=True, text=True, timeout=120)
                record['du_allocated_bytes'] = int(result.stdout.split()[0]) if result.returncode == 0 else None
            if self.mounted or os.path.ismount(self.path):
                if not self.image.is_file() or not (self.run/'l2-mount-intent.json').is_file():
                    raise RuntimeError('Unknown L2 mount; will not unmount it')
                # findmnt verifies that this mount's loop device still has our backing image.
                device = subprocess.check_output(['findmnt', '-n', '-o', 'SOURCE', '--mountpoint', str(self.path)], text=True).strip()
                backing = json.loads(subprocess.check_output(['losetup', '--json', '--list', '--output', 'NAME,BACK-FILE', device], text=True))
                loops = backing.get('loopdevices', [])
                if len(loops) != 1 or Path(loops[0]['back-file']).resolve() != self.image.resolve():
                    raise RuntimeError('Private mount ownership changed; retaining resources for inspection')
                self.command('l2-unmount', ['umount', '--', str(self.path)])
                self.mounted = False
            if self.path.exists():
                if file_identity(self.path) != self.identity or self.path.is_symlink():
                    raise RuntimeError('L2 path identity changed; refusing directory cleanup')
                if any(os.path.ismount(p) for p in self.path.rglob('*') if p.is_dir()):
                    raise RuntimeError('Nested mount under L2; refusing recursive cleanup')
                shutil.rmtree(self.path)
            if self.image.exists():
                self.image.unlink()
            record['released'] = True
            return record
        finally:
            save_json(self.run/'l2-cleanup.json', record)
