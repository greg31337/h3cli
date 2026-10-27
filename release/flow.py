#!/usr/bin/env python3
"""Seven maintainer actions: setup-signing, configure, and five release stages."""
import argparse
import base64
from contextlib import contextmanager
import fcntl
import getpass
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

from common import ROOT, read, write, sha, digest, require, identities, verify, receipt, completed, attempt, run
import build_process
import driver

STAGES = ('build-local','test-local','build-packages','test-packages','publish')
PLATFORMS = {'macos':'macos-arm64','linux':'linux-x86_64'}
PUBLIC = {'macos':driver.MACOS, 'linux':driver.LINUX}
KEY_SERVICE = 'h3cli-release-minisign'
PACKAGE_WORKER = 'import runpy,sys;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'


def profile_defaults(path):
    if path.exists():
        row=read(path);require(row.get('schema')==2,'Unsupported release profile');return row
    old=ROOT/'outputs/release/machines.json'
    if old.exists():
        machine=read(old)
        return dict(schema=2,machines={n:machine[n] for n in PLATFORMS if n in machine},
                    signing={k:v for k,v in machine.get('macos',{}).items()
                             if k in ('secret_key','apple_identity','notary_profile')})
    host='macos' if platform.system()=='Darwin' else 'linux'
    return dict(schema=2,machines={host:{}},signing={})


def value(args, old, name, default=None, prompt=None):
    selected=getattr(args,name,None)
    if selected is None: selected=old.get(name) or default
    if not selected and prompt:
        require(sys.stdin.isatty(),prompt+'; pass --'+name.replace('_','-')+' to configure')
        selected=input(prompt+': ').strip()
    return selected


def signing_check(profile, directory, *, prompt=False):
    """Verify the key without exporting a password to configuration or logs."""
    signing=profile['signing'];secret=Path(signing['secret_key']).expanduser()
    public=Path(signing['public_key']).expanduser()
    require(secret.is_file() and public.is_file(),'Run setup-signing to provision the Minisign key')
    probe=directory/'signing-check';probe.write_text('h3cli unattended signing check\n')
    sign_file(profile,probe,prompt=prompt)
    run(['minisign','-Vm',probe,'-p',public],log=directory/'signing.log')


def sign_file(profile, path, *, prompt=False):
    signing=profile['signing'];secret=str(Path(signing['secret_key']).expanduser())
    args=['minisign','-S','-s',secret,'-m',str(path),'-t','h3cli release '+profile.get('version','setup')]
    password=''
    if platform.system()=='Darwin':
        result=subprocess.run(['security','find-generic-password','-s',KEY_SERVICE,'-a',secret,'-w'],capture_output=True,text=True)
        if result.returncode==0:password=result.stdout.rstrip('\n')
    elif shutil.which('secret-tool'):
        result=subprocess.run(['secret-tool','lookup','service',KEY_SERVICE,'key',secret],capture_output=True,text=True)
        if result.returncode==0:password=result.stdout.rstrip('\n')
    # Read stdin, never an environment variable, command argument or log.
    def sign():
        return subprocess.run(args,input=password+'\n',capture_output=True,text=True,timeout=60)
    result=sign()
    if result.returncode and prompt:
        require(sys.stdin.isatty(),'Run setup-signing in a terminal to unlock the existing Minisign key')
        password=getpass.getpass('Existing Minisign key password (stored in system credential store): ')
        result=sign()
        require(result.returncode==0,'Minisign could not unlock the existing key')
        if platform.system()=='Darwin':
            # security is intentionally not sent through the logging helper.
            stored=subprocess.run(['security','add-generic-password','-U','-s',KEY_SERVICE,'-a',secret,'-w',password],capture_output=True)
            require(stored.returncode==0,'Could not save Minisign password in Keychain')
        else:
            require(shutil.which('secret-tool'),'Install libsecret-tools during setup-signing to store the key password')
            result=subprocess.run(['secret-tool','store','--label=h3cli release signing','service',KEY_SERVICE,'key',secret],input=password,text=True)
            require(result.returncode==0,'Could not save Minisign password in the system keyring')
    require(result.returncode==0,'Minisign key is locked; run ./release/release.sh setup-signing once in a terminal')


def setup(args):
    profile=profile_defaults(args.profile)
    if args.dry_run:
        print('PLAN setup-signing: reuse GitHub, Minisign and installed Apple credentials; store an unlock password in the system keychain if needed');return
    work=attempt(ROOT/'outputs/release/setup','signing')
    missing=[n for n in ('gh','minisign') if not shutil.which(n)]
    if missing:
        run(['brew','install',*missing] if platform.system()=='Darwin' else ['sudo','apt-get','install','-y',*missing,'libsecret-tools'])
    auth=subprocess.run(['gh','auth','status','--hostname','github.com'],capture_output=True)
    if auth.returncode:run(['gh','auth','login','--hostname','github.com','--git-protocol','https','--web'])
    run(['gh','auth','setup-git','--hostname','github.com'])
    signing=profile.setdefault('signing',{})
    signing['secret_key']=str(Path(value(args,signing,'secret_key','~/.config/h3cli-release/h3cli-release.key')).expanduser().resolve())
    secret=Path(signing['secret_key']);public=secret.with_suffix('.pub')
    require(not secret.is_relative_to(ROOT),'Keep signing secrets outside the source tree')
    if not secret.exists():
        secret.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        require(not public.exists(),'Public key already exists without its secret; recover the original key')
        run(['minisign','-G','-s',secret,'-p',public])
    if not public.exists():run(['minisign','-R','-s',secret,'-p',public])
    signing['public_key']=str(public)
    if platform.system()=='Darwin':
        output=run(['security','find-identity','-v','-p','codesigning'],capture=True)
        available=re.findall(r'"(Developer ID Application: [^"]+)"',output)
        identity=value(args,signing,'apple_identity',available[0] if len(available)==1 else None)
        if identity not in available:
            print('Install the existing Developer ID Application certificate and private key in Keychain Access.\n'
                  'For a new certificate, use Xcode Settings > Accounts > Manage Certificates > Developer ID Application.\n'
                  'An Apple Developer membership and account access are required. Rerun setup-signing after installation.')
            raise ValueError('Developer ID Application signing identity is not installed')
        signing['apple_identity']=identity
        signing['notary_profile']=value(args,signing,'notary_profile','h3cli')
        history=subprocess.run(['xcrun','notarytool','history','--keychain-profile',signing['notary_profile'],'--output-format','json'],capture_output=True)
        if history.returncode:
            print('Store the Apple ID and app-specific password in the Keychain profile. Credentials stay out of configuration/logs.')
            run(['xcrun','notarytool','store-credentials',signing['notary_profile']])
    # Persist discovered settings even if an interactive unlock remains needed.
    write(args.profile,profile)
    signing_check(profile,work,prompt=True)
    print('PASS setup-signing: existing credentials ready; profile: '+str(args.profile))


def remote(machine, argv, *, log=None, supervised=None, capture=False):
    if supervised:
        argv=['python3',str(Path(supervised)/'scripts/build_process.py'),'--',*argv]
    return run(['ssh',*driver.ssh_options(machine.get('identity_file','')),machine['host'],shlex.join(list(map(str,argv)))],
               log=log,capture=capture,**({'stdin':subprocess.PIPE} if supervised else {}))


def copy(machine, src, dst, *, receive=False):
    host=machine['host']
    run(['scp','-O',*driver.ssh_options(machine.get('identity_file','')),
         host+':'+shlex.quote(str(src)) if receive else src,
         dst if receive else host+':'+shlex.quote(str(dst))])


def snapshot(profile, base):
    source=base/'source'
    if source.exists():
        head=subprocess.run(['git','-C',source,'rev-parse','HEAD'],capture_output=True,text=True)
        status=subprocess.run(['git','-C',source,'status','--porcelain'],capture_output=True,text=True)
        if head.returncode or status.returncode or head.stdout.strip()!=profile['commit'] or status.stdout:
            saved=attempt(base/'attempts','source-recovery')/'source'
            print('Preserving incomplete/changed internal snapshot: '+str(saved),flush=True)
            source.rename(saved)
    if not source.exists():
        run(['git','clone','--quiet','--no-hardlinks','--no-checkout',ROOT,source])
        run(['git','-C',source,'checkout','--quiet','--detach',profile['commit']])
    require((source/'release/flow.py').is_file(),'Selected commit predates the five-stage tooling; select a commit containing it')
    return source


def configure(args):
    old=profile_defaults(args.profile)
    target=value(args,old,'targets',list(old['machines']))
    if isinstance(target,str):target=target.split(',')
    require(target and len(set(target))==len(target) and all(t in PLATFORMS for t in target),'Targets must be macos,linux or one of them')
    commit=value(args,old,'commit','HEAD') if args.commit else 'HEAD'
    require(not commit.startswith('-'),'Commit must be a revision, not an option')
    commit=run(['git','rev-parse','--verify',commit+'^{commit}'],capture=True)
    repo=value(args,old,'repo') or driver.repo_name(run(['git','remote','get-url','origin'],capture=True))
    repo=driver.repo_name(repo)
    require(re.fullmatch(r'v[0-9][A-Za-z0-9_.-]*',args.version),'Use a version such as v0.6.0')
    machines={}
    for name in target:
        m=dict(old['machines'].get(name,{}))
        for key in ('host','root','models','model','identity_file','dependencies','dependency_cache','build_cache','jobs'):
            supplied=getattr(args,name+'_'+key,None)
            if supplied is not None:m[key]=supplied
        local=name==('macos' if platform.system()=='Darwin' else 'linux')
        if name=='macos':require(local,'macOS builds require starting the workflow on an Apple Silicon Mac')
        if name=='linux' and not local:
            if not m.get('host'):
                require(sys.stdin.isatty(),'Pass --linux-host for the selected Linux target')
                m['host']=input('Linux SSH host: ').strip()
            require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*',m['host']),'Invalid SSH host')
            if m.get('identity_file'):m['identity_file']=str(Path(m['identity_file']).expanduser().resolve())
            home=remote(m,['python3','-c','from pathlib import Path;print(Path.home())'],capture=True) if not args.dry_run else '/home/maintainer'
            m.setdefault('root',home+'/h3cli-releases')
            m.setdefault('models',home+'/models')
        else:
            m.pop('host',None);m.setdefault('root',str(ROOT/'outputs/release/hosts'/name))
            m.setdefault('models',str(ROOT/'models'))
        if not m.get('model'):
            m['model']=m['models']+'/MiniMax-H3' if local and (Path(m['models'])/'MiniMax-H3').exists() else m['models']+'/MiniMaxH3'
        m.setdefault('jobs',8)
        require(1<=m['jobs']<=32,'Jobs must be between 1 and 32')
        for k in ('root','models','model'):
            if local:m[k]=str(Path(m[k]).expanduser().resolve())
            require(m[k].startswith('/') and '\n' not in m[k] and ':' not in m[k], 'Use an absolute host path: '+k)
        m.setdefault('dependency_cache',m['root']+'/dependencies/cache')
        m.setdefault('build_cache',m['root']+'/downloads')
        m.pop('dependencies_sha256',None)
        for k in ('secret_key','apple_identity','notary_profile'):m.pop(k,None)
        machines[name]=m
    signing=dict(old.get('signing',{}))
    signing.setdefault('secret_key',str(Path('~/.config/h3cli-release/h3cli-release.key').expanduser()))
    signing.setdefault('public_key',str(Path(signing['secret_key']).with_suffix('.pub')))
    signing.setdefault('notary_profile','h3cli')
    profile=dict(schema=2,version=args.version,commit=commit,repo=repo,targets=target,machines=machines,
                 signing=signing,channel=args.channel or old.get('channel','prerelease'),offline=args.offline,
                 notes=args.notes.read_text().strip() if args.notes else '')
    changes=run(['git','status','--short'],capture=True)
    print('Selected committed source: '+commit)
    print('Uncommitted changes are excluded:\n'+changes if changes else 'Working tree matches the selected HEAD.' if commit==run(['git','rev-parse','HEAD'],capture=True) else 'Building the selected historical commit.')
    if args.dry_run:
        print('PLAN configure: '+args.version+'; '+', '.join(target)+'; '+repo+'; '+profile['channel']);return
    require(Path(signing['public_key']).is_file(),'Run setup-signing first')
    profile['public_key_sha256']=sha(signing['public_key'])
    flow=Flow(profile,args.profile)
    flow.prepare()
    require(run(['gh','api','repos/'+repo,'--jq','.permissions.push'],capture=True)=='true','GitHub repository is not writable by this account')
    signing_check(profile,flow.base)
    if 'macos' in target:
        require(signing.get('apple_identity','').startswith('Developer ID Application: '),'Run setup-signing to select the Apple identity')
        run(['xcrun','notarytool','history','--keychain-profile',signing['notary_profile'],'--output-format','json'],log=flow.base/'notary-check.log')
    for name in target:flow.worker(name,'provision')
    # public key identity is part of every stage's configuration identity.
    write(args.profile,profile)
    print('PASS configure: '+args.version+'; '+', '.join(PLATFORMS[t] for t in target)+'; '+profile['channel'])


class Flow:
    def __init__(self,profile,path):
        require(profile.get('schema')==2 and profile.get('version'),'Run configure --version first')
        self.p=profile;self.profile_path=path
        self.id=profile['version']+'-'+digest(profile)[:12]
        self.base=ROOT/'outputs/release/runs'/self.id
        self.source=self.base/'source'
        self.packages=self.base/'packages'
        self.current_log=self.base/'command.log'

    def host_base(self,name):
        m=self.p['machines'][name]
        return Path(m['root'])/self.id

    def job(self,name):
        m=self.p['machines'][name]
        local=not m.get('host')
        base=self.base/'hosts'/name if local else self.host_base(name)/'work'
        return dict(m,target=name,base=str(base),commit=self.p['commit'],offline=self.p['offline'],
                    profile_identity=digest(self.p))

    def prepare(self):
        self.base.mkdir(parents=True,exist_ok=True)
        snapshot(self.p,self.base)
        for name in self.p['targets']:
            m=self.p['machines'][name]
            if not m.get('host'):continue
            base=self.host_base(name);source=base/'source'
            remote(m,['mkdir','-p',base])
            probe=('from pathlib import Path;import sys,subprocess;p=Path(sys.argv[1]);'
                   'h=subprocess.run(["git","-C",str(p),"rev-parse","HEAD"],capture_output=True,text=True);'
                   's=subprocess.run(["git","-C",str(p),"status","--porcelain"],capture_output=True,text=True);'
                   'print(int((p/".git").is_dir() and h.returncode==s.returncode==0 and h.stdout.strip()==sys.argv[2] and not s.stdout))')
            existing=remote(m,['python3','-c',probe,source,self.p['commit']],capture=True)
            if existing=='0':
                bundle=attempt(self.base/'transfers','source')/'source.bundle'
                run(['git','-C',self.source,'bundle','create',bundle,'HEAD'])
                copy(m,bundle,base/'source.bundle')
                remote(m,['python3','-c','from pathlib import Path;import sys,tempfile;p=Path(sys.argv[1]);p.exists() and p.rename(Path(tempfile.mkdtemp(prefix="source-recovery-",dir=p.parent))/"source")',source])
                remote(m,['git','clone','--quiet','--no-checkout',base/'source.bundle',source])
                remote(m,['git','-C',source,'checkout','--quiet','--detach',self.p['commit']])
            require(remote(m,['git','-C',source,'rev-parse','HEAD'],capture=True)==self.p['commit'] and
                    not remote(m,['git','-C',source,'status','--porcelain'],capture=True), 'Remote source checkout changed')

    def worker(self,name,stage,**extra):
        job=self.job(name)|extra;path=self.base/(name+'-job.json')
        verifier=None
        if stage=='test-packages':
            # Run the current verifier against the immutable selected source.
            # Updating a check invalidates its receipt without rebuilding or
            # resigning the application under test.
            identity=sha(ROOT/'release/worker.py');job['package_verifier_sha256']=identity
            verifier=self.base/'package-verifiers'/identity/'worker.py'
            verifier.parent.mkdir(parents=True,exist_ok=True)
            if not verifier.exists():shutil.copy2(ROOT/'release/worker.py',verifier)
        write(path,job)
        m=self.p['machines'][name];log=attempt(self.base/'logs',stage+'-'+name)/'worker.log';self.current_log=log
        print(name+': '+stage,flush=True)
        if m.get('host'):
            base=self.host_base(name);source=base/'source'
            copy(m,path,base/'job.json')
            if verifier:
                script=base/('package-verifier-'+job['package_verifier_sha256']+'.py');copy(m,verifier,script)
                command=['python3','-c',PACKAGE_WORKER,source/'release',script,stage,'--job',base/'job.json']
            else:command=['python3',source/'release/worker.py',stage,'--job',base/'job.json']
            remote(m,command,log=log,supervised=source)
            if stage=='provision':return None
            local=self.base/(name+'-'+stage+'.json')
            copy(m,Path(job['base'])/(stage+'.json'),local,receive=True)
            return read(local)
        command=[sys.executable,'-c',PACKAGE_WORKER,self.source/'release',verifier,stage,'--job',path] if verifier else [sys.executable,self.source/'release/worker.py',stage,'--job',path]
        run(command,cwd=self.source,log=log)
        return None if stage=='provision' else read(Path(job['base'])/(stage+'.json'))

    def inputs(self,stage):
        result=dict(profile=self.p)
        for previous in STAGES[:STAGES.index(stage)]:
            marker=self.base/(previous+'.json')
            require(marker.exists(),'Run ./release/release.sh '+previous+' first')
            row=read(marker)
            require(row.get('passed') is True,'Previous stage failed: '+previous)
            require(row['inputs']==digest(self.inputs(previous)),'Stale '+previous+' result; rerun '+previous)
            verify(row['files']);result[previous]=sha(marker)
        return result

    def files(self):
        return [p for p in self.packages.iterdir() if p.is_file()]

    def verify_packages(self,directory=None):
        folder=directory or self.packages
        expected={n for t in self.p['targets'] for n in PUBLIC[t]}|{'release-info.txt','h3cli-release.pub','SHA256SUMS','SHA256SUMS.minisig'}
        require({p.name for p in folder.iterdir()}==expected,'Unexpected or missing release assets')
        driver.verify_checksums(folder,'SHA256SUMS',expected-{'SHA256SUMS','SHA256SUMS.minisig'})
        require(sha(folder/'h3cli-release.pub')==self.p['public_key_sha256'],'Signing public key changed')
        run(['minisign','-Vm',folder/'SHA256SUMS','-p',self.p['signing']['public_key']],log=self.base/'checksum-verification.log')
        require((folder/'release-info.txt').read_text()==self.info(),'Package release identity changed')

    def info(self):
        return ''.join(f'H3_RELEASE_{k.upper()}={self.p[v]}\n' for k,v in [('repo','repo'),('tag','version'),('commit','commit')])

    def apple_verify(self,directory):
        r=object.__new__(driver.Runner)
        r.c=self.p['signing'];r.dry=False;r.root=self.source;r.base=self.base
        # codesign --display writes the identity to stderr, as does the legacy
        # verifier's combined-output command interface.
        r.command=lambda argv,**kw:run(argv,log=None if kw.get('capture') else self.base/'apple-verification.log',
                                      capture=kw.get('capture',False),**({'stderr':subprocess.STDOUT} if kw.get('capture') else {}))
        r.apple_verify(Path(directory))

    def linux_noexec(self,binary,expected):
        helper=ROOT/'release/noexec.py';report=self.base/'linux-noexec.json'
        machine=self.p['machines']['linux'];self.current_log=self.base/'linux-noexec.log'
        if machine.get('host'):
            base=self.host_base('linux');source=base/'source';script=base/'noexec.py';result=base/'noexec.json'
            copy(machine,helper,script)
            remote(machine,['python3',source/'release/checked.py',script,'--binary',binary,'--sha256',expected,
                            '--package-module',source/'scripts/linux/package.py','--report',result],
                   log=self.current_log,supervised=source)
            copy(machine,result,report,receive=True)
        else:
            run([sys.executable,helper,'--binary',binary,'--sha256',expected,
                 '--package-module',self.source/'scripts/linux/package.py','--report',report],log=self.current_log)
        row=read(report)
        require(row.get('passed') is True and row['binary_sha256']==expected and row['script_sha256']==sha(helper),
                'Final Linux launcher noexec check did not match the tested artifact/verifier')
        return [report,helper]

    def build_packages(self,inputs):
        marker=self.base/'build-packages.json'
        row=completed(marker,inputs)
        if row:self.verify_packages();return row
        outputs={}
        for name in self.p['targets']:
            outputs[name]=self.worker(name,'build-packages')['result']
        if 'macos' in outputs:
            unsigned=Path(outputs['macos']['directory'])
            # Signing retry state is keyed by the unsigned bytes and identity.
            signing_id=digest(dict(build=sha(unsigned/'build.json'),signing=self.p['signing']))
            signed=self.base/'signing'/signing_id
            run([sys.executable,self.source/'scripts/macos/release.py','--input',unsigned,'--output',signed,
                 '--identity',self.p['signing']['apple_identity'],'--keychain-profile',self.p['signing']['notary_profile']],log=self.base/'signing.log')
            self.apple_verify(signed);outputs['macos']['directory']=str(signed)
            outputs['macos']['binary']=str(signed/'h3cli-macos-arm64')
        stage=attempt(self.base,'packages-attempt')
        for name,item in outputs.items():
            m=self.p['machines'][name]
            for filename in PUBLIC[name]:
                src=Path(item['directory'])/filename
                if m.get('host'):copy(m,src,stage/filename,receive=True)
                else:shutil.copy2(src,stage/filename)
        shutil.copy2(self.p['signing']['public_key'],stage/'h3cli-release.pub')
        (stage/'release-info.txt').write_text(self.info())
        require(all(p.stat().st_size<2**31 for p in stage.iterdir()),'GitHub release assets must be smaller than 2 GiB')
        (stage/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(stage.iterdir())))
        sign_file(self.p,stage/'SHA256SUMS')
        self.verify_packages(stage)
        if self.packages.exists():self.packages.rename(attempt(self.base,'superseded')/'packages')
        stage.rename(self.packages)
        return receipt(marker,inputs,outputs,self.files())

    def gh(self,*argv,capture=False):
        return run(['gh',*argv,'--repo',self.p['repo']],capture=capture)

    def remote_release(self):
        r=subprocess.run(['gh','api','repos/'+self.p['repo']+'/releases/tags/'+self.p['version']],capture_output=True,text=True)
        if r.returncode:
            require('HTTP 404' in r.stderr,'Cannot inspect GitHub release: '+r.stderr.strip())
            # The tag endpoint can omit an existing draft. The authenticated
            # release listing includes drafts; check it before creating one.
            r=subprocess.run(['gh','api','--paginate','--slurp','repos/'+self.p['repo']+'/releases?per_page=100'],capture_output=True,text=True)
            require(r.returncode==0,'Cannot inspect GitHub drafts: '+r.stderr.strip())
            matches=[release for page in json.loads(r.stdout) for release in page
                     if release['tag_name']==self.p['version']]
            require(len(matches)<=1,'Multiple GitHub releases use the selected tag')
            return matches[0] if matches else None
        return json.loads(r.stdout)

    def tag(self,create=False):
        url='https://github.com/'+self.p['repo']+'.git';ref='refs/tags/'+self.p['version']
        output=run(['git','ls-remote',url,ref,ref+'^{}'],capture=True)
        refs={name:commit for commit,name in (line.split() for line in output.splitlines())}
        if not refs and create:
            # Only the configured commit and its ancestors/tag go to GitHub.
            run(['git','-C',self.source,'push',url,self.p['commit']+':'+ref])
            return self.tag()
        require(refs.get(ref+'^{}',refs.get(ref))==self.p['commit'],'Remote tag missing or points to another commit')

    def notes(self):
        rows=read(self.base/'test-packages.json')['result']
        commits=run(['git','-C',self.source,'log','-8','--format=- %s (%h)',self.p['commit']],capture=True)
        lines=[f"# h3cli {self.p['version']}",'',self.p['notes'] or commits,'',
               'Source commit: `'+self.p['commit']+'`.','',
               'Downloads: '+', '.join('`h3cli-'+PLATFORMS[t]+'`' for t in self.p['targets'])+'.',
               'Corresponding sources, licenses and signed SHA-256 checksums are included. Model weights are downloaded separately.','',
               'Automated validation passed for the native builds and these exact release assets:',
               '- Host/API and release tooling tests; bounded real-model rendering, state, media and HTTP server tests.']
        if 'linux' in rows:lines.append('- Complete 204-output CUDA reference gates for the native build and final Linux executable; current CUDA feature/lifecycle suite; downloaded Linux executable rendering.')
        if 'macos' in rows:lines.append('- Developer ID signatures, inner/outer Apple notarization, stapled DMG, mounted DMG executable and isolated-cache Metal rendering.')
        lines+=['','Tested systems:']+['- '+PLATFORMS[t]+': '+r['result'].get('system','native configured host') for t,r in rows.items()]
        lines+=['','Scope: existing maintainer accounts on the listed hosts. Fresh installations/accounts, browser quarantine, broader OS/GPU combinations and human visual review were not acceptance gates. This experimental build is not claimed to be tested in those environments.','']
        path=self.base/'release-notes.md';path.write_text('\n'.join(lines));return path

    def download_verify(self,remote,allow_partial=False):
        expected={p.name for p in self.files()};names=[a['name'] for a in remote['assets']]
        require(len(names)==len(set(names)) and set(names)<=expected,'Unexpected or duplicate GitHub assets')
        unfinished=[a for a in remote['assets'] if a.get('state','uploaded')!='uploaded']
        for asset in unfinished:
            # GitHub documents empty "starter" assets after failed uploads.
            # Only these placeholders in our matching draft may be discarded.
            require(allow_partial and remote['draft'] and asset.get('state')=='starter' and asset.get('size')==0,
                    'Incomplete or conflicting GitHub asset: '+asset['name'])
            current=self.remote_release()
            require(current and current['draft'] and current['body']==remote['body'] and
                    asset in current['assets'],'Draft changed while resuming an upload')
            run(['gh','api','--method','DELETE','repos/'+self.p['repo']+'/releases/assets/'+str(asset['id'])])
            names.remove(asset['name'])
        if not allow_partial:require(set(names)==expected,'GitHub asset list is incomplete')
        directory=attempt(self.base/'downloads','github')
        if names:self.gh('release','download',self.p['version'],'--dir',directory)
        require({p.name for p in directory.iterdir()}==set(names),'Downloaded asset list differs')
        for name in names:require(sha(directory/name)==sha(self.packages/name),'Remote asset bytes conflict: '+name)
        if not allow_partial:self.verify_packages(directory)
        return directory,set(names)

    def publish(self,inputs):
        self.verify_packages();notes=self.notes()
        remote=self.remote_release()
        if remote:
            self.tag()
            require(remote['prerelease']==(self.p['channel']=='prerelease'),'Existing release channel conflicts')
            require(remote['body']==notes.read_text(),'Existing release notes/identity conflict')
            directory,names=self.download_verify(remote,allow_partial=remote['draft'])
        else:
            self.tag(create=True)
            self.gh('release','create',self.p['version'],'--verify-tag','--draft','--title','h3cli '+self.p['version'],
                    '--notes-file',notes,*(['--prerelease'] if self.p['channel']=='prerelease' else []))
            names=set()
        if not remote or remote['draft']:
            for p in self.files():
                if p.name not in names:self.gh('release','upload',self.p['version'],p)
            remote=self.remote_release();require(remote and remote['draft'],'Draft changed while uploading')
            directory,_=self.download_verify(remote)
        if 'macos' in self.p['targets']:self.apple_verify(directory)
        if 'linux' in self.p['targets']:
            name='h3cli-linux-x86_64';m=self.p['machines']['linux']
            if m.get('host'):
                dest=self.host_base('linux')/'downloaded-h3cli'
                copy(m,directory/name,dest)
                remote_cmd=['chmod','755',dest];remote_fn=globals()['remote'];remote_fn(m,remote_cmd)
            else:dest=directory/name;dest.chmod(0o755)
            self.worker('linux','downloaded',downloaded=str(dest),downloaded_sha256=sha(directory/name))
        self.tag()
        if remote['draft']:
            self.gh('release','edit',self.p['version'],'--draft=false','--latest='+('true' if self.p['channel']=='stable' else 'false'))
        final=self.remote_release()
        require(final and not final['draft'] and final['prerelease']==(self.p['channel']=='prerelease'), 'GitHub did not confirm publication')
        require(final['body']==notes.read_text(),'Public release notes changed')
        self.tag();self.download_verify(final)
        return receipt(self.base/'publish.json',inputs,dict(url=final['html_url']),self.files()+[notes])

    def execute(self,stage):
        self.prepare();inputs=self.inputs(stage)
        if stage=='build-packages':row=self.build_packages(inputs)
        elif stage=='publish':row=self.publish(inputs)
        else:
            rows={};files=[]
            packages=read(self.base/'build-packages.json')['result'] if stage=='test-packages' else {}
            if packages:
                self.verify_packages();files+=self.files()+[ROOT/'release/worker.py']
                if 'macos' in packages:self.apple_verify(self.packages)
            for name in self.p['targets']:
                extra={'signed':packages['macos']['directory']} if packages and name=='macos' else {}
                rows[name]=self.worker(name,stage,**extra)
                if stage=='test-packages' and name=='linux':
                    files+=self.linux_noexec(packages[name]['binary'],rows[name]['result']['binary_sha256'])
                # Remote receipts include exact binary and report hashes. Their
                # originals are rechecked by the remote worker on every retry.
                if self.p['machines'][name].get('host'):files.append(self.base/(name+'-'+stage+'.json'))
                else:files.extend(rows[name]['files'])
            marker=self.base/(stage+'.json')
            old=completed(marker,inputs)
            # Keep completed receipt identities stable on a verified retry;
            # otherwise merely rerunning build-local would stale later stages.
            row=old if old and old['result']==rows and set(old['files'])=={str(Path(f).resolve()) for f in files} else receipt(marker,inputs,rows,files)
        if stage=='build-local':detail=', '.join(PLATFORMS[t] for t in self.p['targets'])+'; binaries: '+', '.join(t+': '+r['result']['binary'] for t,r in row['result'].items())
        elif stage in ('test-local','test-packages'):detail='all required tests passed; report: '+str(self.base/(stage+'.json'))
        elif stage=='build-packages':detail='signed release packages ready; packages: '+str(self.packages)
        else:detail=row['result']['url']
        print('PASS '+stage+': '+detail,flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    def command(name,help):
        c=sub.add_parser(name,help=help)
        c.add_argument('--profile',type=Path,default=ROOT/'outputs/release/profile.json')
        c.add_argument('--dry-run',action='store_true',help='Print PLAN without builds, network, credentials or writes')
        return c
    c=command('setup-signing','One-time GitHub/Minisign/Apple credential setup; no release config required')
    for name in ('secret-key','apple-identity','notary-profile'):c.add_argument('--'+name)
    c=command('configure','Select version/source/hosts; verify credentials and provision missing tools/models')
    c.add_argument('--version',required=True);c.add_argument('--commit');c.add_argument('--repo')
    c.add_argument('--targets',help='macos,linux (default: remembered selected hosts)')
    c.add_argument('--channel',choices=('prerelease','stable'));c.add_argument('--notes',type=Path)
    c.add_argument('--offline',action='store_true')
    for host in PLATFORMS:
        for name in ('root','models','model','dependencies','dependency-cache','build-cache'):
            c.add_argument('--'+host+'-'+name)
        c.add_argument('--'+host+'-jobs',type=int)
    c.add_argument('--linux-host');c.add_argument('--linux-identity-file')
    for name in STAGES:command(name,{
        'build-local':'Build native binaries/test helpers on all selected hosts',
        'test-local':'Test recorded native builds, including the full CUDA reference gate',
        'build-packages':'Package, sign, notarize, staple and collect final deliverables',
        'test-packages':'Test exact final packages, signatures, rendering and numerical output',
        'publish':'Upload, download/verify and publish on GitHub in one invocation'}[name])
    return p


def main(argv=None):
    original=list(sys.argv[1:] if argv is None else argv)
    args=parser().parse_args(original);flow=None
    try:
        if args.command=='setup-signing':setup(args)
        elif args.command=='configure':configure(args)
        else:
            profile=read(args.profile);flow=Flow(profile,args.profile)
            if args.dry_run:
                print('PLAN '+args.command+': '+profile['version']+' from '+profile['commit']+'; '+', '.join(profile['targets']))
                return 0
            flow.base.mkdir(parents=True,exist_ok=True)
            with (flow.base/'command.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with build_process.cancellation():flow.execute(args.command)
        return 0
    except (KeyboardInterrupt,Exception) as exc:
        cancelled=isinstance(exc,KeyboardInterrupt)
        reason='cancelled; owned workers stopped' if cancelled else str(exc)
        log='none (dry run)'
        if not args.dry_run:
            try:
                log=flow.current_log if flow else attempt(ROOT/'outputs/release/logs',args.command+'-failed')/'error.log'
                Path(log).parent.mkdir(parents=True,exist_ok=True)
                with Path(log).open('a') as output:output.write('FAIL '+args.command+': '+reason+'\n')
            except OSError as error:
                log='unavailable: '+str(error)
        retry='./release/release.sh '+shlex.join(original)
        print('FAIL '+args.command+': '+reason+'; log: '+str(log)+'; retry: '+retry,file=sys.stderr,flush=True)
        return 130 if cancelled else 1


if __name__=='__main__':sys.exit(main())
