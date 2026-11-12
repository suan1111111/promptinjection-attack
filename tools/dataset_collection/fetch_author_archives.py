import io,zipfile,json,concurrent.futures
from fetch_candidates import get,root
def run(repo):
 try:
  for branch in ['main','master']:
   try:r=get(f'https://codeload.github.com/{repo}/zip/refs/heads/{branch}');break
   except Exception:
    if branch=='master':raise
  archive=zipfile.ZipFile(io.BytesIO(r.content))
  names=archive.namelist();key=repo.replace('/','__')
  (root/(key+'.zip')).write_bytes(r.content)
  (root/(key+'_files.json')).write_text(json.dumps(names),encoding='utf8')
  print(repo,'bytes',len(r.content),'images',[n for n in names if n.lower().endswith(('.png','.jpg','.webp'))][:50],flush=True)
  for n in names:
   if n.lower().endswith('readme.md') and n.count('/')==1:(root/(key+'_README.md')).write_bytes(archive.read(n))
 except Exception as e:print(repo,str(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(run,['trailofbits/anamorpher','uw-nsl/ArtPrompt','munzzyy/injection-fixtures']))
