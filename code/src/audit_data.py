import hashlib
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def hid(prefix,x):return prefix+hashlib.sha256(x.encode()).hexdigest()[:20] if x else ''

def identity(s):
 if not s or s.lower() in ['nan','none']:return ('','','')
 m=Chem.MolFromSmiles(s)
 if m is None:return ('','','')
 exact=Chem.MolToSmiles(m,isomericSmiles=True)
 frags=Chem.GetMolFrags(m,asMols=True,sanitizeFrags=True)
 parent=max(frags,key=lambda f:(any(a.GetAtomicNum()==6 for a in f.GetAtoms()),f.GetNumHeavyAtoms()))
 parent=rdMolStandardize.Uncharger().uncharge(parent)
 group=Chem.MolToSmiles(parent,isomericSmiles=False)
 return exact,group,hid('CMP_',group)
