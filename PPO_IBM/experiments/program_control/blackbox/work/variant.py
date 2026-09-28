# python variant.py base.py out.py key=val ...  -> copy base with DEF overrides baked in (standalone file)
import sys, re
src = open(sys.argv[1]).read()
ov = {}
for kv in sys.argv[3:]:
    k, v = kv.split('='); ov[k] = float(v)
src = src.replace("        self.p = dict(self.DEF); self.p.update(params or {})",
                  f"        self.p = dict(self.DEF); self.p.update({ov!r}); self.p.update(params or {{}})")
open(sys.argv[2], 'w').write(src)
