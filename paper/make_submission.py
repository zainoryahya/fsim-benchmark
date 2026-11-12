"""Build the self-contained submission folder kbs_submission/ from main_kbs.tex and supplementary.tex.

* cross-references between the two documents (\\ref{S-...} in the main text, \\ref{M-...} in the supplement) are
  replaced by their resolved numbers from the .aux files, and xr-hyper is removed;
* every \\tabinput{tables/...} is replaced by the table body, so no sub-folders are needed;
* figures are renamed Figure_1.pdf, Figure_2.pdf in the order they appear.
Run after compiling both documents (so that main_kbs.aux and supplementary.aux are current).
"""
import os, re, shutil

SRC, OUT = os.path.dirname(os.path.abspath(__file__)), None
OUT = os.path.join(SRC, "kbs_submission")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)


def labels(aux):
    out = {}
    for line in open(os.path.join(SRC, aux)):
        m = re.match(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def flatten(tex, prefix, lab):
    tex = re.sub(r"\\usepackage\{xr-hyper\}\n", "", tex)
    tex = re.sub(r"\\externaldocument\[[A-Z]-\]\{[^}]*\}\n", "", tex)

    def ref(m):
        key = m.group(1)
        if key not in lab:
            raise KeyError(f"unresolved cross-reference {prefix}{key}")
        return lab[key]
    tex = re.sub(r"\\ref\{" + prefix + r"([^}]*)\}", ref, tex)
    tex = re.sub(r"\\tabinput\{([^}]*)\}", lambda m: open(os.path.join(SRC, m.group(1))).read().rstrip("\n"), tex)
    return tex


main = flatten(open(os.path.join(SRC, "main_kbs.tex")).read(), "S-", labels("supplementary.aux"))
figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", main)
for i, f in enumerate(figs, 1):
    shutil.copy(os.path.join(SRC, f), os.path.join(OUT, f"Figure_{i}.pdf"))
    main = main.replace("{" + f + "}", "{" + f"Figure_{i}.pdf" + "}")
open(os.path.join(OUT, "main_kbs.tex"), "w").write(main)
supp = flatten(open(os.path.join(SRC, "supplementary.tex")).read(), "M-", labels("main_kbs.aux"))
open(os.path.join(OUT, "supplementary.tex"), "w").write(supp)
for f in ("refs.bib", "highlights.docx"):
    if os.path.exists(os.path.join(SRC, f)):
        shutil.copy(os.path.join(SRC, f), OUT)
print("figures:", figs)
