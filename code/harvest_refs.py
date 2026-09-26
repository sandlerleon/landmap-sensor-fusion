# -*- coding: utf-8 -*-
"""Fetch and verify every DOI reference via Crossref (abstract via Semantic Scholar where
Crossref has none), so each citation can be checked against the claim it supports.

    python harvest_refs.py
"""
import html
import io
import json
import re
import time

import requests

UA = {"User-Agent": "landmap-refcheck (mailto:sandler.leon@gmail.com)"}
DOIS = {
    "laner2019":     "10.1016/j.wasman.2019.07.007",     # NOTE: corrects a wrong DOI in the source proposal
    "danthurebandara2015": "10.1016/j.wasman.2015.01.041",
    "jones2013":     "10.1016/j.jclepro.2012.05.021",
    "vollprecht2021": "10.3390/pr9020394",
    "yi2019":        "10.1016/j.jenvman.2019.01.101",
    "zhang2022":     "10.1177/0734242x221074114",
    "deidda2022":    "10.1016/j.wasman.2022.03.007",
    "dedonno2024":   "10.1007/978-3-031-52633-6_1",
    "jansen1992":    "10.1111/j.1745-6592.1992.tb00068.x",
    "roberts2017":   "10.1111/ecog.02881",
    "ploton2020":    "10.1038/s41467-020-18321-y",
    "hijmans2012":   "10.1890/11-0826.1",
    "orynbaikyzy2022": "10.3390/rs14061493",
    "breiman2001":   "10.1023/a:1010933404324",
    "chen2016":      "10.1145/2939672.2939785",
    "niculescu2005": "10.1145/1102351.1102430",
    "huang2016":     "10.3390/en9100767",
    "luhede2025":    "10.1016/j.envsci.2025.104164",
    "rendu1976":     "10.1007/978-94-010-1470-0_28",
}


def initials(given):
    return "".join(p[0] for p in re.split(r"[\s\-.]+", given) if p)


def clean(t):
    t = html.unescape(re.sub(r"<[^>]+>", " ", t or ""))
    return re.sub(r"\s+", " ", t).strip()


def main():
    refs, abstracts = {}, {}
    for tag, doi in DOIS.items():
        m = requests.get("https://api.crossref.org/works/" + doi, headers=UA, timeout=40).json()["message"]
        title = clean(m["title"][0])
        assert not title.upper().startswith("WITHDRAWN"), (tag, title)
        au = []
        for a in m.get("author", []):
            if "family" in a:
                fam = a["family"]
                fam = fam.title() if fam.isupper() else fam
                au.append("%s %s" % (fam, initials(a.get("given", ""))))
            elif "name" in a:
                au.append(a["name"])
        cont = clean((m.get("container-title") or [""])[0])
        yr = ((m.get("published-print") or m.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        refs[tag] = {"doi": doi.lower(), "authors": au, "title": title, "container": cont,
                     "year": yr, "volume": m.get("volume"), "issue": m.get("issue"),
                     "page": m.get("page") or m.get("article-number"), "type": m.get("type"),
                     "publisher": m.get("publisher"), "event": clean((m.get("event") or {}).get("name", ""))}
        ab = clean(m.get("abstract", ""))
        if not ab:
            try:
                s = requests.get("https://api.semanticscholar.org/graph/v1/paper/DOI:" + doi,
                                 params={"fields": "abstract,tldr"}, timeout=40).json()
                ab = clean(s.get("abstract") or (s.get("tldr") or {}).get("text") or "")
            except Exception:
                pass
            time.sleep(1.1)
        abstracts[tag] = ab
        print("%-16s %s | %s | %s %s | abstract %d chars"
              % (tag, ", ".join(au[:2]), title[:60], cont[:35], yr, len(ab)))
        time.sleep(0.3)
    io.open("_refs.json", "w", encoding="utf-8").write(json.dumps(refs, indent=1, ensure_ascii=False))
    io.open("_abstracts.txt", "w", encoding="utf-8").write(
        "\n\n".join("## %s  (%s)\n%s" % (k, refs[k]["title"], v or "[no abstract available]")
                    for k, v in abstracts.items()))


if __name__ == "__main__":
    main()
