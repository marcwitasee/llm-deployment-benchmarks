"""Build team_queues.csv: a lookup from Bugzilla (product, component) to a
team queue. Upload it to the volume and load it as a small Delta table that
silver joins on.

    python make_team_queues.py bugzilla_bronze_full.jsonl
"""
import csv
import json
import sys

EXCLUDE = {("Firefox", "General"), ("Core", "General")}  # bugs here can be moved by a model

# Exact component names first, then prefixes. First match wins.
EXACT = {
    "Panning and Zooming": "Graphics", "Web Painting": "Graphics",
    "SVG": "Layout & CSS", "XUL": "Layout & CSS",
    "Disability Access APIs": "Accessibility", "Disability Access": "Accessibility",
    "Keyboard Navigation": "Accessibility",
    "Web Audio": "Media & WebRTC",
    "DOM: Security": "Privacy & Security", "DOM: Web Authentication": "Privacy & Security",
    "DOM: Web Crypto": "Privacy & Security", "Data Loss Prevention": "Privacy & Security",
    "Protections UI": "Privacy & Security", "Site Permissions": "Privacy & Security",
    "about:logins": "Privacy & Security",
    "Window Management": "OS Integration", "Shell Integration": "OS Integration",
    "DLL Services": "OS Integration", "File Handling": "OS Integration",
    "New Tab Page": "New Tab & Messaging", "Messaging System": "New Tab & Messaging",
    "Nimbus Desktop Client": "New Tab & Messaging", "Remote Settings Client": "New Tab & Messaging",
    "PDF Viewer": "Built-in Tools & AI", "Screenshots": "Built-in Tools & AI",
    "Translations": "Built-in Tools & AI",
    "Firefox Accounts": "Accounts, Sync & Install", "Sync": "Accounts, Sync & Install",
    "Installer": "Accounts, Sync & Install", "Migration": "Accounts, Sync & Install",
    "Profile Backup": "Accounts, Sync & Install", "Enterprise Policies": "Accounts, Sync & Install",
    "XML": "DOM & Web APIs", "XSLT": "DOM & Web APIs",
    "Session Restore": "Browser UI",
    "MathML": "Layout & CSS", "WebVR": "Graphics", "Web Speech": "Media & WebRTC",
    "Permission Manager": "Privacy & Security", "js-ctypes": "JavaScript",
    "Fuzzing": "Platform Internals", "Sanitizers": "Platform Internals",
    "AutoConfig (Mission Control Desktop)": "Accounts, Sync & Install",
    "Hardware Abstraction Layer (HAL)": "Platform Internals",
    "Preferences: Backend": "Platform Internals",
    "SQLite and Embedded Database Bindings": "Platform Internals",
    "mozglue": "Platform Internals",
    "Spelling Checker: en-US Dictionary": "Platform Internals", "String": "Platform Internals",
}
PREFIXES = [
    ("Graphics", "Graphics"),
    ("Layout", "Layout & CSS"), ("CSS", "Layout & CSS"),
    ("Print", "Layout & CSS"),  # Printing: *, Print Preview
    ("JavaScript", "JavaScript"), ("XPConnect", "JavaScript"),
    ("Audio/Video", "Media & WebRTC"), ("WebRTC", "Media & WebRTC"),
    ("Networking", "Networking"),
    ("Privacy", "Privacy & Security"), ("Security", "Privacy & Security"),
    ("Widget", "OS Integration"),
    ("Machine Learning", "Built-in Tools & AI"),
    ("DOM", "DOM & Web APIs"), ("Storage", "DOM & Web APIs"),
    ("Gecko Profiler", "Platform Internals"), ("IPC", "Platform Internals"),
    ("XPCOM", "Platform Internals"), ("MFBT", "Platform Internals"),
    ("Memory Allocator", "Platform Internals"), ("Performance", "Platform Internals"),
    ("Cycle Collector", "Platform Internals"), ("Internationalization", "Platform Internals"),
    ("Spelling checker", "Platform Internals"), ("Find Backend", "Platform Internals"),
]


def team_for(product, component):
    if (product, component) in EXCLUDE:
        return None
    if component in EXACT:
        return EXACT[component]
    for prefix, team in PREFIXES:
        if component.startswith(prefix):
            return team
    if product == "Firefox":  # remaining front-end components: tabs, address bar, sidebar...
        return "Browser UI"
    return "UNMAPPED"


def main(path):
    pairs = set()
    for line in open(path):
        try:
            b = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written last line while the pull is still running
        pairs.add((b["product"], b["component"]))
    rows = [(p, c, team_for(p, c)) for p, c in sorted(pairs)]
    with open("team_queues.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product", "component", "team"])
        for p, c, t in rows:
            if t:
                w.writerow([p, c, t])
    unmapped = [f"{p}::{c}" for p, c, t in rows if t == "UNMAPPED"]
    print(f"wrote team_queues.csv: {len(rows)} components")
    if unmapped:
        print("UNMAPPED, add them to EXACT or PREFIXES:", unmapped)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bugzilla_bronze_full.jsonl")
