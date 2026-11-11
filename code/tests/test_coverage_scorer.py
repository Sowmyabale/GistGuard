"""Regression tests for the coverage scorer (reviewer point 3).

Run:  python tests/test_coverage_scorer.py
Each case below was a confirmed false positive/negative in the previous
substring-based scorer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract_evidence import ask_coverage, ask_coverage_all, identity_coverage

CASES = [
    # (name, fn, args, expected)
    ("IRS short name kept (was: FAIL)",
     identity_coverage, ("This is not the real IRS calling.", "irs"), True),
    ("SSA short name kept",
     identity_coverage, ("Not the Social Security Administration.", "ssa"), True),
    ("FBI short name kept",
     identity_coverage, ("Not the FBI.", "fbi"), True),
    ("generic 'bank' does NOT cover 'bank of america' (was: PASS)",
     identity_coverage, ("Contact your bank to check.", "bank of america"), False),
    ("specific brand still passes when named",
     identity_coverage, ("This is not the real Bank of America.", "bank of america"), True),
    ("'pin' substring inside 'shopping' (was: PASS)",
     ask_coverage, ("Stop shopping and hang up.", ["PIN / password / OTP"]), False),
    ("real PIN mention passes",
     ask_coverage, ("Never share your PIN with anyone.", ["PIN / password / OTP"]), True),
    ("any-coverage: one of two items is enough",
     ask_coverage, ("They want your card number.", ["card number / full card", "SSN"]), True),
    ("all-coverage: missing second item fails",
     ask_coverage_all, ("They want your card number.", ["card number / full card", "SSN"]), False),
    ("fraud department accepts generic bank reference",
     identity_coverage, ("This is not a real bank employee.", "fraud department"), True),
    ("grandchild identity via generic family phrase",
     identity_coverage, ("Someone pretending to be your grandchild.", "grandson"), True),
    ("romance profession via scam-generic phrase",
     identity_coverage, ("A fake online romantic partner.", "soldier"), True),
    ("no requested items -> vacuous pass",
     ask_coverage, ("Hang up now.", []), True),
    ("no claimed identity -> vacuous pass",
     identity_coverage, ("Hang up now.", ""), True),
]

def main():
    failed = 0
    for name, fn, args, expected in CASES:
        got = fn(*args)
        ok = got == expected
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  (got={got})")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
