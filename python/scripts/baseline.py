#!/usr/bin/env python
"""Thin command-line entry. Shared implementation lives in utils/motiogen."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.cli import main

if __name__=='__main__':
    raise SystemExit(main())
