#!/usr/bin/env python
"""Test model loading."""
import traceback

try:
    from src.models.recommend import Recommender
    r = Recommender()
    r.load()
    print(f"SUCCESS - Users: {len(r.get_all_users())}")
except Exception as e:
    traceback.print_exc()
