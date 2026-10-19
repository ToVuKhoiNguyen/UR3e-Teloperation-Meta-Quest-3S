#!/usr/bin/env python3
import sys
import os

if __name__ == "__main__":
    # Ensure the workspace root is in PYTHONPATH
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Truyền cờ --record <file> xuống quest_server
    if "--record" in sys.argv:
        try:
            idx = sys.argv.index("--record")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                record_file = sys.argv[idx + 1]
                os.environ["QUEST_RECORD_FILE"] = record_file
                print(f"[Recorder] Bật chế độ ghi log JSONL: {record_file}")
        except Exception:
            pass

    from teleop_quest.main_quest import main
    main()
