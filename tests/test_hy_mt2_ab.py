import json
import sys
from pathlib import Path

from tools import hy_mt2_ab


def test_ab_interleaves_groups_and_tracks_each_run_revision_chain(
        tmp_path, monkeypatch):
    calls = []

    class FakeTranslator:
        def __init__(self, **kwargs):
            self.top_k = kwargs["top_k"]
            self.finish_reason = "stop"

        def translate_streaming(self, source, *, on_token, **_kwargs):
            calls.append((source, self.top_k))
            on_token(source[:1], source[:1])
            return source, 1.0

    source = tmp_path / "inputs.jsonl"
    source.write_text("\n".join([
        json.dumps({"source_key": "same", "revision": 1,
                    "source_text": "alpha"}),
        json.dumps({"source_key": "same", "revision": 2,
                    "source_text": "alphabet"}),
    ]), encoding="utf-8")
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(hy_mt2_ab, "HyMT2Translator", FakeTranslator)
    monkeypatch.setattr(sys, "argv", [
        "hy_mt2_ab.py", str(source), "--url", "http://unused",
        "--runs", "2", "--output", str(output),
    ])

    hy_mt2_ab.main()

    # 每个样本/轮次交错 A/B，并轮换先后，避免 A 全部跑完再跑 B。
    assert [top_k for _source, top_k in calls[:4]] == [5, 20, 20, 5]
    results = [json.loads(line) for line in
               output.read_text(encoding="utf-8").splitlines()]
    revision_two = [item for item in results if item["revision"] == 2]
    by_group_run = {(item["group"], item["run"]): item
                    for item in revision_two}
    # 两个 run 都必须和各自 run 的 revision=1 比，而不是 run=1
    # 错拿刚完成的 run=0 revision=2 当 previous，算出 0。
    expected = hy_mt2_ab.rewrite_ratio("alpha", "alphabet")
    assert {item["rewrite_ratio"] for item in by_group_run.values()} == {expected}


def test_runtime_cli_defaults_use_ab_approved_official_sampling_params():
    source = (Path(__file__).parents[1] / "src" / "translate_stream.py").read_text(
        encoding="utf-8")
    assert 'parser.add_argument("--top-k", type=int, default=20)' in source
    assert ('parser.add_argument("--repetition-penalty", type=float, '
            'default=1.05)' in source)
