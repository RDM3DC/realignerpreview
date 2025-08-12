from realignrq.quantum.io import write_summary

def test_write_summary(tmp_path):
    path = tmp_path / "summary.csv"
    write_summary(path, {"a": 1})
    assert path.exists()
