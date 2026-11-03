def test_engine_exports_customer_segments_only_after_layer4_succeeded():
    """The CRM API serves customer_segments_latest.csv: a serving run that aborts in Layer 4 must not have rewritten it."""
    import os
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "csv_pipeline_engine.py"), encoding="utf-8").read()
    main = src[src.index('if __name__ == "__main__":'):]
    assert main.index("run_layer4(pos_file_path") < main.index("export_customer_segments_safe(df_rfm, segments)")
