from .test_pipeline import Pipeline

J, K, N, d = 6, 4, 100, 3
pipeline = Pipeline(J, K, N, d)
pipeline.generate_data()
pipeline.setup_run()
pipeline.init_hGMM()


# Test percentiles computed correctly
# Test same eventinds loaded