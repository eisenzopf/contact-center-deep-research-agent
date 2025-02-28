#pytest tests/test_categorize.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning
#pytest tests/test_text.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning
#pytest tests/test_match.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning
#pytest tests/test_text_generator.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning
#pytest tests/test_text_generator.py::test_generate_labeled_attribute -v -m llm_debug --log-cli-level=DEBUG -W ignore::DeprecationWarning
#pytest tests/test_analyze.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning --capture=no
#pytest tests/test_recommend.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning --capture=no
#pytest tests/test_review.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning --capture=no
#pytest tests/test_text_generator.py -k "intent" -v --llm-debug --log-cli-level=DEBUG -W ignore::DeprecationWarning --capture=no
#pytest tests/test_text_generator.py::test_generate_intent_batch -v --llm-debug --log-cli-level=DEBUG -W ignore::DeprecationWarning --capture=no
pytest tests/test_categorize.py::test_consolidate_labels --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning --capture=no