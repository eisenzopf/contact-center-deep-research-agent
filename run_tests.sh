#pytest tests/test_categorize.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning
#pytest tests/test_text.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning
#pytest tests/test_match.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning
#pytest tests/test_text_generator.py --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning
pytest tests/test_text_generator.py::test_generate_labeled_attribute -v -m llm_debug --log-cli-level=DEBUG -W ignore::DeprecationWarning
