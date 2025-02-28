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
#pytest tests/test_categorize.py::test_consolidate_labels --llm-debug -v --log-cli-level=DEBUG -W ignore::DeprecationWarning --capture=no

python ./examples/group_intents.py --db /Users/jonathan/Documents/Work/discourse_ai/Research/corpora/banking_2025/db/standard_charter_bank.db --min-count 20 --debug --max-groups 100 --reduction-factor 0.3 --force-target --show-all