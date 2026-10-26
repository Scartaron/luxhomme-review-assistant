import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

class ReviewAssistantTests(unittest.TestCase):
    def test_csv_has_required_columns_and_valid_rows(self):
        rows = main.read_csv(Path(__file__).resolve().parents[1] / "data" / "reviews.csv")
        self.assertGreaterEqual(len(rows), 30)
        self.assertTrue(all(1 <= r["rating"] <= 5 for r in rows))

    def test_baseline_returns_allowed_labels(self):
        row = {"date": "2026-10-22", "sku": "AF-5501", "rating": 2, "text": "Шумит и сломался"}
        result = main.heuristic_label(row)
        self.assertIn(result["topic"], main.TOPICS)
        self.assertIn(result["sentiment"], {"positive", "negative", "mixed"})

    def test_html_escape(self):
        self.assertEqual(main.html_escape('<script>&"'), '&lt;script&gt;&amp;&quot;')

    def test_report_includes_count_and_topic_heading(self):
        rows = [{"date":"2026-10-22","sku":"AF-5501","rating":2,"text":"Шумит"}]
        labels = [{"topic":"Шум","sentiment":"negative","reason":"пример"}]
        report = main.build_report(rows, labels)
        self.assertIn("Всего отзывов", report)
        self.assertIn("Жалобы с примерами", report)

if __name__ == "__main__":
    unittest.main()
